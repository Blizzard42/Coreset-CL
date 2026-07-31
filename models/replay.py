import importlib
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.data_manager import SyntheticDataset
from utils import runlog

EPSILON = 1e-8

class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self.args["selection_method"].lower() == 'full':
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory()
            )

            self.train_loader = DataLoader(
                train_dataset, batch_size=self.args["batch_size"], shuffle=True,
                num_workers=self.args["num_workers"]
            )
        else:
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
            )

            selection_args = dict(epochs=self.args["selection_epochs"],
                                  selection_method=self.args["uncertainty"],
                                  balance=self.args["balance"],
                                  greedy=self.args["submodular_greedy"],
                                  function=self.args["submodular"]
                                  )

            module_name = 'selection.' + self.args["selection_method"].lower()
            method_module = importlib.import_module(module_name)
            selection_function = getattr(method_module, self.args["selection_method"])
            selection_method = selection_function(self._network, train_dataset, self.args, self.args["fraction"],
                                                  self.args["seed"],
                                                  self._device, self._cur_task, **selection_args)
            subset = selection_method.select()
            if subset.get("record") is not None:
                runlog.log_distill(subset["record"])
            if subset.get("synthetic") is not None:
                # distilled (x, y) points replace the real-subset dataset; the
                # rehearsal memory still comes from REAL rows (the method's
                # uniform-init indices -- the same handicap as a Random coreset)
                X_syn, y_syn, y_soft = subset["synthetic"]
                dst_subset = SyntheticDataset(X_syn, y_syn, soft_labels=y_soft)
            else:
                dst_subset = torch.utils.data.Subset(train_dataset, subset["indices"])

            self.subset_indices = subset["indices"]
            if self._cur_task > 0:
                memory_set = data_manager.get_dataset(
                    [],
                    source="train",
                    mode="train",
                    appendent=self._get_memory()
                )

                concatenated_dataset = ConcatDataset([memory_set, dst_subset])

                self.train_loader = DataLoader(
                    concatenated_dataset, batch_size=self.args["batch_size"], shuffle=True,
                    num_workers=self.args["num_workers"]
                )
            else:

                self.train_loader = DataLoader(
                    dst_subset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"]
                )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"]
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if self.args["selection_method"].lower() == 'full':
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        else:
            self.build_rehearsal_memory(data_manager, self.samples_per_class, self.subset_indices)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _make_opt_sched(self, lr, wd, milestones, gamma, epochs):
        """Optimizer/scheduler with config knobs for the MLP backbones:
        "optimizer": "sgd" (default) | "adamw"; "scheduler": "milestones"
        (default) | "cosine".  Defaults reproduce upstream exactly."""
        if self.args.get("optimizer", "sgd") == "adamw":
            optimizer = optim.AdamW(self._network.parameters(), lr=lr,
                                    weight_decay=wd)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lr,
                                  momentum=0.9, weight_decay=wd)
        if self.args.get("scheduler", "milestones") == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=epochs)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=gamma)
        return optimizer, scheduler

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer, scheduler = self._make_opt_sched(
                self.args["init_lr"], self.args["init_weight_decay"],
                self.args["init_milestones"], self.args["init_lr_decay"],
                self.args["init_epoch"])
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer, scheduler = self._make_opt_sched(
                self.args["lrate"], self.args["weight_decay"],
                self.args["milestones"], self.args["lrate_decay"],
                self.args["epochs"])
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                targets = targets.type(torch.LongTensor)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                if self.args.get("clip", 0):
                    nn.utils.clip_grad_norm_(self._network.parameters(),
                                             self.args["clip"])
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if self.args.get("dense_eval", False) or epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = None
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )
            runlog.log_epoch(self._cur_task, epoch + 1, "init",
                             losses / len(train_loader), train_acc, test_acc)

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                targets = targets.type(torch.LongTensor)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                if self.args.get("clip", 0):
                    nn.utils.clip_grad_norm_(self._network.parameters(),
                                             self.args["clip"])
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if self.args.get("dense_eval", False) or epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = None
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            runlog.log_epoch(self._cur_task, epoch + 1, "inc",
                             losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        logging.info(info)
