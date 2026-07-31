"""Kip -- the KIP dataset-distillation baseline (Nguyen-Chen-Lee, arXiv:2011.00050)
as a Coreset-CL selection method, ported from PrecisionGD e_43 kip.py.

Per task: m synthetic (x, y) pairs minimize the target-batch kernel-ridge-
regression loss  (1/bs) sum_t ||ytil_t - K(z_t,X_s)(K_ss + reg I)^{-1} y_s||^2,
reg = kip_reg * tr(K_ss)/m.  Kernel: "poly" (zeta.zeta')^L = the zMLP's
function-space kernel AND (prop.) its NTK -- the matched cell for the zmlp
backbone; "ntk" = the FC1 ReLU arccos NTK -- matched for relu_mlp.  Image init
from real rows, learned labels, class-balanced target batches, Label Solve
adopted only when it lowers the exact task-train KRR MSE.  Synthetic points live
at NATURAL scale throughout (the e_43 convention); labels hardened for the CE
pipeline (soft labels ride along).

Subclasses CoresetMethod directly (network ignored, never trained).

Config keys (optional): distill_L 2, distill_fit_steps 2000, distill_fit_lr 1e-3,
    distill_eval_every 200, distill_fit_dtype "fp32", distill_n_fit 0,
    kip_kernel "poly", kip_reg 1e-6, kip_bs 1024, kip_no_class_balance false

select() -> {"indices", "synthetic", "record"} (indices-only on divergence).
"""
import logging

import numpy as np
import torch

from selection.coresetmethod import CoresetMethod
from selection import distill_common as dc


class Kip(CoresetMethod):
    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None,
                 device=None, task_id=None, **kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed, device,
                         task_id, **kwargs)
        self._device = device
        self.task_id = task_id
        self.L = int(args.get("distill_L", 2))
        self.fit_steps = int(args.get("distill_fit_steps", 2000))
        self.fit_lr = float(args.get("distill_fit_lr", 1e-3))
        self.eval_every = int(args.get("distill_eval_every", 200))
        self.fit_dtype = dict(fp32=torch.float32, fp64=torch.float64)[
            args.get("distill_fit_dtype", "fp32")]
        self.n_fit = int(args.get("distill_n_fit", 0))
        self.kern = args.get("kip_kernel", "poly")
        self.kip_reg = float(args.get("kip_reg", 1e-6))
        self.kip_bs = int(args.get("kip_bs", 1024))
        self.balanced = not bool(args.get("kip_no_class_balance", False))

    def _sampler(self, T, gen):
        n, C = T["n"], len(T["classes"])
        if not self.balanced:
            return lambda: torch.randint(0, n, (self.kip_bs,), generator=gen).to(
                self._device)
        cls = [torch.nonzero(T["y_local"] == c, as_tuple=False).flatten().cpu()
               for c in range(C)]
        assert all(len(c) > 0 for c in cls), "a task class is empty"
        per = max(1, self.kip_bs // C)

        def draw():
            picks = [c[torch.randint(0, len(c), (per,), generator=gen)]
                     for c in cls]
            return torch.cat(picks).to(self._device)
        return draw

    def select(self, **kwargs):
        L, kern = self.L, self.kern
        T = dc.task_tensors(self.dst_train, self._device, self.n_fit,
                            self.random_seed)
        n = T["n"]
        m = max(len(T["classes"]), min(n, self.coreset_size))
        kf = dc.KFUNS[kern]

        rng = np.random.default_rng(int(self.random_seed or 0) + int(m))
        idx = torch.as_tensor(np.sort(rng.choice(n, size=int(m), replace=False)),
                              dtype=torch.long, device=self._device)
        dt = self.fit_dtype
        Zt_tr = T["Z64"] if dt is torch.float64 else T["Z32"]
        Y_tr = T["Y64"] if dt is torch.float64 else T["Y32"]
        Xp = T["Z64"][idx].to(dt).clone().requires_grad_(True)
        Yp = T["Y64"][idx].to(dt).clone().requires_grad_(True)
        un = dc.exact_krr_train(Xp, Yp, T, kern, L, self.kip_reg)
        logging.info(f"[kip {kern} t{self.task_id}] n={n} m={m} uniform-rows KRR "
                     f"train_mse={un['train_mse']:.6f} acc={un['train_acc']:.4f}")

        opt = torch.optim.Adam(
            [dict(params=[Xp], lr=self.fit_lr * max(dc._rms(Xp), 1e-30)),
             dict(params=[Yp], lr=self.fit_lr * max(dc._rms(Yp), 1e-30))])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, dc.fit_sched(self.fit_steps))
        gen = torch.Generator(device="cpu").manual_seed(
            int(self.random_seed or 0) + 13)
        draw = self._sampler(T, gen)
        eye = torch.eye(m, dtype=dt, device=self._device)

        curve = dict(step=[], mb_loss=[], krr_train_mse=[], krr_train_acc=[],
                     x_norm_mean=[], y_norm_mean=[], solve_adopted=[])
        status, last_mb = "ok", float("nan")

        def snapshot(step):
            with torch.no_grad():
                sn = dc.exact_krr_train(Xp, Yp, T, kern, L, self.kip_reg)
                adopted = False
                Ystar = dc.label_solve(Xp, T, kern, L, self.kip_reg).to(dt)
                post = dc.exact_krr_train(Xp, Ystar, T, kern, L, self.kip_reg)
                if post["train_mse"] < sn["train_mse"]:
                    Yp.data.copy_(Ystar)
                    sn, adopted = post, True
                rec = dict(step=int(step), mb_loss=float(last_mb),
                           krr_train_mse=sn["train_mse"],
                           krr_train_acc=sn["train_acc"],
                           x_norm_mean=sn["x_norm_mean"],
                           y_norm_mean=sn["y_norm_mean"], solve_adopted=adopted)
                for k in curve:
                    curve[k].append(rec[k])
            return rec

        snapshot(0)
        for t in range(self.fit_steps):
            ii = draw()
            try:
                Kss = kf(Xp, Xp, L, exact=False)
                Lc = torch.linalg.cholesky(
                    Kss + dc.krr_reg(Kss, self.kip_reg) * eye)
                sol = torch.cholesky_solve(Yp, Lc)
                pred = kf(Zt_tr[ii], Xp, L, exact=False) @ sol
                loss = ((pred - Y_tr[ii]) ** 2).sum() / float(ii.shape[0])
            except torch.linalg.LinAlgError:
                status = f"diverged@{t}"
                logging.info(f"[kip {kern} t{self.task_id}] DIVERGED at step {t}: "
                             f"cholesky failed (K_ss not PD)")
                break
            last_mb = float(loss.detach().item())
            if not np.isfinite(last_mb) or abs(last_mb) > dc.DIVERGE_ABS:
                status = f"diverged@{t}"
                logging.info(f"[kip {kern} t{self.task_id}] DIVERGED at step {t}: "
                             f"mb loss={last_mb}")
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (t + 1) % self.eval_every == 0 or t == self.fit_steps - 1:
                sn = snapshot(t + 1)
                if (t + 1) % (10 * self.eval_every) == 0:
                    logging.info(
                        f"[kip {kern} t{self.task_id}] step {t + 1}/"
                        f"{self.fit_steps} mb={sn['mb_loss']:.4f} "
                        f"krr_train_mse={sn['krr_train_mse']:.4f} "
                        f"krr_train_acc={sn['krr_train_acc']:.4f}")

        record = dict(method="kip", kernel=kern, task=int(self.task_id or 0),
                      n=n, m=int(m), L=L, kip_reg=self.kip_reg, status=status,
                      uniform_krr_train_mse=un["train_mse"],
                      uniform_krr_train_acc=un["train_acc"], curve=curve)
        if status != "ok":
            logging.info(f"[kip {kern} t{self.task_id}] fit diverged -> falling "
                         f"back to the uniform rows (indices only)")
            return {"indices": T["row_idx"][idx.cpu().numpy()], "record": record}

        fin = dc.exact_krr_train(Xp, Yp, T, kern, L, self.kip_reg)
        record.update({f"final_krr_{k}": v for k, v in fin.items()})
        logging.info(f"[kip {kern} t{self.task_id}] {status} "
                     f"krr_train_mse={fin['train_mse']:.6f} "
                     f"krr_train_acc={fin['train_acc']:.4f} "
                     f"(uniform {un['train_mse']:.6f}/{un['train_acc']:.4f})")

        X, y_hard, y_soft = dc.pack_synthetic(Xp.detach().double(),
                                              Yp.detach().double(), T)
        record["label_hist"] = np.bincount(
            y_hard, minlength=int(T["classes"].max().item()) + 1).tolist()
        return {"indices": T["row_idx"][idx.cpu().numpy()],
                "synthetic": (X, y_hard, y_soft), "record": record}
