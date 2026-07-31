"""In-process per-run recorder for e_44 debug curves.

trainer starts/finishes one RUN per (_train call = one model x fraction x seed);
models append per-epoch train/test accuracy; distillation selection methods
append their fit records.  JSON is written atomically to runlogs/ inside the
repo (gitignored; transfer via ssh per repo policy).  plot_runs.py (e_44 dir)
renders the e_43-style figures from these files.
"""
import json
import os

_RUN = None
_PATH = None


def _jsonable(v):
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False


def start(args, out_dir):
    global _RUN, _PATH
    os.makedirs(out_dir, exist_ok=True)
    name = "{}_{}_{}_{}_f{}_s{}.json".format(
        args.get("prefix", "run"), args.get("model_name", "?"),
        args.get("convnet_type", "?"), args.get("selection_method", "?"),
        args.get("fraction", "?"), args.get("seed", "?"))
    _PATH = os.path.join(out_dir, name)
    _RUN = dict(config={k: v for k, v in args.items() if _jsonable(v)},
                epochs=[], distill=[], task_eval=[])


def log_epoch(task, epoch, phase, train_loss, train_acc, test_acc):
    """One training epoch.  train_acc/test_acc in [0,100] (harness convention);
    test_acc may be None when dense eval is off for that epoch."""
    if _RUN is not None:
        _RUN["epochs"].append(dict(task=int(task), epoch=int(epoch), phase=phase,
                                   train_loss=float(train_loss),
                                   train_acc=float(train_acc),
                                   test_acc=None if test_acc is None
                                   else float(test_acc)))


def log_distill(record):
    if _RUN is not None:
        _RUN["distill"].append(record)


def log_task_eval(task, cnn_top1, nme_top1, grouped=None):
    if _RUN is not None:
        _RUN["task_eval"].append(dict(task=int(task), cnn_top1=cnn_top1,
                                      nme_top1=nme_top1, grouped=grouped))


def finish():
    global _RUN, _PATH
    if _RUN is None:
        return None
    tmp = _PATH + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_RUN, f, default=float)
    os.replace(tmp, _PATH)
    p, _RUN, _PATH = _PATH, None, None
    return p
