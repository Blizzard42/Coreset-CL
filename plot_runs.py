"""plot_runs -- e_43-style debug figures from e_44 runlog JSONs.

Per-run figure (default): 4 panels + hyperparameter table
    1 train loss over cumulative epochs (log y, task boundaries as vlines)
    2 train/test accuracy over cumulative epochs (+ post-task top1 markers)
    3 distillation fit curve if present (cpx: exact J; kip: KRR train MSE)
    4 avg incremental accuracy after each task (cnn / nme)

Overlay figure (--overlay): one figure comparing many runs on panels 2+4,
colour per run, labelled by (selection_method, convnet, fraction).

    python plot_runs.py runlogs/*.json
    python plot_runs.py --overlay runlogs/smoke_mlp_*.json
Outputs land in plots/ next to this file (gitignored; transfer via ssh).
"""
import argparse
import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(_HERE, "plots")

TABLE_KEYS = ("prefix", "model_name", "convnet_type", "selection_method",
              "dataset", "init_cls", "increment", "seed", "fraction",
              "optimizer", "scheduler", "clip", "no_augment", "dense_eval",
              "init_epoch", "epochs", "init_lr", "lrate", "batch_size",
              "memory_size", "backbone_dropout", "mlp_depth",
              "distill_L", "distill_fit_steps", "distill_fit_lr",
              "distill_lambda_mv", "distill_absolute_mse", "distill_n_fit",
              "kip_kernel", "kip_reg", "kip_bs")


def load(path):
    with open(path) as f:
        r = json.load(f)
    r["_path"] = path
    return r


def run_label(r):
    c = r["config"]
    return "{}/{} {} f{}".format(c.get("selection_method"), c.get("convnet_type"),
                                 c.get("model_name"), c.get("fraction"))


def cum_axis(r):
    """epochs entries -> (x cumulative epoch, task boundary xs)."""
    xs, bounds, off, last_task = [], [], 0, None
    for e in r["epochs"]:
        if last_task is not None and e["task"] != last_task:
            off = xs[-1] if xs else 0
            bounds.append(off)
        last_task = e["task"]
        xs.append(off + e["epoch"])
    return np.asarray(xs, dtype=float), bounds


def panel_acc(ax, r, color=None, label=None, lw=1.6):
    xs, bounds = cum_axis(r)
    tr = [e["train_acc"] for e in r["epochs"]]
    te = [(np.nan if e["test_acc"] is None else e["test_acc"])
          for e in r["epochs"]]
    lab = label or run_label(r)
    ax.plot(xs, tr, color=color or "#1f77b4", lw=lw * 0.8, ls=":",
            label=f"{lab} train")
    tem = np.asarray(te, dtype=float)
    if np.isfinite(tem).any():
        fin = np.isfinite(tem)
        ax.plot(xs[fin], tem[fin], color=color or "#d62728", lw=lw,
                label=f"{lab} test (seen classes)")
    for b in bounds:
        ax.axvline(b, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("cumulative epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, lw=0.5)


def panel_task_eval(ax, r, color=None, label=None):
    ev = r.get("task_eval", [])
    if not ev:
        return
    t = [e["task"] for e in ev]
    ax.plot(t, [e["cnn_top1"] for e in ev], "o-", color=color or "#d62728",
            lw=1.6, label=f"{label or run_label(r)} cnn")
    if any(e.get("nme_top1") is not None for e in ev):
        ax.plot(t, [e.get("nme_top1") for e in ev], "s--",
                color=color or "#1f77b4", lw=1.2,
                label=f"{label or run_label(r)} nme")
    ax.set_xlabel("task")
    ax.set_ylabel("top1 on seen classes (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, lw=0.5)


def figure_single(r):
    fig = plt.figure(figsize=(19.0, 4.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.8, 0.62])

    ax = fig.add_subplot(gs[0, 0])
    xs, bounds = cum_axis(r)
    ax.plot(xs, [e["train_loss"] for e in r["epochs"]], color="#1f77b4", lw=1.4)
    for b in bounds:
        ax.axvline(b, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("cumulative epoch")
    ax.set_ylabel("train CE loss")
    ax.set_title("training loss over time", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    ax = fig.add_subplot(gs[0, 1])
    panel_acc(ax, r)
    ax.set_title("accuracy over time", fontsize=10)
    ax.legend(fontsize=6, loc="lower left", framealpha=0.9)

    ax = fig.add_subplot(gs[0, 2])
    ds = r.get("distill", [])
    for i, d in enumerate(ds):
        c = d.get("curve", {})
        if d.get("method") == "cpx" and c.get("J"):
            ax.plot(c["step"], c["J"], lw=1.4, label=f"t{d['task']} exact J")
            ax.axhline(d.get("uniform_J"), lw=0.8, ls="--", color="black",
                       alpha=0.5)
            ax.set_ylabel("exact J")
        elif d.get("method") == "kip" and c.get("krr_train_mse"):
            ax.plot(c["step"], c["krr_train_mse"], lw=1.4,
                    label=f"t{d['task']} KRR mse")
            ax.axhline(d.get("uniform_krr_train_mse"), lw=0.8, ls="--",
                       color="black", alpha=0.5)
            ax.set_ylabel("task-train KRR MSE")
    if ds:
        ax.set_yscale("log")
        ax.set_xlabel("fit step")
        st = ", ".join(sorted({d.get("status", "?") for d in ds}))
        ax.set_title(f"distillation fit ({st})", fontsize=10)
        ax.legend(fontsize=6, loc="best", framealpha=0.9)
    else:
        ax.axis("off")
        ax.set_title("no distillation record", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    ax = fig.add_subplot(gs[0, 3])
    panel_task_eval(ax, r)
    ax.set_title("post-task top1", fontsize=10)
    ax.legend(fontsize=6, loc="best", framealpha=0.9)

    axt = fig.add_subplot(gs[0, 4])
    axt.axis("off")
    c = r["config"]
    lines = [(k, c[k]) for k in TABLE_KEYS if k in c]
    axt.text(0.02, 1.0, "\n".join(f"{k:<18} {v}" for k, v in lines),
             family="monospace", fontsize=6.0, va="top",
             transform=axt.transAxes)
    fig.suptitle(run_label(r), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(
        PLOTS, os.path.basename(r["_path"]).replace(".json", ".png"))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_runs] wrote {out}")


def figure_overlay(runs, tag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8))
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for r in runs:
        col = next(colors)
        panel_acc(ax1, r, color=col, lw=1.4)
        panel_task_eval(ax2, r, color=col)
    ax1.set_title("accuracy over time", fontsize=10)
    ax2.set_title("post-task top1 (avg incremental accuracy)", fontsize=10)
    ax1.legend(fontsize=5.5, loc="lower left", framealpha=0.9)
    ax2.legend(fontsize=5.5, loc="best", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(PLOTS, f"overlay_{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_runs] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--tag", default="cmp")
    a = ap.parse_args()
    os.makedirs(PLOTS, exist_ok=True)
    runs = [load(p) for p in a.paths]
    if a.overlay:
        figure_overlay(runs, a.tag)
    else:
        for r in runs:
            figure_single(r)


if __name__ == "__main__":
    main()
