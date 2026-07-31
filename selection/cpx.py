"""Cpx -- the PrecisionGD (M, V) free-atom distillation as a Coreset-CL selection
method (e_41/e_43 `cpx` arm, per-task).

Per task: m = round(fraction * n) synthetic atoms are fit by Adam on the joint
moment objective  J = lam*relF_M + (1-lam)*relF_V  (or raw J_M/J_V with
distill_absolute_mse), initialized AT the uniform coreset (atoms = drawn rows at
absorbed scale m^{-1/(2L)}, targets = closed-form Y*), with the y-solve adopted at
each snapshot when it lowers exact J.  The final atoms are rescaled to NATURAL
scale (x m^{1/(2L)}, y x m^{1/2}) so they enter the harness's CE loader as
ordinary data points; labels are hardened argmax(y + pi) (v1 -- soft labels ride
along in the payload).

Subclasses CoresetMethod directly, NOT EarlyTrain: the passed network is ignored
and never trained (EarlyTrain mutates the live CL model -- e_44 README trap #3).

Config keys (all optional):
    distill_L 2, distill_fit_steps 2000, distill_fit_lr 1e-3, distill_fit_bs 8192,
    distill_eval_every 200, distill_lambda_mv 0.5, distill_absolute_mse false,
    distill_fit_dtype "fp32", distill_n_fit 0 (0 = all task rows)

select() -> {"indices": <uniform init rows>, "synthetic": (X, y_hard, y_soft),
             "record": <fit curve + finals for the runlog>}
On a diverged fit it falls back to indices-only (the uniform coreset), loudly.
"""
import logging

import numpy as np
import torch

from selection.coresetmethod import CoresetMethod
from selection import distill_common as dc


class Cpx(CoresetMethod):
    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None,
                 device=None, task_id=None, **kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed, device,
                         task_id, **kwargs)
        self._device = device
        self.task_id = task_id
        self.L = int(args.get("distill_L", 2))
        self.fit_steps = int(args.get("distill_fit_steps", 2000))
        self.fit_lr = float(args.get("distill_fit_lr", 1e-3))
        self.fit_bs = int(args.get("distill_fit_bs", 8192))
        self.eval_every = int(args.get("distill_eval_every", 200))
        self.lam = float(args.get("distill_lambda_mv", 0.5))
        self.absolute = bool(args.get("distill_absolute_mse", False))
        self.fit_dtype = dict(fp32=torch.float32, fp64=torch.float64)[
            args.get("distill_fit_dtype", "fp32")]
        self.n_fit = int(args.get("distill_n_fit", 0))

    def select(self, **kwargs):
        L, lam, absolute = self.L, self.lam, self.absolute
        T = dc.task_tensors(self.dst_train, self._device, self.n_fit,
                            self.random_seed)
        n = T["n"]
        m = max(len(T["classes"]), min(n, self.coreset_size))
        mom = dc.moments(T, L, verbose=logging.info)

        # uniform init: m rows w/o replacement, e_43's (seed + m) draw convention
        rng = np.random.default_rng(int(self.random_seed or 0) + int(m))
        idx = torch.as_tensor(np.sort(rng.choice(n, size=int(m), replace=False)),
                              dtype=torch.long, device=self._device)
        B0 = dc.atom_scale(m, L) * T["Z64"][idx]
        _, K1, _, h = dc.joint_kernels(B0, T["Z64"], T["Y64"], L)
        Y0 = dc.ls_solve(K1, h)
        un = dc.exact_state(B0, (float(m) ** -0.5) * T["Y64"][idx], T, mom, L,
                            lam, absolute)
        logging.info(f"[cpx t{self.task_id}] n={n} m={m} uniform J={un['J']:.6e}")

        dt = self.fit_dtype
        Zt_tr = T["Z64"] if dt is torch.float64 else T["Z32"]
        Y_tr = T["Y64"] if dt is torch.float64 else T["Y32"]
        Bp = B0.to(dt).clone().requires_grad_(True)
        Yp = Y0.to(dt).clone().requires_grad_(True)
        opt = torch.optim.Adam(
            [dict(params=[Bp], lr=self.fit_lr * max(dc._rms(Bp), 1e-30)),
             dict(params=[Yp], lr=self.fit_lr * max(dc._rms(Yp), 1e-30))])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, dc.fit_sched(self.fit_steps))
        gen = torch.Generator(device="cpu").manual_seed(
            int(self.random_seed or 0) + 13)
        bs = int(min(self.fit_bs, n))

        curve = dict(step=[], J=[], J_M=[], J_V=[], relF_M=[], relF_V=[],
                     x_norm_mean=[], y_norm_mean=[], solve_adopted=[])
        status = "ok"

        def snapshot(step):
            with torch.no_grad():
                sn, K1s, hs = dc.exact_state(Bp, Yp, T, mom, L, lam, absolute,
                                             want_kernels=True)
                adopted = False
                Ystar = dc.ls_solve(K1s, hs).to(dt)
                post = dc.exact_state(Bp, Ystar, T, mom, L, lam, absolute)
                if post["J"] < sn["J"]:
                    Yp.data.copy_(Ystar)
                    sn, adopted = post, True
                sn.update(step=int(step), solve_adopted=adopted)
                for k in curve:
                    curve[k].append(sn[k])
            return sn

        snapshot(0)
        for t in range(self.fit_steps):
            ii = torch.randint(0, n, (bs,), generator=gen).to(self._device)
            loss = dc.mb_J(Bp, Yp, Zt_tr[ii], Y_tr[ii], mom, L, lam, absolute)
            lv = float(loss.detach().item())
            if not np.isfinite(lv) or abs(lv) > dc.DIVERGE_ABS:
                status = f"diverged@{t}"
                logging.info(f"[cpx t{self.task_id}] DIVERGED at step {t}: mb J={lv}")
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (t + 1) % self.eval_every == 0 or t == self.fit_steps - 1:
                sn = snapshot(t + 1)
                if (t + 1) % (10 * self.eval_every) == 0:
                    logging.info(
                        f"[cpx t{self.task_id}] step {t + 1}/{self.fit_steps} "
                        f"J={sn['J']:.6e} relF_M={sn['relF_M']:.3e} "
                        f"relF_V={sn['relF_V']:.3e}")

        record = dict(method="cpx", task=int(self.task_id or 0), n=n, m=int(m),
                      L=L, lam=lam, absolute=absolute, status=status,
                      uniform_J=un["J"], curve=curve)
        if status != "ok":
            logging.info(f"[cpx t{self.task_id}] fit diverged -> falling back to "
                         f"the uniform coreset (indices only)")
            return {"indices": T["row_idx"][idx.cpu().numpy()], "record": record}

        fin = dc.exact_state(Bp, Yp, T, mom, L, lam, absolute)
        record.update({f"final_{k}": v for k, v in fin.items()})
        logging.info(f"[cpx t{self.task_id}] {status} J={fin['J']:.6e} "
                     f"relF_M={fin['relF_M']:.3e} relF_V={fin['relF_V']:.3e} "
                     f"(uniform {un['J']:.6e})")

        # absorbed scale -> natural scale: x * m^{1/(2L)}, y * m^{1/2}
        B_nat = Bp.detach().double() * (float(m) ** (1.0 / (2 * L)))
        Y_nat = Yp.detach().double() * (float(m) ** 0.5)
        X, y_hard, y_soft = dc.pack_synthetic(B_nat, Y_nat, T)
        record["label_hist"] = np.bincount(
            y_hard, minlength=int(T["classes"].max().item()) + 1).tolist()
        return {"indices": T["row_idx"][idx.cpu().numpy()],
                "synthetic": (X, y_hard, y_soft), "record": record}
