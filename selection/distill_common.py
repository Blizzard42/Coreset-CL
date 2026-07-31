"""Shared machinery for the distillation selection methods (Cpx, Kip), ported from
PrecisionGD e_43 (coreset.py / kip.py).  The paid-for conventions kept verbatim:

* ZETA UNITS: zeta = z / sqrt(d) where z is the trained representation (here the
  harness's Normalize(ToTensor(img)), flattened -- per-channel standardization, a
  documented deviation from e_43's per-feature stats).  Every inner product is a
  plain matmul with no constant.
* fp DISCIPLINE: every reported J/MSE is fp64 at the actually-stored (fp32-rounded)
  parameters; TF32 off; negative squared errors abort loudly (PrecisionError).
* Centred one-hot targets Y = onehot - pi over the TASK's classes (local index),
  accuracy rule argmax(f + pi).
* Fit: Adam, RELATIVE lr (lr * rms(param group at init)), warmup(100)+cosine, fixed
  budget, closed-form y-solve adopted only when it lowers the exact objective.

CL adaptation: everything operates on ONE task's training data (the n rows the
harness passes to selection); moments/S constants are computed fresh per task on
those rows (no cache -- task data changes every call).
"""
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

WARMUP_FIT = 100
DIVERGE_ABS = 1e12


class PrecisionError(RuntimeError):
    """A reported squared error came out negative: a precision bug, not data."""


# ================================================================ task data
def task_tensors(dst_train, device, n_fit=0, seed=0):
    """Extract the task's training set in the TRAINED representation.

    Iterates dst_train with its own transform -- which must be the clean
    (no-augmentation) pipeline; pair distillation methods with "no_augment": true.
    -> dict(Z64 zeta fp64 [n,d], Z32, Y64 centred fp64 [n,C_task], Y32, y_local,
       classes (sorted global labels [C_task]), pi (fp64 local priors), n, d, sqd)
    n_fit > 0 subsamples the rows used for BOTH the moments and the fit."""
    loader = DataLoader(dst_train, batch_size=512, shuffle=False, num_workers=0)
    xs, ys = [], []
    for _, x, y in loader:
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(y)
    Z = torch.cat(xs).to(device)
    y = torch.cat(ys).long().to(device)
    row_idx = np.arange(len(Z))          # fit-row -> original dataset row
    if n_fit and n_fit < len(Z):
        row_idx = np.sort(
            np.random.default_rng(int(seed)).permutation(len(Z))[:int(n_fit)])
        sel = torch.as_tensor(row_idx, dtype=torch.long, device=device)
        Z, y = Z[sel], y[sel]
    classes = torch.unique(y).sort().values
    lut = torch.full((int(classes.max().item()) + 1,), -1,
                     dtype=torch.long, device=device)
    lut[classes] = torch.arange(len(classes), device=device)
    y_local = lut[y]
    C = len(classes)
    n, d = Z.shape
    sqd = float(math.sqrt(d))
    Z64 = Z.double() / sqd
    pi = torch.bincount(y_local, minlength=C).double() / float(n)
    Y64 = torch.nn.functional.one_hot(y_local, C).double() - pi
    mx = float((Z64 * Z64).sum(dim=1).max().item())
    if mx > 1e6:
        raise PrecisionError(f"max|zeta|^2 = {mx:.3e}: representation not standardized?")
    return dict(Z64=Z64, Z32=Z64.float(), Y64=Y64, Y32=Y64.float(),
                y_local=y_local, classes=classes, pi=pi, n=int(n), d=int(d),
                sqd=sqd, row_idx=row_idx)


# ================================================================ numerics
def ipow(x, k):
    """x**k for integer k >= 1 by exact-rounded multiplies."""
    k = int(k)
    assert k >= 1
    r, b = None, x
    while k:
        if k & 1:
            r = b if r is None else r * b
        k >>= 1
        if k:
            b = b * b
    return r


def dots(A, B, exact=True):
    """THE one inner-product site.  Zeta units: plain matmul, no constant."""
    if exact:
        return A.double() @ B.double().T
    return A @ B.T


def ls_solve(K, rhs, jit_rel=1e-10, escalate=6):
    """Ridge-jittered solve with x100 jitter escalation and a lstsq last resort."""
    K, rhs = K.double(), rhs.double()
    jit = float(jit_rel) * max(float(K.diagonal().mean().item()), 1e-300)
    eye = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    for _ in range(int(escalate)):
        try:
            X = torch.cholesky_solve(rhs, torch.linalg.cholesky(K + jit * eye))
            if torch.isfinite(X).all():
                return X
        except Exception:
            pass
        jit *= 100.0
    return torch.linalg.lstsq(K, rhs).solution


def _nonneg(J, parts, where, tol=1e-10):
    sc = max(abs(float(x)) for x in parts)
    if J < -tol * sc:
        raise PrecisionError(f"[{where}] J = {J:.6e} < -{tol:g}*{sc:.6e}: eval path "
                             f"not pure fp64 or S from another featurization")
    return J


def _rms(t):
    td = t.detach().double()
    return float(td.pow(2).mean().sqrt().item()) if td.numel() else 0.0


def fit_sched(steps, warmup=WARMUP_FIT):
    return lambda t: min(1.0, (t + 1) / float(warmup)) * 0.5 * (
        1.0 + math.cos(math.pi * min(t, steps) / float(steps)))


# ================================================================ (M, V) machinery
def atom_scale(m, L):
    """m^{-1/(2L)}: the absorbed uniform weight u = 1/m."""
    return float(m) ** (-1.0 / (2 * L))


def moments(T, L, verbose=print):
    """S_M, S_V of the TASK data by the O(n^2 d) fp64 pass (fresh per task)."""
    Zt, Y, n = T["Z64"], T["Y64"], T["n"]
    chunk = max(64, int(4.0e7 // max(n, 1)))
    sm = torch.zeros((), dtype=torch.float64, device=Zt.device)
    sv = torch.zeros((), dtype=torch.float64, device=Zt.device)
    for a0 in range(0, n, chunk):
        sl = slice(a0, min(a0 + chunk, n))
        DL = ipow(dots(Zt[sl], Zt), L)
        sm += (DL * DL).sum()
        sv += ((Y[sl] @ Y.T) * DL).sum()
        del DL
    out = dict(S_M=float(sm.item()) / n ** 2, S_V=float(sv.item()) / n ** 2)
    verbose(f"  [moments] n={n} S_M={out['S_M']:.6e} S_V={out['S_V']:.6e}")
    return out


def joint_kernels(B, Zt, Y, L, chunk=None):
    m, N = B.shape[0], Zt.shape[0]
    ch = int(chunk or max(256, 3.0e7 // max(m, 1)))
    K1 = ipow(dots(B, B), L)
    K2 = K1 * K1
    g2 = torch.zeros(m, dtype=torch.float64, device=B.device)
    h = torch.zeros(m, Y.shape[1], dtype=torch.float64, device=B.device)
    for n0 in range(0, N, ch):
        PL = ipow(dots(B, Zt[n0:n0 + ch]), L)
        g2 += (PL * PL).sum(dim=1)
        h += PL @ Y[n0:n0 + ch].double()
        del PL
    return K2, K1, g2 / N, h / N


def combine_J(J_M, J_V, S_M, S_V, lam, absolute):
    if absolute:
        return lam * J_M + (1.0 - lam) * J_V
    return lam * J_M / S_M + (1.0 - lam) * J_V / S_V


def exact_state(B, Yat, T, mom, L, lam, absolute, want_kernels=False):
    """THE exact fp64 evaluator of the (M, V) objective."""
    S_M, S_V = mom["S_M"], mom["S_V"]
    Bd, Yd = B.detach().double(), Yat.detach().double()
    K2, K1, g2, h = joint_kernels(Bd, T["Z64"], T["Y64"], L)
    quadM = float(K2.sum().item())
    linM = float(g2.sum().item())
    J_M = _nonneg(quadM - 2.0 * linM + S_M, (S_M, quadM, 2 * linM), "J_M")
    quadV = float(torch.einsum("ac,ab,bc->", Yd, K1, Yd).item())
    linV = float((Yd * h).sum().item())
    J_V = _nonneg(quadV - 2.0 * linV + S_V, (S_V, quadV, 2 * linV), "J_V")
    out = dict(J=combine_J(J_M, J_V, S_M, S_V, lam, absolute), J_M=J_M, J_V=J_V,
               relF_M=J_M / S_M, relF_V=J_V / S_V,
               x_norm_mean=float(Bd.norm(dim=1).mean().item()),
               y_norm_mean=float(Yd.norm(dim=1).mean().item()))
    return (out, K1, h) if want_kernels else out


def mb_J(B, Yat, Zb, Yb, mom, L, lam, absolute):
    """DIFFERENTIABLE minibatch estimate of J (Gram exact, linear terms minibatch)."""
    S_M, S_V = mom["S_M"], mom["S_V"]
    DsL = ipow(dots(B, B, exact=False), L)
    PL = ipow(dots(B, Zb, exact=False), L)
    rawM = (DsL * DsL).sum() - 2.0 * (PL * PL).mean(dim=1).sum()
    h = (PL @ Yb) / float(Zb.shape[0])
    rawV = torch.einsum("ac,ab,bc->", Yat, DsL, Yat) - 2.0 * (Yat * h).sum()
    if absolute:
        return lam * (rawM + S_M) + (1.0 - lam) * (rawV + S_V)
    return lam * (rawM / S_M + 1.0) + (1.0 - lam) * (rawV / S_V + 1.0)


# ================================================================ kip kernels
def k_poly(A, B, L, exact=True):
    if exact:
        A, B = A.double(), B.double()
    return ipow(A @ B.T, L)


def k_ntk(A, B, L, exact=True):
    """1-hidden-layer FC ReLU NTK (closed-form arccos kernel); L unused (depth is
    fixed by the closed form, matching the e_43/e_44 relu net at L = 2)."""
    if exact:
        A, B = A.double(), B.double()
    na = A.norm(dim=1, keepdim=True).clamp_min(1e-30)
    nb = B.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C_ = A @ B.T
    cos = (C_ / (na * nb.T)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    th = torch.arccos(cos)
    k1 = (na * nb.T) * (torch.sin(th) + (math.pi - th) * cos) / (2.0 * math.pi)
    return k1 + C_ * ((math.pi - th) / (2.0 * math.pi))


KFUNS = {"poly": k_poly, "ntk": k_ntk}


def krr_reg(Kss, kip_reg):
    m = Kss.shape[0]
    return float(kip_reg) * Kss.diagonal().sum() / float(m)


def exact_krr_train(Xs, Ys, T, kern, L, kip_reg, chunk=None):
    """fp64 KRR predictor scored on the TASK train rows (no test data at selection
    time).  -> dict(train_mse, train_acc, x_norm_mean, y_norm_mean)."""
    kf = KFUNS[kern]
    Xd, Yd = Xs.detach().double(), Ys.detach().double()
    m = Xd.shape[0]
    Kss = kf(Xd, Xd, L, exact=True)
    reg = krr_reg(Kss, kip_reg)
    eye = torch.eye(m, dtype=torch.float64, device=Xd.device)
    alpha = ls_solve(Kss + reg * eye, Yd)
    ch = int(chunk or max(256, 3.0e7 // max(m, 1)))
    Z, Y, yl, pi = T["Z64"], T["Y64"], T["y_local"], T["pi"]
    se = torch.zeros((), dtype=torch.float64, device=Xd.device)
    corr = torch.zeros((), dtype=torch.float64, device=Xd.device)
    n = Z.shape[0]
    for i in range(0, n, ch):
        f = kf(Z[i:i + ch], Xd, L, exact=True) @ alpha
        se += ((f - Y[i:i + ch]) ** 2).sum()
        corr += ((f + pi).argmax(dim=1) == yl[i:i + ch]).sum()
    return dict(train_mse=float(se.item()) / n, train_acc=float(corr.item()) / n,
                x_norm_mean=float(Xd.norm(dim=1).mean().item()),
                y_norm_mean=float(Yd.norm(dim=1).mean().item()))


def label_solve(Xs, T, kern, L, kip_reg, chunk=None):
    """Closed-form Label Solve: y* = C w, M1 w = M2 (see e_43 kip.py docstring)."""
    kf = KFUNS[kern]
    Xd = Xs.detach().double()
    m, N = Xd.shape[0], T["n"]
    Kss = kf(Xd, Xd, L, exact=True)
    reg = krr_reg(Kss, kip_reg)
    eye = torch.eye(m, dtype=torch.float64, device=Xd.device)
    ch = int(chunk or max(256, 3.0e7 // max(m, 1)))
    M1 = torch.zeros(m, m, dtype=torch.float64, device=Xd.device)
    M2 = torch.zeros(m, T["Y64"].shape[1], dtype=torch.float64, device=Xd.device)
    for i in range(0, N, ch):
        Ksn = kf(Xd, T["Z64"][i:i + ch], L, exact=True)
        M1 += Ksn @ Ksn.T
        M2 += Ksn @ T["Y64"][i:i + ch]
        del Ksn
    w = ls_solve(M1, M2)
    return (Kss + reg * eye) @ w


# ================================================================ output packing
def harden_labels(Y_nat, T):
    """Learned (soft, centred, local-index) labels -> hard GLOBAL class labels via
    the Bayes rule argmax(y + pi).  v1: the CE pipeline needs int labels; the soft
    vectors ride along in the synthetic payload for a future MSE mode."""
    y_loc = (Y_nat.double() + T["pi"]).argmax(dim=1)
    return T["classes"][y_loc].cpu().numpy()


def pack_synthetic(B_zeta_nat, Y_nat, T):
    """Natural-scale zeta atoms + centred labels -> (X_z float32 [m,d], y_hard
    int64 [m], y_soft float32 [m,C_task]).  X converted back to z units so the
    SyntheticDataset rows look exactly like transformed real images."""
    X = (B_zeta_nat.double() * T["sqd"]).float().cpu().numpy()
    y_hard = harden_labels(Y_nat, T)
    y_soft = Y_nat.detach().float().cpu().numpy()
    return X, y_hard, y_soft
