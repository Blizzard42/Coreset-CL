"""Flat MLP backbones ported from e_43 (PrecisionGD) coreset.py/kip.py `Net`.

The e_43 model is  f(x) = W_L diag(x) ... W_2 diag(x) W_1 x  (zmlp: degree-L
homogeneous polynomial; relu: same shapes with relu gates).  Here the readout W_L
is the harness's incremental fc head, so the backbone is the chain up to the last
hidden activation:

    zmlp  h_1 = x W_1^T,  h_{l+1} = (x * h_l) W_{l+1}^T,  features = x * h_{L-1}
    relu  features = relu(h_{L-1})

e_43 feeds zeta = z/sqrt(d) with in_gain sqrt(d); feeding the standardized z
directly with in_gain 1 is the identical arithmetic, and z is exactly what the
harness's Normalize(ToTensor(img)) pipeline produces (per-channel rather than
e_43's per-feature standardization -- a documented deviation).  Init 1/sqrt(fan_in)
keeps hidden rms ~ 1 under |z| ~ sqrt(d).

Dropout (optional, config "backbone_dropout") is applied on every hidden gate
activation in train mode -- the e_43 regularized-coreset knob that fixed the zMLP
accuracy-decay; eval-mode nets are deterministic.

IMPORTANT for CL runs: pair these with "no_augment": true.  The distillation
selection methods fit atoms in exactly this flattened-normalized representation,
and augmentation would break representation equality between selection and
training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBackbone(nn.Module):
    def __init__(self, arch="zmlp", d_in=3072, depth=2, dropout=0.0):
        super().__init__()
        assert arch in ("zmlp", "relu")
        assert depth >= 2, "depth-1 backbone would have no hidden layer"
        self.arch, self.d_in, self.depth = arch, int(d_in), int(depth)
        self.p_drop = float(dropout)
        self.out_dim = self.d_in
        # W_1 .. W_{L-1}: all d_in x d_in (e_43 uses hidden width k = d)
        self.Ws = nn.ParameterList(
            nn.Parameter(torch.randn(self.d_in, self.d_in) / np.sqrt(self.d_in))
            for _ in range(self.depth - 1))

    def forward(self, x):
        z = x.reshape(x.shape[0], -1)
        h = z @ self.Ws[0].T
        for W in list(self.Ws)[1:]:
            h = (z * h) if self.arch == "zmlp" else torch.relu(h)
            if self.p_drop > 0.0:
                h = F.dropout(h, self.p_drop, self.training)
            h = h @ W.T
        feats = (z * h) if self.arch == "zmlp" else torch.relu(h)
        if self.p_drop > 0.0:
            feats = F.dropout(feats, self.p_drop, self.training)
        return {"features": feats, "fmaps": []}


def zmlp(args):
    return MLPBackbone("zmlp", d_in=args.get("mlp_d_in", 3072),
                       depth=args.get("mlp_depth", 2),
                       dropout=args.get("backbone_dropout", 0.0))


def relu_mlp(args):
    return MLPBackbone("relu", d_in=args.get("mlp_d_in", 3072),
                       depth=args.get("mlp_depth", 2),
                       dropout=args.get("backbone_dropout", 0.0))
