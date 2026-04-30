"""ExcelFormer model + training loop via pytorch_frame.

ExcelFormer (Liang et al. 2024): semi-permutation-invariant attention
designed to beat GBDT on tabular. Two main attention components:
  - DIAM (Distance-aware Inter-feature Attention Mechanism)
  - AIUM (Anchor-Inter-feature User-aware Mechanism)
Plus 'mixup' augmentation in feature/hidden space for regularization.

Untested on this comp. Mathematically distinct from all 16+ prior NN
nulls (MLP / FT-T / TabPFN / DAE / RealMLP / Trompt / Mambular / KAN /
TabM / TabNet — TabNet just nulled at 17th).

Key params (paper defaults for tabular classification):
  in_channels = 32 (per-feature embedding dim)
  num_layers = 5
  num_heads = 32
  diam/aium/residual_dropout = 0.2
  mixup = 'hidden'  (their key contribution to regularization)
  beta = 0.5  (mixup mixing ratio)

ONLY accepts numerical features; cats are pre-factorized in features.py.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_frame import stype
from torch_frame.data import DataLoader, Dataset
from torch_frame.nn.models import ExcelFormer


def make_dataset(df, nums, target=None):
    col_to_stype = {c: stype.numerical for c in nums}
    if target is not None:
        col_to_stype[target] = stype.categorical
    ds = Dataset(df=df, col_to_stype=col_to_stype, target_col=target)
    ds.materialize()
    return ds


def build_model(train_ds, num_classes: int, num_cols: int, device,
                in_channels: int = 32,
                num_layers: int = 5,
                num_heads: int = 32,
                diam_dropout: float = 0.2,
                aium_dropout: float = 0.2,
                residual_dropout: float = 0.2,
                mixup_mode: str = "hidden",
                beta: float = 0.5) -> ExcelFormer:
    model = ExcelFormer(
        in_channels=in_channels,
        out_channels=num_classes,
        num_cols=num_cols,
        num_layers=num_layers,
        num_heads=num_heads,
        col_stats=train_ds.col_stats,
        col_names_dict=train_ds.tensor_frame.col_names_dict,
        diam_dropout=diam_dropout,
        aium_dropout=aium_dropout,
        residual_dropout=residual_dropout,
        mixup=mixup_mode,
        beta=beta,
    ).to(device)
    return model


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for tf in loader:
        tf = tf.to(device)
        optimizer.zero_grad()
        # ExcelFormer with mixup returns (out, mixed_y) tuple
        y = tf.y.long()
        if hasattr(model, "mixup") and model.mixup is not None:
            out, y_mixed = model(tf, mixup_encoded=True)
            # When mixup active, y is replaced by soft mixture
            loss = F.cross_entropy(out, y_mixed)
        else:
            out = model(tf)
            loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader, device) -> torch.Tensor:
    model.eval()
    probs = []
    for tf in loader:
        tf = tf.to(device)
        # No mixup at inference
        out = model(tf)
        probs.append(F.softmax(out, dim=1).cpu())
    return torch.cat(probs, dim=0)


def fit_one_fold(train_ds, val_ds, test_ds, device,
                 n_epochs: int, batch_size: int = 512,
                 lr: float = 3e-4, weight_decay: float = 1e-5,
                 num_classes: int = 3,
                 in_channels: int = 32,
                 num_layers: int = 5,
                 num_heads: int = 32,
                 diam_dropout: float = 0.2,
                 aium_dropout: float = 0.2,
                 residual_dropout: float = 0.2,
                 mixup_mode: str = "hidden",
                 beta: float = 0.5):
    num_cols = len(train_ds.tensor_frame.col_names_dict[stype.numerical])
    model = build_model(train_ds, num_classes, num_cols, device,
                        in_channels, num_layers, num_heads,
                        diam_dropout, aium_dropout, residual_dropout,
                        mixup_mode, beta)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=n_epochs)
    tl = DataLoader(train_ds.tensor_frame, batch_size=batch_size,
                    shuffle=True)
    vl = DataLoader(val_ds.tensor_frame, batch_size=batch_size)
    el = DataLoader(test_ds.tensor_frame, batch_size=batch_size)
    for ep in range(n_epochs):
        loss = train_epoch(model, tl, opt, device)
        sched.step()
        if ep % 5 == 0 or ep == n_epochs - 1:
            print(f"      epoch {ep+1}/{n_epochs} train_loss={loss:.4f}",
                  flush=True)
    p_val = predict_proba(model, vl, device).numpy()
    p_test = predict_proba(model, el, device).numpy()
    return p_val, p_test
