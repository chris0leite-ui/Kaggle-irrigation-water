"""KAN model + training loop.

efficient_kan.KAN takes a list of layer widths [in, h1, h2, ..., out]
and parameterises each edge as a B-spline of order=spline_order with
grid_size knots over grid_range. Forward is standard PyTorch.

Training: AdamW + cosine LR schedule + class-balanced sample weights
(per-batch reweight, since CE doesn't directly take per-sample weights
when using CrossEntropyLoss(reduction='none') manually).
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KAN


def make_class_weights(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    w_per_class = inv * (n_classes / inv.sum())
    return w_per_class.astype(np.float32)


def _build_kan(in_dim: int, hidden: list[int], out_dim: int,
               grid_size: int, spline_order: int,
               grid_range: tuple[float, float], dropout: float):
    layers = [in_dim] + list(hidden) + [out_dim]
    return KAN(
        layers,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=list(grid_range),
    )


def _forward_with_dropout(model, x, dropout: float, training: bool):
    if dropout > 0 and training:
        # KAN doesn't include dropout; insert it externally.
        # Simple input-feature dropout applied once before the network.
        mask = (torch.rand_like(x) > dropout).float() / (1.0 - dropout)
        x = x * mask
    return model(x)


def fit_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_va: np.ndarray, X_te: np.ndarray,
                 *, hidden: list[int], grid_size: int, spline_order: int,
                 grid_range: tuple[float, float], dropout: float,
                 n_epochs: int, batch_size: int, lr: float,
                 weight_decay: float, label_smoothing: float = 0.0,
                 num_classes: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = X_tr.shape[1]
    print(f"      build KAN [{in_dim}, "
          f"{', '.join(str(h) for h in hidden)}, {num_classes}] "
          f"device={device}", flush=True)
    model = _build_kan(in_dim, hidden, num_classes, grid_size,
                       spline_order, grid_range, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params={n_params:,}", flush=True)

    cw = torch.tensor(make_class_weights(y_tr, num_classes),
                      dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    n_train = len(X_tr)
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * n_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total_steps, 1))
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_va_t = torch.from_numpy(X_va).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)

    rng = np.random.default_rng(42)
    for ep in range(n_epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx].to(device, non_blocking=True)
            yb = y_tr_t[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = _forward_with_dropout(model, xb, dropout, True)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item()) * len(idx)
        ep_loss /= max(n_train, 1)
        print(f"      epoch {ep + 1}/{n_epochs} loss={ep_loss:.5f} "
              f"lr={sched.get_last_lr()[0]:.2e} wall={time.time()-t0:.1f}s",
              flush=True)

    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, len(X_va_t), 8192):
            chunks.append(F.softmax(model(X_va_t[i:i + 8192]),
                                    dim=-1).cpu().numpy())
        p_va = np.concatenate(chunks, axis=0)
        chunks = []
        for i in range(0, len(X_te_t), 8192):
            chunks.append(F.softmax(model(X_te_t[i:i + 8192]),
                                    dim=-1).cpu().numpy())
        p_te = np.concatenate(chunks, axis=0)
    return p_va.astype(np.float32), p_te.astype(np.float32)
