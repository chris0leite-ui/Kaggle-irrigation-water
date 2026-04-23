"""Training helpers: fold loop, log-bias tuning, cosine schedule.

Kept as small pure-ish functions so the orchestrator can call them
from both the Optuna objective and the final refit loop.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1.0 + math.cos(math.pi * prog))


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    """3-param coord-ascent log-bias; wide High grid."""
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 5.0, 76)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(
                    y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def predict_probs(model, x_num, x_dig, x_cat, device, batch: int = 8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch):
            xn = x_num[i:i+batch].to(device, non_blocking=True)
            xd = x_dig[i:i+batch].to(device, non_blocking=True) if x_dig is not None else None
            xc = x_cat[i:i+batch].to(device, non_blocking=True) if x_cat is not None else None
            out.append(model(xn, xd, xc).cpu())
    logits = torch.cat(out, 0).numpy()
    return torch.softmax(torch.from_numpy(logits), dim=1).numpy()


def train_one_fold(model, opt, loaders, tensors_val, y_val, log_prior,
                   epochs: int, device, grad_clip: float = 1.0,
                   warmup_frac: float = 0.1, base_lr: float = 3e-4,
                   log_fn=print, log_prefix: str = ""):
    """Train for `epochs`; return best-val probs + best-val bal_acc.

    loaders: dict with 'train' -> DataLoader yielding (x_num, x_dig, x_cat, y).
    tensors_val: tuple (x_num, x_dig, x_cat) CPU tensors.
    """
    total_steps = epochs * len(loaders["train"])
    warmup_steps = int(warmup_frac * total_steps)
    step = 0
    best_bal, best_probs = -1.0, None
    for ep in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for batch in loaders["train"]:
            xn, xd, xc, yb = batch
            lr = cosine_lr(step, total_steps, warmup_steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            xn = xn.to(device, non_blocking=True)
            xd = xd.to(device, non_blocking=True) if xd is not None else None
            xc = xc.to(device, non_blocking=True) if xc is not None else None
            yb = yb.to(device, non_blocking=True)
            logits = model(xn, xd, xc) + log_prior.unsqueeze(0)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item() * yb.size(0)
            n_seen += yb.size(0)
            step += 1
        val_probs = predict_probs(model, *tensors_val, device)
        bal = balanced_accuracy_score(y_val, val_probs.argmax(axis=1))
        log_fn(f"{log_prefix}ep {ep+1:2d}/{epochs}  loss {running/n_seen:.4f}  "
               f"val bal {bal:.5f}  lr {lr:.2e}")
        if bal > best_bal:
            best_bal = bal
            best_probs = val_probs
    return best_probs, best_bal


def make_loader(x_num, x_dig, x_cat, y, batch: int, shuffle: bool):
    tensors = [torch.from_numpy(x_num).float()]
    if x_dig is not None:
        tensors.append(torch.from_numpy(x_dig).long())
    else:
        tensors.append(torch.zeros(len(x_num), 0, dtype=torch.long))
    if x_cat is not None:
        tensors.append(torch.from_numpy(x_cat).long())
    else:
        tensors.append(torch.zeros(len(x_num), 0, dtype=torch.long))
    tensors.append(torch.from_numpy(y).long())
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=shuffle)
