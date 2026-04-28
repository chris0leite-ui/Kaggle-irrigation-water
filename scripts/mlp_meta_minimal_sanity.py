"""Minimal-input meta sanity check (CLAUDE.md 2026-04-28 mandatory).

Tests whether MLP-meta's apparent OOF lift is from orthogonal signal in the
62-component bank, OR from cross-component pattern memorization (the trap
that closed the macrorec family minimal test at OOF 0.98051 < LB-best 4-stack).

The 2-component minimal test:
  Train an MLP with the SAME architecture on ONLY:
    - LB-best 3-stack log-probs (3 dims)
    - Single component's log-probs (3 dims)
  = 6-dim input, no dist features, no other components.

If this 2-comp MLP lands BELOW LB-best 3-stack OOF (~0.98061), the
candidate component contributes NOTHING orthogonal to LB-3-stack — the
N-component meta's lift was cross-component memorization.

Usage:
  python scripts/mlp_meta_minimal_sanity.py --component recipe_full_te
  python scripts/mlp_meta_minimal_sanity.py --component xgb_metastack
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from mlp_meta_macrorec import MLPMeta, macrorec_loss  # noqa: E402
from tier1b_xgb_metastack import build_lbbest_stack, _normed  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
EPS = 1e-12
BIAS = np.array([1.4324, 1.4689, 3.4008])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + BIAS).argmax(1))


def train_minimal(X_tr, y_tr, X_va, n_epochs=30, lr=1e-3, wd=1e-5,
                  lam_ce=0.3, seed=42):
    torch.manual_seed(seed)
    mu = X_tr.mean(0, keepdims=True)
    sd = X_tr.std(0, keepdims=True) + 1e-6
    Xt = ((X_tr - mu) / sd).astype(np.float32)
    Xv = ((X_va - mu) / sd).astype(np.float32)
    n_class = 3
    model = MLPMeta(Xt.shape[1], hidden=(128, 64, 32), n_class=n_class).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    cw = np.bincount(y_tr, minlength=n_class).astype(np.float32)
    cw = cw.sum() / np.maximum(cw, 1) / n_class
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    Xt_t = torch.tensor(Xt, dtype=torch.float32, device=DEVICE)
    yt_t = torch.tensor(y_tr.astype(np.int64), dtype=torch.long, device=DEVICE)
    n = len(Xt_t)
    batch = 4096
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            logits = model(Xt_t[b])
            loss = macrorec_loss(logits, yt_t[b], n_class, lam_ce, cw_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        Xv_t = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
        return torch.softmax(model(Xv_t), dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", type=str, required=True,
                    help="Component name (oof_<name>.npy in artifacts)")
    ap.add_argument("--lam_ce", type=float, default=0.3)
    ap.add_argument("--n_epochs", type=int, default=30)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log(f"loading anchor and component {args.component!r}")
    lb_oof, _ = build_lbbest_stack(y)
    comp_oof = _normed(np.load(ART / f"oof_{args.component}.npy"))

    lb_log = np.log(np.clip(lb_oof, 1e-9, 1.0))
    comp_log = np.log(np.clip(comp_oof, 1e-9, 1.0))
    X = np.concatenate([lb_log, comp_log], axis=1).astype(np.float32)
    log(f"  feature shape: {X.shape}  (LB-3-stack 3 + {args.component} 3)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t1 = time.time()
        vp = train_minimal(X[tr_idx], y[tr_idx], X[va_idx],
                           n_epochs=args.n_epochs, lam_ce=args.lam_ce,
                           seed=SEED + fold)
        oof[va_idx] = vp
        b = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        log(f"  fold {fold+1} val_argmax = {b:.5f}  wall = {time.time()-t1:.1f}s")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
    tuned_bal = bal(oof, y)
    lb_bal = bal(lb_oof, y)
    log(f"\n=== MINIMAL-INPUT META (LB-3-stack + {args.component}) ===")
    log(f"  argmax bal_acc       = {argmax_bal:.5f}")
    log(f"  @recipe-bias bal_acc = {tuned_bal:.5f}")
    log(f"  LB-3-stack OOF bal   = {lb_bal:.5f}")
    log(f"  Δ vs LB-3-stack      = {tuned_bal - lb_bal:+.5f}")
    if tuned_bal < lb_bal:
        log("  VERDICT: BELOW LB-3-stack → component adds NO orthogonal signal")
    elif tuned_bal - lb_bal < 0.0001:
        log("  VERDICT: marginal — component contribution within fold noise")
    else:
        log("  VERDICT: lifts vs LB-3-stack → component HAS orthogonal signal")

    out = dict(component=args.component, n_epochs=args.n_epochs,
               lam_ce=args.lam_ce,
               argmax_bal=float(argmax_bal),
               tuned_bal=float(tuned_bal),
               lb_3stack_bal=float(lb_bal),
               delta_vs_lb_3stack=float(tuned_bal - lb_bal))
    out_path = ART / f"mlp_meta_minimal_{args.component}_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
