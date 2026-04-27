"""Mech D: per-row attention over the LB-best component bank.

Architecture:
  context (14d) → MLP(64) → softmax(N_components) = per-row attention weights
  blend_log_probs(i, k) = sum_c weights[i, c] * log(component_probs[c, i, k])
  output = softmax(blend_log_probs)
  loss   = cross-entropy(output, y)

The constraint that the output is a CONVEX combination of per-component
log-probs is a strong inductive bias — the row-conditional analog of
the global CMA-ES blend (which had in-sample upper bound 0.98091, just
+0.00007 above LB-best 4-stack OOF). Different from prior meta-stackers
(LR/XGB) which output 3-class probs DIRECTLY rather than mixing weights.

Pool: 62 LB-best v1 components (from tier1b_xgb_metastack_results.json).
Same 5-fold StratifiedKFold(seed=42) split as every other OOF on disk.

Outputs:
  scripts/artifacts/oof_mech_d.npy
  scripts/artifacts/test_mech_d.npy
  scripts/artifacts/mech_d_results.json
"""
from __future__ import annotations

import json
import os
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, DATA, BIAS, build_lbbest_stack, iso_cal, log, bal_at_bias,
)

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
SEED = 42
N_FOLDS = 1 if SMOKE else 5
N_EPOCHS = 5 if SMOKE else 30
BATCH_SIZE = 4096
LR = 1e-3
HIDDEN = 64
DROPOUT = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)


class AttentionGate(nn.Module):
    def __init__(self, n_context: int, n_components: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_context, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, n_components),
        )

    def forward(self, ctx, log_comps):
        """ctx: (B, n_context) — row context features
        log_comps: (B, n_components, 3) — per-component log-probs
        Returns: (B, 3) — per-row blended log-probs (NOT yet softmaxed)
        """
        weights = F.softmax(self.net(ctx), dim=1)  # (B, n_components)
        # blend = sum_c weights[i,c] * log_comps[i,c,:]
        blend = torch.einsum("bc,bck->bk", weights, log_comps)
        return blend


def load_lb_pool():
    """Load the 62 components used by LB-best meta-stacker v1."""
    d = json.loads((ART / "tier1b_xgb_metastack_results.json").read_text())
    names = d["components"]
    pool = {}
    for name in names:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not (oof_p.exists() and test_p.exists()):
            log(f"WARN: missing {name}")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != 630_000:
            continue
        pool[name] = (o, t)
    return pool


def build_context(df, lb_oof, lb_test, is_train: bool):
    d = add_distance_features(df)
    ctx_cols = ["dgp_score", "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                "sm_abs", "rf_abs", "tc_abs", "ws_abs", "min_axis_abs"]
    base = d[ctx_cols].to_numpy(dtype=np.float32)
    # Standardize each col (using train stats — caller pre-normalizes)
    if is_train:
        mean = base.mean(0); std = base.std(0) + 1e-6
        base = (base - mean) / std
    # Add anchor probs + max-prob + entropy
    p = lb_oof if is_train else lb_test
    max_p = p.max(1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1, keepdims=True)
    return np.concatenate([base, p, max_p, ent], axis=1).astype(np.float32)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_full = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int64)

    log("loading anchors")
    lb3_oof, lb3_test = build_lbbest_stack(y_full)
    mv1 = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    mv1_te = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    mv1_iso, mv1_iso_te = iso_cal(mv1, mv1_te, y_full)
    lb4_oof = log_blend([lb3_oof, mv1_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, mv1_iso_te], np.array([0.7, 0.3]))
    log(f"  LB-best 4-stack OOF = {bal_at_bias(lb4_oof, y_full):.5f}")

    log("loading 62-component LB-best v1 pool")
    pool = load_lb_pool()
    names = sorted(pool.keys())
    log(f"  loaded {len(names)} components")
    n_comp = len(names)

    # Stack components: (N_train, n_comp, 3) and (N_test, n_comp, 3)
    log_oof = np.stack([np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in names], axis=1)
    log_test = np.stack([np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in names], axis=1)

    # Build context features
    log("building context features")
    # Use full-train stats for scaling (no fold-leak: same scaler for tr+va; lb_oof
    # is per-row OOF, so it's fold-honest)
    ctx_train = build_context(train, lb4_oof, lb4_test, is_train=True)
    ctx_test = build_context(test, lb4_oof, lb4_test, is_train=False)
    # Apply train-derived stats to test
    # (build_context above standardizes train in-place; apply to test now)
    train_means = ctx_train.mean(0); train_stds = ctx_train.std(0) + 1e-6
    # We already standardized ctx_train; rebuild test with same means/stds
    ctx_test = (ctx_test - train_means) / train_stds
    n_ctx = ctx_train.shape[1]
    log(f"  context dim: {n_ctx}, components: {n_comp}")

    # 5-fold leak-safe training
    skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
    oof_blend = np.zeros((len(y_full), 3), dtype=np.float32)
    test_blend_folds = []
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(ctx_train, y_full)):
        if fold >= N_FOLDS:
            break
        t1 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AttentionGate(n_ctx, n_comp).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        X_tr = torch.from_numpy(ctx_train[tr_idx]).to(device)
        L_tr = torch.from_numpy(log_oof[tr_idx]).to(device)
        Y_tr = torch.from_numpy(y_full[tr_idx]).to(device)
        X_va = torch.from_numpy(ctx_train[va_idx]).to(device)
        L_va = torch.from_numpy(log_oof[va_idx]).to(device)
        Y_va = torch.from_numpy(y_full[va_idx]).to(device)
        X_te = torch.from_numpy(ctx_test).to(device)
        L_te = torch.from_numpy(log_test).to(device)

        best_va_bal = 0.0
        best_state = None
        for ep in range(N_EPOCHS):
            model.train()
            perm = torch.randperm(len(X_tr), device=device)
            losses = []
            for i in range(0, len(perm), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                logits = model(X_tr[idx], L_tr[idx])
                loss = F.cross_entropy(logits, Y_tr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                vp = []
                for i in range(0, len(X_va), BATCH_SIZE):
                    vp.append(F.softmax(model(X_va[i:i+BATCH_SIZE], L_va[i:i+BATCH_SIZE]), dim=1))
                vp = torch.cat(vp).cpu().numpy()
            va_bal = balanced_accuracy_score(y_full[va_idx], vp.argmax(1))
            if va_bal > best_va_bal:
                best_va_bal = va_bal
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            log(f"  fold {fold+1}/{N_FOLDS} ep {ep+1}/{N_EPOCHS}  "
                f"loss={np.mean(losses):.4f}  va_bal={va_bal:.5f}  best={best_va_bal:.5f}")

        # Restore best, predict val + test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(X_va), BATCH_SIZE):
                vp.append(F.softmax(model(X_va[i:i+BATCH_SIZE], L_va[i:i+BATCH_SIZE]), dim=1))
            vp = torch.cat(vp).cpu().numpy()
            tp = []
            for i in range(0, len(X_te), BATCH_SIZE):
                tp.append(F.softmax(model(X_te[i:i+BATCH_SIZE], L_te[i:i+BATCH_SIZE]), dim=1))
            tp = torch.cat(tp).cpu().numpy()
        oof_blend[va_idx] = vp.astype(np.float32)
        test_blend_folds.append(tp.astype(np.float32))
        fold_results.append(dict(fold=fold + 1, best_va_bal=float(best_va_bal),
                                 wall=float(time.time() - t1)))
        log(f"  fold {fold+1}/{N_FOLDS} DONE  va_bal={best_va_bal:.5f}  "
            f"wall={time.time()-t1:.1f}s")

    test_blend = np.mean(test_blend_folds, axis=0).astype(np.float32)
    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_mech_d{suffix}.npy", oof_blend)
    np.save(ART / f"test_mech_d{suffix}.npy", test_blend)
    log(f"saved oof_mech_d{suffix}.npy + test_mech_d{suffix}.npy")

    if not SMOKE:
        argmax_bal = balanced_accuracy_score(y_full, oof_blend.argmax(1))
        tuned_bal = bal_at_bias(oof_blend, y_full)
        log(f"\n=== Mech D standalone ===")
        log(f"  argmax OOF      = {argmax_bal:.5f}")
        log(f"  @recipe-bias    = {tuned_bal:.5f}")
        log(f"  LB-best 4-stack = {bal_at_bias(lb4_oof, y_full):.5f}  "
            f"Δ = {tuned_bal - bal_at_bias(lb4_oof, y_full):+.5f}")
        out = dict(
            n_components=n_comp, components=names, n_context=n_ctx,
            n_epochs=N_EPOCHS, hidden=HIDDEN, lr=LR, dropout=DROPOUT,
            fold_results=fold_results,
            argmax_oof=float(argmax_bal),
            tuned_oof=float(tuned_bal),
            lb4_oof=float(bal_at_bias(lb4_oof, y_full)),
            elapsed_sec=float(time.time() - t0),
        )
        (ART / "mech_d_results.json").write_text(json.dumps(out, indent=2))
        log(f"wrote mech_d_results.json")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
