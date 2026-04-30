"""Recipe-MLP base component: 4-layer MLP on the V10 recipe feature matrix
WITHOUT OTE (which makes it complementary to recipe variants in v1's pool).

Mechanism: load recipe_full_te's load_and_engineer() to get the rich FE
matrix (nums + threshold flags + LR logits + digits + num_as_cat + FREQ +
ORIG mean/std), drop OTE-key columns and string cats, standardize
remaining numeric features, train MLP per fold with 5-fold
StratifiedKFold(seed=42).

Architecture (matches path_a 2026-04-26 description):
  Input → Linear(1024) → BN → GELU → Dropout(0.15)
        → Linear(512)  → BN → GELU → Dropout(0.15)
        → Linear(256)  → BN → GELU → Dropout(0.15)
        → Linear(128)  → BN → GELU → Dropout(0.15)
        → Linear(3)
  Loss: weighted CrossEntropy (class-balanced sample weights)
  Optim: AdamW lr=1e-3, weight_decay=1e-4, cosine schedule
  25 epochs, batch=2048

Per-fold checkpointing for rehydrate resilience.

Outputs:
  scripts/artifacts/oof_recipe_mlp.npy
  scripts/artifacts/test_recipe_mlp.npy
  scripts/artifacts/recipe_mlp_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
# We need recipe_full_te.load_and_engineer to build the rich FE matrix
import os
# Force defaults — no env-var FE additions, just core V10 recipe
for k in ("EXTRA_FE", "EXTRA_OOD", "EXTRA_KNN10K", "EXTRA_OOD9",
          "DROP_DETERMINISTIC", "DROP_SCORES", "ANCHOR_WEIGHT_ALPHA",
          "TTA_BOUNDARY", "THREE_WAY_OTE", "NN_DIST_PATH",
          "CLEANLAB_TREATMENT", "DAE_EMBED_PATH", "SMOKE",
          "EXTRA_W8", "EXTRA_INSTAB"):
    os.environ.pop(k, None)

from recipe_full_te import load_and_engineer  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
DEVICE = "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] MLP: {m}", flush=True)


class RecipeMLP(nn.Module):
    def __init__(self, in_dim, hidden=(1024, 512, 256, 128), n_class=3, p=0.15):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_class))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_fold(X_tr, y_tr, X_va, X_te, in_dim, sample_w, n_epochs=25, batch=2048):
    """Train MLP for one fold, return val + test predictions (softmax probs)."""
    model = RecipeMLP(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss(reduction="none")

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    sw_t  = torch.from_numpy(sample_w).float()
    X_va_t = torch.from_numpy(X_va).float()
    X_te_t = torch.from_numpy(X_te).float()
    n = len(X_tr_t)

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(X_tr_t[idx])
            loss = (crit(logits, y_tr_t[idx]) * sw_t[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 4 or ep == n_epochs - 1:
            log(f"    epoch {ep+1}/{n_epochs} lr={sched.get_last_lr()[0]:.2e} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        # Batch eval to avoid OOM on big arrays
        def predict(X):
            out = []
            for i in range(0, len(X), 8192):
                chunk = X[i:i + 8192].to(DEVICE)
                out.append(model(chunk).softmax(1).cpu().numpy())
            return np.vstack(out)
        va_p = predict(X_va_t)
        te_p = predict(X_te_t)
    return va_p, te_p


def main():
    t0 = time.time()
    log("loading + engineering V10 recipe FE matrix")
    train, test, info, _ = load_and_engineer()
    y = train[TARGET].to_numpy().astype(np.int64)
    log(f"  train {train.shape}  test {test.shape}")

    # Identify numeric columns (drop string cats + target)
    nums = info.get("nums", [])
    cats = info.get("cats", [])
    digits = info.get("digits", [])
    num_as_cat = info.get("num_as_cat", [])
    tres = info.get("tres", [])
    logits_cols = info.get("logits", [])
    freq = info.get("freq", [])
    orig_stats = info.get("orig_stats", [])
    extra = info.get("extra_domain", []) + info.get("extra_decimal", []) + info.get("extra_w8", [])
    gby = info.get("gby_cols", [])

    # Build numeric feature list (skip cats which are still strings, and skip target)
    feat_cols = (nums + digits + num_as_cat + tres + logits_cols
                 + freq + orig_stats + extra + gby)
    feat_cols = [c for c in feat_cols if c in train.columns and c != TARGET]
    log(f"  numeric features: {len(feat_cols)} (skipping {len(cats)} string cats, OTE cols not generated)")

    X_train = train[feat_cols].astype(np.float32).to_numpy()
    X_test = test[feat_cols].astype(np.float32).to_numpy()
    in_dim = X_train.shape[1]
    log(f"  feature matrix: train {X_train.shape}  test {X_test.shape}")

    # Replace NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Class-balanced sample weights
    n_class = 3
    counts = np.bincount(y, minlength=n_class)
    cw = len(y) / (n_class * counts)
    sample_w = cw[y].astype(np.float32)
    log(f"  class counts: {counts}, weights: {cw}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_folds = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        ckpt_oof = ART / f"_recipe_mlp_fold{fold}_oof.npy"
        ckpt_te  = ART / f"_recipe_mlp_fold{fold}_test.npy"
        ckpt_meta = ART / f"_recipe_mlp_fold{fold}_meta.json"
        if ckpt_oof.exists() and ckpt_te.exists() and ckpt_meta.exists():
            log(f"  fold {fold+1}/{N_FOLDS} resuming from checkpoint")
            vp = np.load(ckpt_oof); tp = np.load(ckpt_te)
            mi = json.loads(ckpt_meta.read_text())
            oof[va_idx] = vp.astype(np.float32)
            test_folds.append(tp)
            log(f"    val_argmax={mi['argmax_bal']:.5f} (cached)")
            continue
        t1 = time.time()
        log(f"  fold {fold+1}/{N_FOLDS} training MLP")
        # Standardize per-fold (fit on train, apply to val + test)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train[tr_idx]).astype(np.float32)
        X_va_s = scaler.transform(X_train[va_idx]).astype(np.float32)
        X_te_s = scaler.transform(X_test).astype(np.float32)
        # Replace any inf/nan introduced by scaling (zero-variance cols)
        X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_va_s = np.nan_to_num(X_va_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_s = np.nan_to_num(X_te_s, nan=0.0, posinf=0.0, neginf=0.0)

        sw_tr = sample_w[tr_idx]
        vp, tp = train_fold(X_tr_s, y[tr_idx], X_va_s, X_te_s, in_dim, sw_tr)
        oof[va_idx] = vp.astype(np.float32)
        test_folds.append(tp)

        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        np.save(ckpt_oof, vp.astype(np.float32))
        np.save(ckpt_te, tp.astype(np.float32))
        ckpt_meta.write_text(json.dumps({"argmax_bal": float(argmax_bal)}))
        log(f"    fold {fold+1} val_argmax={argmax_bal:.5f} wall={time.time()-t1:.1f}s [ckpt]")

    test_pred = np.mean(test_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_recipe_mlp.npy", oof)
    np.save(ART / "test_recipe_mlp.npy", test_pred)

    oof_argmax = balanced_accuracy_score(y, oof.argmax(1))
    log(f"\n=== recipe-MLP standalone ===")
    log(f"  argmax bal_acc = {oof_argmax:.5f}")

    # Tuned log-bias (recipe bias)
    BIAS = np.array([1.4324, 1.4689, 3.4008])
    pred_tuned = (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1)
    tuned_bal = balanced_accuracy_score(y, pred_tuned)
    log(f"  @recipe-bias = {tuned_bal:.5f}")

    out = dict(
        feat_cols=feat_cols,
        n_features=in_dim,
        oof_argmax=float(oof_argmax),
        oof_tuned_recipe_bias=float(tuned_bal),
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / "recipe_mlp_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {json_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
