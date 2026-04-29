"""Recipe-MLP with OTE features (vs recipe_mlp without OTE).

Same 4-layer MLP architecture as recipe_mlp.py, but the input feature
matrix INCLUDES the per-fold OTE columns (~351 OTE features for 117
categorical keys × 3 classes).

Hypothesis: with OTE included, the MLP's standalone OOF should rise
above ~0.97 — past the meta-XGB's absorption threshold (recipe_mlp
without OTE was at 0.96369, below threshold). At ≥0.97 standalone +
NN inductive bias, the meta should be able to extract orthogonal
signal.

Per-fold OTE matches recipe pipeline exactly. 5-fold StratifiedKFold(seed=42).

Outputs:
  scripts/artifacts/oof_recipe_mlp_ote.npy
  scripts/artifacts/test_recipe_mlp_ote.npy
  scripts/artifacts/recipe_mlp_ote_results.json
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
from _recipe_helpers import build_fe, ART, SUB, DATA, TARGET, SEED, N_FOLDS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

DEVICE = "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] MLP-OTE: {m}", flush=True)


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
    model = RecipeMLP(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss(reduction="none")
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    sw_t = torch.from_numpy(sample_w).float()
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
            log(f"    epoch {ep+1}/{n_epochs} loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        def predict(X):
            out = []
            for i in range(0, len(X), 8192):
                chunk = X[i:i + 8192].to(DEVICE)
                out.append(model(chunk).softmax(1).cpu().numpy())
            return np.vstack(out)
        return predict(X_va_t), predict(X_te_t)


def main():
    t0 = time.time()
    log("loading + engineering V10 recipe FE")
    train, test, info, te_keys, static_cols = build_fe()
    y = train[TARGET].to_numpy().astype(np.int64)
    log(f"  train {train.shape}  test {test.shape}  te_keys {len(te_keys)}  static {len(static_cols)}")

    n_class = 3
    counts = np.bincount(y, minlength=n_class)
    cw = len(y) / (n_class * counts)
    sample_w = cw[y].astype(np.float32)
    log(f"  class counts: {counts}, weights: {cw}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_folds = []
    fold_argmaxes = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        ckpt_oof = ART / f"_recipe_mlp_ote_fold{fold}_oof.npy"
        ckpt_te  = ART / f"_recipe_mlp_ote_fold{fold}_test.npy"
        ckpt_meta = ART / f"_recipe_mlp_ote_fold{fold}_meta.json"
        if ckpt_oof.exists() and ckpt_te.exists() and ckpt_meta.exists():
            log(f"  fold {fold+1}/{N_FOLDS} resuming from checkpoint")
            vp = np.load(ckpt_oof); tp = np.load(ckpt_te)
            mi = json.loads(ckpt_meta.read_text())
            oof[va_idx] = vp.astype(np.float32)
            test_folds.append(tp)
            fold_argmaxes.append(mi["argmax_bal"])
            log(f"    val_argmax={mi['argmax_bal']:.5f} (cached)")
            continue

        t1 = time.time()
        log(f"  fold {fold+1}/{N_FOLDS} fitting OTE")
        tr_df = train.iloc[tr_idx].reset_index(drop=True).copy()
        va_df = train.iloc[va_idx].reset_index(drop=True).copy()
        tr_shuf = tr_df.sample(frac=1.0, random_state=SEED + fold).reset_index(drop=True)
        ote = OrderedTE(a=1.0)
        ote.fit(tr_shuf, te_keys, TARGET)
        tr_with_te = ote.transform(tr_df)
        va_with_te = ote.transform(va_df)
        te_with_te = ote.transform(test)

        ote_cols = ote.te_col_names()
        feat_cols = [c for c in static_cols + ote_cols if c in tr_with_te.columns]
        seen = set(); feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]

        X_tr = np.nan_to_num(tr_with_te[feat_cols].astype(np.float32).to_numpy(), nan=0., posinf=0., neginf=0.)
        X_va = np.nan_to_num(va_with_te[feat_cols].astype(np.float32).to_numpy(), nan=0., posinf=0., neginf=0.)
        X_te = np.nan_to_num(te_with_te[feat_cols].astype(np.float32).to_numpy(), nan=0., posinf=0., neginf=0.)
        log(f"    feat_dim={X_tr.shape[1]} (static {len(static_cols)} + ote {len(ote_cols)})")

        scaler = StandardScaler()
        X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr), nan=0., posinf=0., neginf=0.).astype(np.float32)
        X_va_s = np.nan_to_num(scaler.transform(X_va), nan=0., posinf=0., neginf=0.).astype(np.float32)
        X_te_s = np.nan_to_num(scaler.transform(X_te), nan=0., posinf=0., neginf=0.).astype(np.float32)

        sw_tr = sample_w[tr_idx]
        in_dim = X_tr_s.shape[1]
        log(f"    training MLP (in_dim={in_dim})")
        vp, tp = train_fold(X_tr_s, y[tr_idx], X_va_s, X_te_s, in_dim, sw_tr)
        oof[va_idx] = vp.astype(np.float32)
        test_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_argmaxes.append(argmax_bal)
        np.save(ckpt_oof, vp.astype(np.float32))
        np.save(ckpt_te, tp.astype(np.float32))
        ckpt_meta.write_text(json.dumps({"argmax_bal": float(argmax_bal)}))
        log(f"    fold {fold+1} val_argmax={argmax_bal:.5f} wall={time.time()-t1:.1f}s [ckpt]")

    test_pred = np.mean(test_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_recipe_mlp_ote.npy", oof)
    np.save(ART / "test_recipe_mlp_ote.npy", test_pred)

    oof_argmax = balanced_accuracy_score(y, oof.argmax(1))
    BIAS = np.array([1.4324, 1.4689, 3.4008])
    pred_tuned = (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1)
    tuned_bal = balanced_accuracy_score(y, pred_tuned)
    log(f"\n=== recipe-MLP-OTE standalone ===")
    log(f"  argmax = {oof_argmax:.5f}")
    log(f"  @recipe-bias = {tuned_bal:.5f}")
    log(f"  vs recipe_mlp (no OTE): 0.96177 / 0.96369")

    out = dict(
        oof_argmax=float(oof_argmax),
        oof_tuned_recipe_bias=float(tuned_bal),
        per_fold_argmax=[float(x) for x in fold_argmaxes],
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "recipe_mlp_ote_results.json").write_text(json.dumps(out, indent=2))
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
