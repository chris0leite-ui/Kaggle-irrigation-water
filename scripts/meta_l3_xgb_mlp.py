"""Experiment B: L3 weighted average of XGB-meta + small MLP-meta.

Cdeotte's published recipe: "Multiple Level 2 models, e.g. GBDT meta + NN
meta, then Level 3 weighted average". We have the XGB meta on disk
(oof_xgb_metastack.npy → LB-best 4-stack at α=0.30 = LB 0.98094). This
script trains a small MLP meta-learner on the SAME meta-feature matrix
(63 components × 3 cls + 14 dist features + LB-best 3-stack 3 cls = ~206
dims), then blends XGB-meta and MLP-meta at L3 via fixed weighted average,
isotonic-calibrates the L3 output, and stacks into LB-best 3-stack.

Why MLP-meta has a chance where 15 NN-on-recipe nulls: input dim is ~206
not 443+ recipe cols → much smaller capacity-vs-info ratio → less prone
to magnitude trap. Inductive bias differs from XGB depth-4 trees.

5-fold StratifiedKFold(seed=42), aligned with every other OOF.
Outputs:
  oof_mlp_metastack.npy / test_mlp_metastack.npy
  oof_meta_l3_xgb_mlp.npy / test_meta_l3_xgb_mlp.npy
  meta_l3_xgb_mlp_results.json (per-α blend sweep + per-class recall +
  gate decision vs LB-best 4-stack)
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

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    BIAS, build_lbbest_stack, iso_cal, _normed,
)
from tier1b_helpers import load_pool  # shape-filtered (excludes partial-fold artefacts)  # noqa: E402

import os
SMOKE = os.environ.get("SMOKE") == "1"
ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 2 if SMOKE else 5
TARGET = "Irrigation_Need"
EPS = 1e-12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1))


class MLPMeta(nn.Module):
    """Small MLP for meta-stacking. ~50k params on 206-dim input."""

    def __init__(self, in_dim, hidden=(128, 64, 32), n_class=3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, n_class)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, X_va, y_va, n_class=3, n_epochs=30, batch=4096,
              lr=1e-3, wd=1e-5, seed=42):
    """Train a single MLP on standardised meta features. Returns val + test preds."""
    torch.manual_seed(seed)
    mu = X_tr.mean(0, keepdims=True)
    sd = X_tr.std(0, keepdims=True) + 1e-6
    Xt = ((X_tr - mu) / sd).astype(np.float32)
    Xv = ((X_va - mu) / sd).astype(np.float32)

    model = MLPMeta(Xt.shape[1], hidden=(128, 64, 32), n_class=n_class).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    cw = np.bincount(y_tr, minlength=n_class).astype(np.float32)
    cw = cw.sum() / np.maximum(cw, 1) / n_class
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE))

    Xt_t = torch.tensor(Xt, dtype=torch.float32, device=DEVICE)
    yt_t = torch.tensor(y_tr.astype(np.int64), dtype=torch.long, device=DEVICE)

    n = len(Xt_t)
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            logits = model(Xt_t[b])
            loss = loss_fn(logits, yt_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        Xv_t = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
        vp = torch.softmax(model(Xv_t), dim=1).cpu().numpy()
    return vp, mu, sd, model


def predict_mlp(model, X, mu, sd):
    Xs = ((X - mu) / sd).astype(np.float32)
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(Xs_t), dim=1).cpu().numpy()


def main():
    t0 = time.time()
    log(f"config: SMOKE={SMOKE}  N_FOLDS={N_FOLDS}  DEVICE={DEVICE}")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    if SMOKE:
        sub = np.random.default_rng(SEED).choice(len(train), size=20_000, replace=False)
        train = train.iloc[sub].reset_index(drop=True)
        y = y[sub]
        log(f"  SMOKE: subsampled to {len(train)}")

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y if not SMOKE else
                                          pd.read_csv(DATA / "train.csv")[TARGET].map(CLS2IDX).to_numpy().astype(np.int32))
    if SMOKE:
        lb_oof = lb_oof[sub]
        lb_test = lb_test[: 5000]  # placeholder
    log(f"  LB-best 3-stack OOF bal@bias = {bal(lb_oof, y):.5f}")

    log("loading pool")
    pool = load_pool()
    if SMOKE:
        pool = {n: (o[sub], t[: 5000]) for n, (o, t) in pool.items()}
    log(f"  {len(pool)} 3-class components loaded")

    # Build meta-feature matrix (mirrors tier1b_xgb_metastack.py exactly).
    log("constructing meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test if not SMOKE else test.head(5000))
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float32)
    log(f"  meta-feature shape: {X_tr.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_mlp = np.zeros((len(train), 3), dtype=np.float32)
    test_mlp = np.zeros((len(X_te), 3), dtype=np.float32)
    n_epochs = 5 if SMOKE else 30

    # Per-fold checkpoint pattern (rehydrate-resilient).
    ck_prefix = "mlp_metastack" + ("_smoke" if SMOKE else "")
    cached: set[int] = set()
    for fold_check in range(1, N_FOLDS + 1):
        ck_oof = ART / f"oof_{ck_prefix}_fold{fold_check}.npy"
        ck_test = ART / f"test_{ck_prefix}_fold{fold_check}.npy"
        if ck_oof.exists() and ck_test.exists():
            cached.add(fold_check)
    if cached:
        log(f"  resume: {len(cached)} fold(s) cached: {sorted(cached)}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        log(f"=== fold {fold+1}/{N_FOLDS} ===")
        if (fold + 1) in cached:
            ck_oof = ART / f"oof_{ck_prefix}_fold{fold+1}.npy"
            ck_test = ART / f"test_{ck_prefix}_fold{fold+1}.npy"
            vp = np.load(ck_oof)
            tp = np.load(ck_test)
            oof_mlp[va_idx] = vp
            test_mlp += tp / N_FOLDS
            b_arg = balanced_accuracy_score(y[va_idx], vp.argmax(1))
            log(f"  fold {fold+1} CACHED  val_argmax_bal_acc = {b_arg:.5f}")
            continue
        vp, mu, sd, model = train_mlp(
            X_tr[tr_idx], y[tr_idx], X_tr[va_idx], y[va_idx],
            n_epochs=n_epochs, seed=SEED + fold)
        tp = predict_mlp(model, X_te, mu, sd).astype(np.float32)
        oof_mlp[va_idx] = vp
        test_mlp += tp / N_FOLDS
        # checkpoint immediately for rehydrate resilience
        np.save(ART / f"oof_{ck_prefix}_fold{fold+1}.npy", vp)
        np.save(ART / f"test_{ck_prefix}_fold{fold+1}.npy", tp)
        b_arg = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        log(f"  fold {fold+1} val_argmax_bal_acc = {b_arg:.5f}  wall = {time.time()-t1:.1f}s")

    np.save(ART / "oof_mlp_metastack.npy", oof_mlp)
    np.save(ART / "test_mlp_metastack.npy", test_mlp)

    mlp_argmax = balanced_accuracy_score(y, oof_mlp.argmax(1))
    mlp_tuned = bal(oof_mlp, y)
    log(f"=== MLP-meta standalone ===")
    log(f"  argmax bal_acc        = {mlp_argmax:.5f}")
    log(f"  @recipe-bias bal_acc  = {mlp_tuned:.5f}")

    # Load XGB-meta from disk.
    if SMOKE:
        log("SMOKE: skipping L3 + blend gate (XGB-meta is on full 630k, not subset)")
        return
    xgb_meta_oof = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    xgb_meta_test = _normed(np.load(ART / "test_xgb_metastack.npy"))
    xgb_meta_iso_oof, xgb_meta_iso_test = iso_cal(xgb_meta_oof, xgb_meta_test, y)
    mlp_meta_iso_oof, mlp_meta_iso_test = iso_cal(oof_mlp, test_mlp, y)

    # L3: weighted average of XGB-meta-iso and MLP-meta-iso.
    # Sweep over W_MLP ∈ {0.1, 0.2, ..., 0.5} (XGB dominant since it produced LB-best).
    log(f"=== L3 weighted average XGB-meta-iso × MLP-meta-iso ===")
    log(f"{'W_MLP':>6} {'L3 OOF':>9} {'4st OOF':>9} {'4st Δ':>9} {'best_α':>8} {'best Δ':>9}")
    rows = []
    alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    lb_bal = bal(lb_oof, y)
    # 4-stack reference (using XGB-meta-iso alone at α=0.30)
    lb4_oof = log_blend([lb_oof, xgb_meta_iso_oof], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_oof, y)
    log(f"reference: LB-best 3-stack OOF={lb_bal:.5f}  4-stack OOF={lb4_bal:.5f}")
    for w_mlp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # L3 weighted average in PROBABILITY space (cdeotte uses arithmetic, not log).
        l3_oof = (1 - w_mlp) * xgb_meta_iso_oof + w_mlp * mlp_meta_iso_oof
        l3_oof = l3_oof / l3_oof.sum(axis=1, keepdims=True)
        # Stack into LB-best 3-stack via log-blend at α-sweep.
        best_a, best_d = 0.0, 0.0
        for a in alphas:
            blend = log_blend([lb_oof, l3_oof], np.array([1 - a, a]))
            d = bal(blend, y) - lb4_bal
            if d > best_d:
                best_a, best_d = a, d
        # Build the best blend to extract per-class recall.
        blend = log_blend([lb_oof, l3_oof], np.array([1 - best_a, best_a]))
        pred = (np.log(np.clip(blend, EPS, 1)) + BIAS).argmax(1)
        pcr = np.array([(pred[y == k] == k).mean() for k in range(3)])
        pred_anchor = (np.log(np.clip(lb4_oof, EPS, 1)) + BIAS).argmax(1)
        pcr_anchor = np.array([(pred_anchor[y == k] == k).mean() for k in range(3)])
        pcr_delta = pcr - pcr_anchor
        rows.append({
            "w_mlp": w_mlp, "best_alpha": best_a,
            "blend_oof": float(bal(blend, y)),
            "delta_4stack": float(best_d),
            "pcr_delta": pcr_delta.tolist(),
            "pcr_pass": bool((pcr_delta >= -5e-4).all()),
            "errs": int((pred != y).sum()),
        })
        log(f"{w_mlp:>6.2f} {bal(l3_oof, y):>9.5f} {bal(blend, y):>9.5f} "
            f"{best_d:>+9.5f} {best_a:>8.3f} {best_d:>+9.5f}")

    best = max(rows, key=lambda r: r["delta_4stack"])
    log(f"\nBEST: w_mlp={best['w_mlp']}  α={best['best_alpha']}  Δ={best['delta_4stack']:+.5f}  "
        f"PCR pass={best['pcr_pass']}")
    gate_pass = bool(best["delta_4stack"] >= 2e-4 and best["pcr_pass"])
    log(f"GATE: {'PASS' if gate_pass else 'FAIL'} (need Δ ≥ +2e-4 AND PCR ≥ -5e-4)")

    # Save L3 OOF + test for the best w_mlp (NOT a submission — diagnostic only).
    if best["w_mlp"] > 0:
        w = best["w_mlp"]
        l3_oof = (1 - w) * xgb_meta_iso_oof + w * mlp_meta_iso_oof
        l3_test = (1 - w) * xgb_meta_iso_test + w * mlp_meta_iso_test
        l3_oof = l3_oof / l3_oof.sum(axis=1, keepdims=True)
        l3_test = l3_test / l3_test.sum(axis=1, keepdims=True)
        np.save(ART / "oof_meta_l3_xgb_mlp.npy", l3_oof.astype(np.float32))
        np.save(ART / "test_meta_l3_xgb_mlp.npy", l3_test.astype(np.float32))

    out = dict(
        config=dict(n_folds=N_FOLDS, n_epochs=n_epochs, hidden=(128, 64, 32)),
        n_components=len(component_names),
        feature_dim=int(X_tr.shape[1]),
        mlp_argmax=float(mlp_argmax),
        mlp_tuned=float(mlp_tuned),
        lb_best_3stack_oof=float(lb_bal),
        lb_best_4stack_oof=float(lb4_bal),
        rows=rows,
        best=best,
        gate_pass=gate_pass,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "meta_l3_xgb_mlp_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote oof_mlp_metastack.npy + test + meta_l3_xgb_mlp_results.json")


if __name__ == "__main__":
    main()
