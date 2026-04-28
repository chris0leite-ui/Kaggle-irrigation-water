"""MLP meta-stacker with macro-recall surrogate gradient loss.

Combines two structurally distinct mechanisms:
  - MLP architecture (smooth optimization, no discrete tree splits) — the
    only meta architecture with positive OOF→LB transfer outside of XGB
    (mlp_metastack v1: LB 0.98091, gap +0.00027 = 5x tighter than LR-meta).
  - Macro-recall surrogate loss (boundary-row gradient peaked at p_true=0.5)
    — first training-time mechanism on this comp to produce clean ADD-High
    direction at the blend level (recipe_macrorecall.py 27th-saturation
    closure note: "G4 PASSES cleanly at every alpha").

Hypothesis: MLP's smooth optimization landscape may handle the macrorec
gradient differently from XGB's discrete tree splits. XGB-meta-macrorec
saturated because best_iter=3 (gradient satiates instantly on a
near-perfect bank); MLP iterates 30 epochs over the WHOLE dataset and
the boundary-row gradient stays informative throughout.

Loss: L = lam_ce * CE_balanced + (1 - lam_ce) * (-R_macro_surrogate)
  where R_macro_surrogate = (1/K) sum_k (1/N_k) sum_{i: y_i=k} p_ik
  (= mean over classes of the within-class soft-recall)

Output: oof_mlp_meta_macrorec_lam{LAM}.npy + test counterpart.

Same input matrix as meta_l3_xgb_mlp.py (~206-dim):
  LB-best 3-stack log-probs (3) + 14 dist/rule meta cols + N pool components * 3 cls

5-fold StratifiedKFold(seed=42) for OOF alignment.
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

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    BIAS, build_lbbest_stack, iso_cal, _normed,
)
from tier1b_helpers import load_pool  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 2 if SMOKE else 5
TARGET = "Irrigation_Need"
EPS = 1e-12

# Loss-mix coefficient: 0.0 = pure macrorec surrogate, 1.0 = pure CE.
# Default 0.3 mirrors recipe_macrorecall.py's selected sweet spot.
LAM_CE = float(os.environ.get("LAM_CE", "0.3"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "5" if SMOKE else "30"))
# CURATED=1 → use the EXACT 62-component bank from the LB-best XGB-meta
# (defense against bank-extension OOF-overfit trap; CLAUDE.md
# 2026-04-26 cross-poll v3 LB regress confirmed this).
CURATED = os.environ.get("CURATED", "1") == "1"
SUFFIX = f"_lam{LAM_CE:g}".replace(".", "")
if CURATED:
    SUFFIX += "_curated"
OUT_NAME = f"mlp_meta_macrorec{SUFFIX}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1))


class MLPMeta(nn.Module):
    """Same architecture as meta_l3_xgb_mlp.py for direct comparison."""

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


def macrorec_loss(logits, y, n_class, lam_ce, class_weight):
    """L = lam_ce * CE_balanced + (1 - lam_ce) * (-R_macro_surrogate).

    R_macro = (1/K) * mean_k of [ sum_{i: y_i=k} p_ik / N_k ]
            = (1/K) * mean over k of within-class soft-recall

    Implementation: build mask matrix (N, K) of one-hot y.
    p_class_k = (mask * p).sum(0) gives per-class probability mass.
    Divide by mask.sum(0) (clip to >=1) to get within-class recall.
    Mean over K -> macro-recall surrogate.
    """
    log_p = F.log_softmax(logits, dim=1)  # (N, K)
    p = log_p.exp()  # (N, K)
    # Per-class balanced CE
    ce = F.nll_loss(log_p, y, weight=class_weight)
    # Macro-recall surrogate
    onehot = F.one_hot(y, num_classes=n_class).float()  # (N, K)
    class_count = onehot.sum(0).clamp(min=1.0)  # (K,)
    p_true_per_class = (onehot * p).sum(0)  # (K,) sum over rows where y=k of p_ik
    recall_per_class = p_true_per_class / class_count  # (K,)
    R = recall_per_class.mean()
    mr_loss = -R
    return lam_ce * ce + (1.0 - lam_ce) * mr_loss


def train_mlp(X_tr, y_tr, X_va, y_va, n_class=3, n_epochs=30, batch=4096,
              lr=1e-3, wd=1e-5, lam_ce=0.3, seed=42):
    """Train MLP with macrorec-mixed loss. Returns val + test preds."""
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
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    Xt_t = torch.tensor(Xt, dtype=torch.float32, device=DEVICE)
    yt_t = torch.tensor(y_tr.astype(np.int64), dtype=torch.long, device=DEVICE)

    n = len(Xt_t)
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            logits = model(Xt_t[b])
            loss = macrorec_loss(logits, yt_t[b], n_class, lam_ce, cw_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == n_epochs - 1:
            log(f"    epoch {epoch+1}/{n_epochs}  avg_loss={epoch_loss/n_batches:.5f}")

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
    log(f"config: SMOKE={SMOKE}  LAM_CE={LAM_CE}  N_EPOCHS={N_EPOCHS}  "
        f"DEVICE={DEVICE}  SUFFIX={SUFFIX}")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    if SMOKE:
        sub = np.random.default_rng(SEED).choice(len(train), size=20_000, replace=False)
        sub.sort()
        train = train.iloc[sub].reset_index(drop=True)
        y_full = pd.read_csv(DATA / "train.csv")[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
        y = y[sub] if False else y_full[sub]
        log(f"  SMOKE: subsampled to {len(train)}")

    log("building LB-best 3-stack anchor")
    if SMOKE:
        # build_lbbest_stack on full data, then subset
        y_full = pd.read_csv(DATA / "train.csv")[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
        lb_oof_full, lb_test = build_lbbest_stack(y_full)
        lb_oof = lb_oof_full[sub]
        # Also subset test for SMOKE
        lb_test = lb_test[: 5000]
    else:
        lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best 3-stack OOF bal@bias = {bal(lb_oof, y):.5f}")

    log("loading pool")
    pool = load_pool()
    if CURATED:
        # Use EXACT same 62-component bank as the LB-best XGB-meta-stacker
        # (loaded from tier1b_xgb_metastack_results.json's components list).
        # Defense against the bank-extension trap documented in CLAUDE.md.
        meta_results = json.loads((ART / "tier1b_xgb_metastack_results.json").read_text())
        target_components = set(meta_results["components"])
        before = len(pool)
        pool = {n: v for n, v in pool.items() if n in target_components}
        missing = target_components - set(pool.keys())
        if missing:
            log(f"  WARNING: {len(missing)} target components not loadable: {sorted(missing)[:5]}")
        log(f"  CURATED: filtered pool {before} → {len(pool)} components "
            f"(target=62, missing={len(missing)})")
    if SMOKE:
        pool = {n: (o[sub], t[: 5000]) for n, (o, t) in pool.items()}
    log(f"  {len(pool)} 3-class components loaded")

    # Build meta-feature matrix.
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
    log(f"  meta-feature shape: {X_tr.shape}  ({len(component_names)} components × 3 cls)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_mlp = np.zeros((len(train), 3), dtype=np.float32)
    test_mlp = np.zeros((len(X_te), 3), dtype=np.float32)

    # Per-fold checkpoint pattern.
    ck_prefix = OUT_NAME + ("_smoke" if SMOKE else "")
    cached: set[int] = set()
    for fold_check in range(1, N_FOLDS + 1):
        ck_oof = ART / f"oof_{ck_prefix}_fold{fold_check}.npy"
        ck_test = ART / f"test_{ck_prefix}_fold{fold_check}.npy"
        if ck_oof.exists() and ck_test.exists():
            cached.add(fold_check)
    if cached:
        log(f"  resume: {len(cached)} fold(s) cached: {sorted(cached)}")

    fold_argmax_balaccs = []
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
            fold_argmax_balaccs.append(float(b_arg))
            log(f"  fold {fold+1} CACHED  val_argmax_bal = {b_arg:.5f}")
            continue
        vp, mu, sd, model = train_mlp(
            X_tr[tr_idx], y[tr_idx], X_tr[va_idx], y[va_idx],
            n_epochs=N_EPOCHS, lam_ce=LAM_CE, seed=SEED + fold)
        tp = predict_mlp(model, X_te, mu, sd).astype(np.float32)
        oof_mlp[va_idx] = vp
        test_mlp += tp / N_FOLDS
        np.save(ART / f"oof_{ck_prefix}_fold{fold+1}.npy", vp)
        np.save(ART / f"test_{ck_prefix}_fold{fold+1}.npy", tp)
        b_arg = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_argmax_balaccs.append(float(b_arg))
        log(f"  fold {fold+1} val_argmax_bal = {b_arg:.5f}  wall = {time.time()-t1:.1f}s")

    np.save(ART / f"oof_{OUT_NAME}.npy", oof_mlp)
    np.save(ART / f"test_{OUT_NAME}.npy", test_mlp)

    mlp_argmax = balanced_accuracy_score(y, oof_mlp.argmax(1))
    mlp_tuned = bal(oof_mlp, y)
    log(f"\n=== {OUT_NAME} STANDALONE ===")
    log(f"  argmax bal_acc       = {mlp_argmax:.5f}")
    log(f"  @recipe-bias bal_acc = {mlp_tuned:.5f}")

    if SMOKE:
        log("SMOKE complete; skipping iso-cal + blend gate (full bank not used)")
        return

    # Iso-cal then blend gate vs LB-best 3-stack.
    log("\n=== ISO-CAL + BLEND GATE vs LB-best 3-stack ===")
    mlp_iso_oof, mlp_iso_test = iso_cal(oof_mlp, test_mlp, y)
    np.save(ART / f"oof_{OUT_NAME}_iso.npy", mlp_iso_oof)
    np.save(ART / f"test_{OUT_NAME}_iso.npy", mlp_iso_test)

    lb_bal = bal(lb_oof, y)
    log(f"  LB-best 3-stack OOF: {lb_bal:.5f}")
    mlp_iso_tuned = bal(mlp_iso_oof, y)
    log(f"  MLP-meta-macrorec iso @ recipe-bias: {mlp_iso_tuned:.5f}")

    log(f"\n  alpha-sweep (REPLACE-v1 architecture: lb_3stack + alpha * mlp_iso)")
    log(f"  {'alpha':>6} {'OOF':>9} {'Δ vs LB-3':>10} {'recH':>7}")
    rows = []
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    for a in alphas:
        blend = log_blend([lb_oof, mlp_iso_oof], np.array([1 - a, a]))
        b_blend = bal(blend, y)
        d = b_blend - lb_bal
        # Per-class recall
        pred_blend = (np.log(np.clip(blend, EPS, 1.0)) + BIAS).argmax(1)
        rec_h = ((pred_blend == 2) & (y == 2)).sum() / (y == 2).sum()
        rows.append({"alpha": a, "oof": float(b_blend), "delta": float(d),
                     "recH": float(rec_h)})
        tag = " ← peak" if rows and d == max(r["delta"] for r in rows) else ""
        log(f"  {a:>6.3f} {b_blend:>9.5f} {d:>+10.5f} {rec_h:>7.4f}{tag}")

    out = dict(
        config=dict(LAM_CE=LAM_CE, N_EPOCHS=N_EPOCHS, SEED=SEED),
        n_components=len(component_names),
        components=component_names,
        feature_dim=int(X_tr.shape[1]),
        per_fold_argmax=fold_argmax_balaccs,
        mlp_standalone_argmax=float(mlp_argmax),
        mlp_standalone_tuned=float(mlp_tuned),
        mlp_iso_tuned=float(mlp_iso_tuned),
        lb_best_3stack_oof=float(lb_bal),
        alpha_sweep=rows,
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / f"{OUT_NAME}_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"\nwrote {json_path}")
    log(f"total wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
