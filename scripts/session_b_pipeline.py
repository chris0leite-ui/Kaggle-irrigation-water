"""Session B: multi-seed fold bagging — full greedy+nonrule pipeline at a
configurable FOLD_SEED.

Purpose: all existing OOFs share `StratifiedKFold(shuffle=True, random_state=42)`.
Every ensemble decision was made on ONE split. The OOF→LB calibration we've
been treating as the source of truth may be partly a lucky-split artifact.
This script runs the full LB-best pipeline at different fold seeds so we can
(a) estimate cross-split variance, (b) build a multi-seed bag that averages
test probs across independent fold assignments.

What it builds per run (one FOLD_SEED):
  1. xgb_dist_routed_v3 (rows with dgp_score in {0,1,2} routed to rule)
  2. xgb_specialist_678 (XGB on spec domain scores {6,7,8})
  3. xgb_nonrule (3-class XGB on 13 non-rule features)
  4. hybrid_v3 (routed_v3 overridden by spec_678 on {6,7,8})
  5. greedy log-blend (0.45 hybrid + 0.40 routed + 0.15 spec)
  6. LB-best log-blend (0.85 greedy + 0.15 nonrule)

XGB training `seed=42` is held CONSTANT across all runs — we're isolating
FOLD variance from MODEL variance (seed bag of XGB was already shown to be
near-deterministic on 2026-04-22).

Run:
  FOLD_SEED=7 python3 scripts/session_b_pipeline.py
  FOLD_SEED=123 python3 scripts/session_b_pipeline.py

Artefacts written per seed (suffix `_fs{seed}`):
  scripts/artifacts/oof_routed_v3_fs{seed}.npy
  scripts/artifacts/test_routed_v3_fs{seed}.npy
  scripts/artifacts/oof_spec_678_fs{seed}.npy
  scripts/artifacts/test_spec_678_fs{seed}.npy
  scripts/artifacts/oof_nonrule_fs{seed}.npy
  scripts/artifacts/test_nonrule_fs{seed}.npy
  scripts/artifacts/oof_greedy_fs{seed}.npy
  scripts/artifacts/test_greedy_fs{seed}.npy
  scripts/artifacts/oof_lb_best_fs{seed}.npy
  scripts/artifacts/test_lb_best_fs{seed}.npy
  scripts/artifacts/session_b_fs{seed}.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


FOLD_SEED = int(os.environ.get("FOLD_SEED", "42"))
XGB_SEED = 42           # held constant
N_FOLDS = 5
ROUTED_SCORES = (0, 1, 2)
SPEC_SCORES = (6, 7, 8)
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}

ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)
    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)
    return out


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def log_blend(probs_list, weights, eps=1e-9):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    logits = np.zeros_like(probs_list[0])
    for wi, p in zip(w, probs_list):
        logits += wi * np.log(np.clip(p, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


def train_routed(X, y, X_test, tr_scores, te_scores, skf, dist_feat_cols):
    """Same as xgb_dist_routed_v3: drop scores {0,1,2} from train, route at predict."""
    tr_routed_mask = np.isin(tr_scores, ROUTED_SCORES)
    te_routed_mask = np.isin(te_scores, ROUTED_SCORES)
    xgb_params = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=XGB_SEED,
    )
    rule_prob_low = np.array([1.0 - 2e-9, 1e-9, 1e-9], dtype=np.float64)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
    test_pred_xgb = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_filt = tr_idx[~np.isin(tr_scores[tr_idx], ROUTED_SCORES)]
        va_filt = va_idx[~np.isin(tr_scores[va_idx], ROUTED_SCORES)]
        dtr = xgb.DMatrix(X.iloc[tr_filt], label=y[tr_filt], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_filt], label=y[va_filt], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        dva_full = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        val_pred_all = booster.predict(dva_full, iteration_range=(0, bi + 1))
        va_mask = tr_routed_mask[va_idx]
        oof[va_idx[~va_mask]] = val_pred_all[~va_mask]
        oof[va_idx[va_mask]] = rule_prob_low
        test_pred_xgb += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        log(f"  [routed] fold {fold+1}/{N_FOLDS}  best_iter={bi}  ({time.time()-t0:.1f}s)")
    test_pred = test_pred_xgb.copy()
    test_pred[te_routed_mask] = rule_prob_low
    return oof, test_pred, best_iters


def train_spec_678(X, y, X_test, tr_scores, te_scores, skf):
    """3-class XGB trained only on scores {6,7,8} rows; predicts only on those rows."""
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)
    xgb_params = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=XGB_SEED,
    )
    oof_spec = np.zeros((len(X), 3), dtype=np.float64)
    test_spec = np.zeros((len(X_test), 3), dtype=np.float64)
    dte_spec = xgb.DMatrix(X_test.iloc[te_spec_mask], enable_categorical=True)
    spec_idx_te = np.where(te_spec_mask)[0]
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_sp = tr_idx[np.isin(tr_scores[tr_idx], SPEC_SCORES)]
        va_sp = va_idx[np.isin(tr_scores[va_idx], SPEC_SCORES)]
        if len(tr_sp) == 0 or len(va_sp) == 0:
            log(f"  [spec] fold {fold+1}/{N_FOLDS}  empty spec subset; skipping")
            continue
        dtr = xgb.DMatrix(X.iloc[tr_sp], label=y[tr_sp], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_sp], label=y[va_sp], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        oof_spec[va_sp] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred = booster.predict(dte_spec, iteration_range=(0, bi + 1))
        for i, pos in enumerate(spec_idx_te):
            test_spec[pos] += test_pred[i] / N_FOLDS
        log(f"  [spec] fold {fold+1}/{N_FOLDS}  best_iter={bi}  n_tr={len(tr_sp)}  "
            f"n_va={len(va_sp)}  ({time.time()-t0:.1f}s)")
    return oof_spec, test_spec, best_iters, tr_spec_mask, te_spec_mask


def train_nonrule(tr, te, y, skf, known_cat_cols):
    """Non-rule 3-class XGB. Strip rule cols + all distance-derived cols.
    Categorical cols are passed by name (they've already been int-mapped
    upstream; we just need to re-cast them to 'category' for XGB).
    """
    # non-rule allowed set: everything except rule cols + distance-derived + id/target
    engineered_rule_cols = {
        "dry", "norain", "hot", "windy", "nomulch", "kc_active",
        "dgp_score", "rule_pred",
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "score_dist_low_mid", "score_dist_mid_high",
        "min_boundary_dist", "min_axis_abs",
        "sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc",
    }
    nonrule_cols = [c for c in tr.columns
                    if c not in RULE_COLS
                    and c not in (TARGET, ID)
                    and c not in engineered_rule_cols]
    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()
    # re-cast original categoricals (now int32) back to 'category' dtype for XGB
    cat_cols = [c for c in nonrule_cols if c in known_cat_cols]
    num_cols = [c for c in nonrule_cols if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    log(f"  [nonrule] features: {X.shape[1]} ({len(num_cols)} num + {len(cat_cols)} cat)")
    log(f"  [nonrule] cat_cols: {cat_cols}")
    xgb_params = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=XGB_SEED,
    )
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        log(f"  [nonrule] fold {fold+1}/{N_FOLDS}  best_iter={bi}  ({time.time()-t0:.1f}s)")
    return oof, test_pred, best_iters


def main():
    log(f"=== Session B pipeline at FOLD_SEED={FOLD_SEED} (XGB_SEED={XGB_SEED}) ===")
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values

    # dist feature set (matches benchmark_xgb_dist.py / xgb_dist_routed_v3.py)
    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")
    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    log(f"dist feature set: {len(feat_cols)} cols")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    # --- Component 1: routed_v3 ---
    log("\n--- training xgb_dist_routed_v3 ---")
    t0 = time.time()
    oof_routed, test_routed, bi_routed = train_routed(X, y, X_test, tr_scores, te_scores, skf, feat_cols)
    log(f"routed_v3 done in {time.time()-t0:.1f}s")

    bias_r, tuned_r = tune_log_bias(oof_routed, y, prior)
    log(f"routed_v3 tuned OOF = {tuned_r:.5f}")

    # --- Component 2: spec_678 ---
    log("\n--- training xgb_specialist_678 ---")
    t0 = time.time()
    oof_spec, test_spec, bi_spec, tr_spec_mask, te_spec_mask = train_spec_678(
        X, y, X_test, tr_scores, te_scores, skf
    )
    log(f"spec_678 done in {time.time()-t0:.1f}s")

    # spec-domain eval
    spec_y = y[tr_spec_mask]
    spec_oof = oof_spec[tr_spec_mask]
    spec_argmax_bal = balanced_accuracy_score(spec_y, spec_oof.argmax(axis=1))
    log(f"spec_678 in-domain argmax bal_acc = {spec_argmax_bal:.5f}")

    # --- Component 3: xgb_nonrule ---
    log("\n--- training xgb_nonrule ---")
    t0 = time.time()
    oof_nr, test_nr, bi_nr = train_nonrule(tr, te, y, skf, set(cat_cols))
    log(f"nonrule done in {time.time()-t0:.1f}s")

    bias_nr, tuned_nr = tune_log_bias(oof_nr, y, prior)
    log(f"nonrule tuned OOF = {tuned_nr:.5f}")

    # --- Build hybrid_v3: routed OOF with spec override on {6,7,8} ---
    oof_hybrid = oof_routed.copy()
    oof_hybrid[tr_spec_mask] = oof_spec[tr_spec_mask]
    test_hybrid = test_routed.copy()
    test_hybrid[te_spec_mask] = test_spec[te_spec_mask]
    bias_h, tuned_h = tune_log_bias(oof_hybrid, y, prior)
    log(f"hybrid_v3 tuned OOF = {tuned_h:.5f}")

    # --- Build greedy: log-blend 0.45 hybrid + 0.40 routed + 0.15 spec ---
    # For spec component in greedy: need full-row probs. Rows outside spec domain
    # fall back to rule_prob_low (consistent with routing strategy).
    rule_prob_low = np.array([1.0 - 2e-9, 1e-9, 1e-9], dtype=np.float64)
    oof_spec_full = oof_spec.copy()
    oof_spec_full[~tr_spec_mask] = rule_prob_low
    test_spec_full = test_spec.copy()
    test_spec_full[~te_spec_mask] = rule_prob_low

    oof_greedy = log_blend([oof_hybrid, oof_routed, oof_spec_full], [0.45, 0.40, 0.15])
    test_greedy = log_blend([test_hybrid, test_routed, test_spec_full], [0.45, 0.40, 0.15])
    bias_g, tuned_g = tune_log_bias(oof_greedy, y, prior)
    log(f"greedy tuned OOF = {tuned_g:.5f}")

    # --- Build LB-best: 0.85 greedy + 0.15 nonrule (log-blend) ---
    oof_lb = log_blend([oof_greedy, oof_nr], [0.85, 0.15])
    test_lb = log_blend([test_greedy, test_nr], [0.85, 0.15])
    bias_lb, tuned_lb = tune_log_bias(oof_lb, y, prior)
    log(f"LB-best (greedy+nonrule) tuned OOF = {tuned_lb:.5f}")

    # Confusion matrix on LB-best
    log_lb = np.log(np.clip(oof_lb, 1e-9, 1.0))
    pred_lb = (log_lb + bias_lb).argmax(axis=1)
    cm = confusion_matrix(y, pred_lb)
    log(f"LB-best confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # --- Save per-seed artefacts ---
    suffix = f"fs{FOLD_SEED}"
    np.save(ART / f"oof_routed_v3_{suffix}.npy", oof_routed)
    np.save(ART / f"test_routed_v3_{suffix}.npy", test_routed)
    np.save(ART / f"oof_spec_678_{suffix}.npy", oof_spec)
    np.save(ART / f"test_spec_678_{suffix}.npy", test_spec)
    np.save(ART / f"oof_nonrule_{suffix}.npy", oof_nr)
    np.save(ART / f"test_nonrule_{suffix}.npy", test_nr)
    np.save(ART / f"oof_greedy_{suffix}.npy", oof_greedy)
    np.save(ART / f"test_greedy_{suffix}.npy", test_greedy)
    np.save(ART / f"oof_lb_best_{suffix}.npy", oof_lb)
    np.save(ART / f"test_lb_best_{suffix}.npy", test_lb)

    results = {
        "fold_seed": FOLD_SEED,
        "xgb_seed": XGB_SEED,
        "n_folds": N_FOLDS,
        "class_priors": prior.tolist(),
        "best_iters_routed_v3": bi_routed,
        "best_iters_spec_678": bi_spec,
        "best_iters_nonrule": bi_nr,
        "tuned_oof": {
            "routed_v3": tuned_r,
            "spec_678_in_domain_argmax": float(spec_argmax_bal),
            "nonrule": tuned_nr,
            "hybrid_v3": tuned_h,
            "greedy": tuned_g,
            "lb_best_greedy_nonrule": tuned_lb,
        },
        "log_bias": {
            "routed_v3": bias_r.tolist(),
            "hybrid_v3": bias_h.tolist(),
            "greedy": bias_g.tolist(),
            "lb_best": bias_lb.tolist(),
        },
        "confusion_matrix_lb_best": cm.tolist(),
    }
    with open(ART / f"session_b_{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> {ART}/session_b_{suffix}.json and OOF/test artefacts (_{suffix})")


if __name__ == "__main__":
    main()
