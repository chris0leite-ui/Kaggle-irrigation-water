"""R5: LightGBM-meta with class_weight=None on LB-best 7-component bank.

Different L2 model class than RF natural (LB 0.98129) and XGB-meta v1
(Tier-1b, OOF 0.98041). LightGBM is gradient-boosted (closer to XGB)
but with leaf-wise growth (different tree topology). With:
  - objective='multiclass'
  - is_unbalance=False (no class re-balancing)
  - bagging_fraction=0.8 + bagging_freq=5 (similar to RF bootstrap)
  - light reg (no L1/L2)

R5 LIFTS  → LightGBM produces a different operating point
            than RF natural that beats LB 0.98129.
R5 NULLS  → LightGBM converges to similar Pareto corner as RF/XGB.
R5 REGRSS → leaf-wise GBDT loses to bagging on natural-cal bank.

Same 5-fold seed=42 split, same 7-component bank as LB 0.98129.

Outputs (suffix _lgbm_natural):
  scripts/artifacts/oof_lgbm_meta_natural.npy
  scripts/artifacts/test_lgbm_meta_natural.npy
  scripts/artifacts/lgbm_meta_natural_results.json
  submissions/submission_lgbm_meta_natural_standalone.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = "_lgbm_natural"

# 7 LB-best components — frozen identical to LB 0.98129 setup
BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
]

META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def load_bank(n_tr, n_te):
    log(f"loading LB-best 7-component bank for LGBM-meta")
    pool = {}
    for name in BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  MISSING {name}")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape != (n_tr, 3) or t.shape != (n_te, 3):
            log(f"  SKIP {name}: shape mismatch")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (normed(o), normed(t))
        log(f"  + {name}")
    return pool


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(n_tr, n_te)
    if len(pool) != len(BANK):
        log(f"ERROR: pool size {len(pool)} != expected {len(BANK)} — abort")
        return

    log("building feature matrix")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)
    names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in names]
    log_te = [safelog(pool[n][1]) for n in names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  components={len(names)}")

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    del X_tr, X_te
    import gc; gc.collect()

    n_est = 100 if SMOKE else 500
    n_folds = 2 if SMOKE else 5
    if SMOKE:
        sub_idx = np.arange(50_000)
        X_tr_s = X_tr_s[sub_idx]
        y_use = y[sub_idx]
        n_tr_use = len(sub_idx)
    else:
        y_use = y
        n_tr_use = n_tr

    # LightGBM hyperparameters mirroring "natural-cal" target:
    #  - no class re-balancing (is_unbalance=False, no class_weight)
    #  - bagging via bagging_fraction + freq (analog to RF bootstrap)
    #  - heavy min_data_in_leaf for stable per-leaf class counts
    #  - small leaf size + low LR + bagging → "RF-like" regularization
    lgb_params = dict(
        objective="multiclass",
        num_class=3,
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        feature_fraction=1.0,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=0.0,
        lambda_l2=0.0,
        verbose=-1,
        seed=SEED,
        n_jobs=-1,
    )
    log(f"config: n_folds={n_folds} n_estimators={n_est} {lgb_params}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y_use))
    oof = np.zeros((n_tr_use, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        train_set = lgb.Dataset(X_tr_s[tr_idx], y_use[tr_idx])
        val_set = lgb.Dataset(X_tr_s[va_idx], y_use[va_idx], reference=train_set)
        model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=n_est,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        best_it = model.best_iteration
        p_va = model.predict(X_tr_s[va_idx], num_iteration=best_it).astype(np.float32)
        p_te = model.predict(X_te_s, num_iteration=best_it).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y_use[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        best_iters.append(int(best_it))
        log(f"  fold {fold}/{n_folds} argmax={bal:.5f}  best_iter={best_it}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y_use, oof.argmax(1))
    prior = np.bincount(y_use, minlength=3) / len(y_use)
    bias, tuned = tune_log_bias(oof, y_use, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y_use, pred_at_bias)
    errs = (pred_at_bias != y_use).sum()
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]  errs={int(errs)}")

    # Drift diagnostic
    neg_log_prior = -np.log(prior)
    drift = bias - neg_log_prior
    log(f"  -log(prior) = {[round(float(p), 4) for p in neg_log_prior]}")
    log(f"  drift       = {[round(float(d), 4) for d in drift]}  max|drift|={float(np.abs(drift).max()):.4f}")

    suffix = "_smoke" if SMOKE else SUFFIX
    np.save(ART / f"oof_lgbm_meta_natural{suffix.replace('_lgbm_natural', '')}.npy", oof)
    np.save(ART / f"test_lgbm_meta_natural{suffix.replace('_lgbm_natural', '')}.npy", test_pred)

    summary = dict(
        smoke=SMOKE, n_folds=n_folds, n_estimators=n_est, seed=SEED,
        bank=BANK, bank_loaded=names,
        feature_count=len(names) * 3 + len(META_COLS),
        lgb_params=lgb_params,
        fold_scores=fold_scores,
        best_iters=best_iters,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        bias_H=float(bias[2]),
        drift=drift.tolist(),
        max_abs_drift=float(np.abs(drift).max()),
        per_class_recall=pcr.tolist(),
        errs=int(errs),
    )
    out_p = ART / f"lgbm_meta_natural{suffix.replace('_lgbm_natural', '')}_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")

    if SMOKE:
        log("SMOKE — skipping diagnostic + submission emission")
        return

    # Diagnostic vs LB 0.98129
    log("=== vs LB 0.98129 (RF natural 7-component bank) ===")
    lb_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    lb_test = np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32)
    bias_lb, tuned_lb = tune_log_bias(lb_oof, y, prior)
    pred_lb = (safelog(lb_oof) + bias_lb).argmax(1)
    pcr_lb = per_class_recall(y, pred_lb)
    errs_lb = (pred_lb != y).sum()
    log(f"  LB 0.98129: tuned={tuned_lb:.5f}  bias={bias_lb.round(4).tolist()}  errs={int(errs_lb)}")
    log(f"  Δ tuned = {tuned - tuned_lb:+.5f}")
    log(f"  Δ PCR  = [L={pcr[0] - pcr_lb[0]:+.5f} M={pcr[1] - pcr_lb[1]:+.5f} H={pcr[2] - pcr_lb[2]:+.5f}]")
    log(f"  Δ errs = {int(errs) - int(errs_lb):+d}")

    test_pred_r5 = (safelog(test_pred) + bias).argmax(1)
    test_pred_lb = (safelog(lb_test) + bias_lb).argmax(1)
    rows_diff = (test_pred_r5 != test_pred_lb).sum()
    net_h = int(((test_pred_r5 == 2) & (test_pred_lb != 2)).sum() -
                ((test_pred_lb == 2) & (test_pred_r5 != 2)).sum())
    churn_h = int(((test_pred_r5 == 2) ^ (test_pred_lb == 2)).sum())
    log(f"  test rows diff: {int(rows_diff)}/{n_te}")
    log(f"  test net_H flip: {net_h:+d}  churn_H: {churn_h}")

    out_csv = SUB / f"submission_lgbm_meta_natural_standalone.csv"
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_r5],
    })
    sub.to_csv(out_csv, index=False)
    log(f"wrote candidate {out_csv}  (NOT LB-probed — awaiting user approval)")
    log(f"  class counts: {sub[TARGET].value_counts().to_dict()}")

    diag = dict(
        r5_tuned=float(tuned), lb_tuned=float(tuned_lb),
        delta_tuned=float(tuned - tuned_lb),
        delta_pcr=[float(pcr[i] - pcr_lb[i]) for i in range(3)],
        delta_errs=int(errs) - int(errs_lb),
        rows_diff=int(rows_diff),
        net_h=net_h, churn_h=churn_h,
        candidate_csv=str(out_csv),
        drift=drift.tolist(),
    )
    out_d = ART / f"lgbm_meta_natural_diag.json"
    out_d.write_text(json.dumps(diag, indent=2, default=float))


if __name__ == "__main__":
    main()
