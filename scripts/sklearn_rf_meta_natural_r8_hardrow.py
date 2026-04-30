"""R8: RF natural with hard-row sample weighting (bank-disagreement based).

Hypothesis from R7 closure: OOF argmax lift ≠ macro-recall lift on
this problem. To improve macro-recall, the new meta needs to fit
boundary rows BETTER, not majority rows MORE.

Mechanism: per-row sample weight derived from bank-component
disagreement. Rows where bank components confidently agree are
"easy" (down-weighted to 0.5). Rows where components disagree
are "hard" (up-weighted to 1.5). RF natural's capacity then
focuses on hard rows where macro-recall is decided.

Hard-row signal: 1 - mean(max_prob across 7 bank components)
  - All components agree, max_prob=1: weight 0.5 (easy, well-handled)
  - Components disagree, mean max_prob 0.5: weight 1.0 (mid)
  - Maximum disagreement, mean max_prob → 1/3: weight ~1.5 (hard)

Same 7-component LB-best bank, same RF HPs as LB 0.98129.

Outputs (suffix _r8_hardrow):
  oof_sklearn_rf_meta_natural_r8_hardrow.npy
  test_sklearn_rf_meta_natural_r8_hardrow.npy
  sklearn_rf_meta_natural_r8_hardrow_results.json
  submission_sklearn_rf_meta_natural_r8_hardrow_standalone.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
SUFFIX = "_r8_hardrow"

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
    log(f"loading LB-best 7-component bank")
    pool = {}
    for name in BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  MISSING {name}"); continue
        o = normed(np.load(oof_p).astype(np.float32))
        t = normed(np.load(test_p).astype(np.float32))
        pool[name] = (o, t)
        log(f"  + {name}")
    return pool


def hard_row_weight(pool):
    """Per-row weight: 1.5 - mean(max_prob across components).

    Rows where components agree confidently (max_prob ≈ 1.0): weight ≈ 0.5
    Rows where components disagree (max_prob ≈ 0.5): weight ≈ 1.0
    Maximum disagreement (max_prob ≈ 1/3): weight ≈ 1.17
    """
    names = sorted(pool.keys())
    max_probs = np.stack([pool[n][0].max(1) for n in names], axis=1)  # (n_tr, 7)
    mean_max = max_probs.mean(1)  # (n_tr,)
    weight = 1.5 - mean_max
    return weight, mean_max


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(n_tr, n_te)
    if len(pool) != len(BANK):
        log(f"ERROR: pool size {len(pool)} != {len(BANK)}"); return

    sw, mean_max = hard_row_weight(pool)
    log(f"hard-row weight stats:")
    log(f"  mean_max_prob across components: p25={np.percentile(mean_max, 25):.4f} "
        f"p50={np.percentile(mean_max, 50):.4f} p75={np.percentile(mean_max, 75):.4f}")
    log(f"  sample_weight: min={sw.min():.4f} mean={sw.mean():.4f} max={sw.max():.4f}")
    log(f"  weight distribution by class:")
    for k, name in enumerate(["Low", "Medium", "High"]):
        mask = y == k
        log(f"    {name:6s}: weight mean={sw[mask].mean():.4f}  std={sw[mask].std():.4f}")

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
    log(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    del X_tr, X_te
    import gc; gc.collect()

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    if SMOKE:
        sub_idx = np.arange(50_000)
        X_tr_s = X_tr_s[sub_idx]
        y_use = y[sub_idx]
        sw_use = sw[sub_idx]
        n_tr_use = len(sub_idx)
    else:
        y_use = y
        sw_use = sw
        n_tr_use = n_tr
    log(f"config: n_folds={n_folds} n_est={n_est} max_depth={max_depth} seed={SEED}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y_use))
    oof = np.zeros((n_tr_use, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_leaf=20, max_features="sqrt",
            bootstrap=True, n_jobs=-1, random_state=SEED,
            class_weight=None, verbose=0,
        )
        rf.fit(X_tr_s[tr_idx], y_use[tr_idx], sample_weight=sw_use[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y_use[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold}/{n_folds} argmax={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y_use, oof.argmax(1))
    prior = np.bincount(y_use, minlength=3) / len(y_use)
    bias, tuned = tune_log_bias(oof, y_use, prior)
    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y_use, pred_at_bias)
    errs = (pred_at_bias != y_use).sum()
    neg_log_prior = -np.log(prior)
    drift = bias - neg_log_prior
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")
    log(f"  drift = {drift.round(4).tolist()}  max|drift|={float(np.abs(drift).max()):.4f}")
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]  errs={int(errs)}")

    suffix = "_smoke" if SMOKE else SUFFIX
    np.save(ART / f"oof_sklearn_rf_meta_natural{suffix}.npy", oof)
    np.save(ART / f"test_sklearn_rf_meta_natural{suffix}.npy", test_pred)
    summary = dict(
        smoke=SMOKE, n_folds=n_folds, n_est=n_est, max_depth=max_depth, seed=SEED,
        bank=BANK, fold_scores=fold_scores,
        overall_argmax=float(overall), tuned_log_bias=float(tuned),
        log_bias=bias.tolist(), drift=drift.tolist(),
        per_class_recall=pcr.tolist(), errs=int(errs),
        sw_min=float(sw.min()), sw_mean=float(sw.mean()), sw_max=float(sw.max()),
    )
    out_p = ART / f"sklearn_rf_meta_natural{suffix}_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")

    if SMOKE:
        log("SMOKE — skipping diagnostic"); return

    # Diagnostic vs LB 0.98129
    log("=== vs LB 0.98129 (RF natural full-weight) ===")
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

    test_pred_r8 = (safelog(test_pred) + bias).argmax(1)
    test_pred_lb = (safelog(lb_test) + bias_lb).argmax(1)
    rows_diff = (test_pred_r8 != test_pred_lb).sum()
    net_h = int(((test_pred_r8 == 2) & (test_pred_lb != 2)).sum() -
                ((test_pred_lb == 2) & (test_pred_r8 != 2)).sum())
    churn_h = int(((test_pred_r8 == 2) ^ (test_pred_lb == 2)).sum())
    log(f"  test rows diff: {int(rows_diff)}/{n_te}")
    log(f"  test net_H flip: {net_h:+d}  churn_H: {churn_h}")

    out_csv = SUB / f"submission_sklearn_rf_meta_natural{SUFFIX}_standalone.csv"
    sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_pred_r8]})
    sub.to_csv(out_csv, index=False)
    log(f"wrote candidate {out_csv}  (NOT LB-probed — awaiting user approval)")
    log(f"  class counts: {sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
