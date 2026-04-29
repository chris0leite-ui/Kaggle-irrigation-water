"""R12: 9-component RF natural bank with R10's 8 + T2 pseudo as 9th.

R10 (8c) added Tier 1b 4-stack → +0.00040 OOF, LB -0.00010 (near-tie).
R11 (9c) added 3way_recipe025 (deterministic blend) → did NOT compound.

R12 tests: adding T2 PSEUDO-AUGMENTED component (genuinely NEW signal
from 198k pseudo rows trained with LB 0.98129 labeler at τ=0.99) as 9th.

Structurally different from R11's 3way addition:
  R11: 3way = log_blend(recipe + s1 + s7) — deterministic, no new info
  R12: T2 = recipe XGB trained on (real + 198k pseudo rows) — NEW signal

T2 standalone: tuned 0.97990, drift [+0.7, +0.2, -0.4]
  - Higher than recipe alone (+0.00023)
  - drift_H = -0.4 (similar direction to rawashishsin's -0.4)

Hypothesis: T2's pseudo-augmented signal compounds R10's bank-extension
lift because it's training-data-level diversity, not blend.

Outputs (suffix _r12_with_t2):
  oof_sklearn_rf_meta_natural_r12_with_t2.npy
  test_sklearn_rf_meta_natural_r12_with_t2.npy
  sklearn_rf_meta_natural_r12_with_t2_results.json
  submission_sklearn_rf_meta_natural_r12_with_t2_standalone.csv
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
from tier1b_helpers import build_lbbest_stack, iso_cal, normed  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = "_r12_with_t2"

BANK_BASE = [
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


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = {}
    for name in BANK_BASE:
        oof = normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        tst = normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        pool[name] = (oof, tst)
        log(f"  + {name}")

    # 8th: tier1b 4-stack
    log("building Tier 1b 4-stack as 8th component")
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    w = np.array([0.7, 0.3])
    s4_o = np.exp(w[0] * safelog(lb3_o) + w[1] * safelog(meta_o_iso))
    s4_o = normed(s4_o / s4_o.sum(1, keepdims=True))
    s4_t = np.exp(w[0] * safelog(lb3_t) + w[1] * safelog(meta_t_iso))
    s4_t = normed(s4_t / s4_t.sum(1, keepdims=True))
    pool["tier1b_4stack"] = (s4_o, s4_t)
    log(f"  + tier1b_4stack (LB 0.98094)")

    # 9th: T2 pseudo-label component (LB 0.98129 labeler @ τ=0.99)
    log("loading T2 pseudo-label component")
    t2_oof = normed(np.load(ART / "oof_recipe_pseudolabel_lb98129labeler_t099.npy").astype(np.float32))
    t2_test = normed(np.load(ART / "test_recipe_pseudolabel_lb98129labeler_t099.npy").astype(np.float32))
    pool["t2_pseudo_lb98129"] = (t2_oof, t2_test)
    log(f"  + t2_pseudo_lb98129 (recipe XGB trained on real + 198k pseudo @ τ=0.99)")

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
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    if SMOKE:
        sub_idx = np.arange(50_000)
        X_tr_s = X_tr_s[sub_idx]
        y_use = y[sub_idx]
    else:
        y_use = y

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y_use))
    oof = np.zeros((len(y_use), 3), dtype=np.float32)
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
        rf.fit(X_tr_s[tr_idx], y_use[tr_idx])
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
    log(f"=== R12 OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")
    log(f"  drift = {drift.round(4).tolist()}  max|drift|={float(np.abs(drift).max()):.4f}")
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]  errs={int(errs)}")

    suffix = "_smoke" if SMOKE else SUFFIX
    np.save(ART / f"oof_sklearn_rf_meta_natural{suffix}.npy", oof)
    np.save(ART / f"test_sklearn_rf_meta_natural{suffix}.npy", test_pred)
    summary = dict(
        smoke=SMOKE, bank_size=len(names), bank=names,
        fold_scores=fold_scores,
        overall_argmax=float(overall), tuned_log_bias=float(tuned),
        log_bias=bias.tolist(), drift=drift.tolist(),
        per_class_recall=pcr.tolist(), errs=int(errs),
    )
    out_p = ART / f"sklearn_rf_meta_natural{suffix}_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")

    if SMOKE:
        log("SMOKE — skipping diagnostic"); return

    log("=== vs LB 0.98129 ===")
    lb_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    lb_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    bias_lb, tuned_lb = tune_log_bias(lb_oof, y, prior)
    pred_lb = (safelog(lb_oof) + bias_lb).argmax(1)
    pcr_lb = per_class_recall(y, pred_lb)
    errs_lb = (pred_lb != y).sum()
    log(f"  LB 0.98129: tuned={tuned_lb:.5f}  errs={int(errs_lb)}")
    log(f"  Δ tuned = {tuned - tuned_lb:+.5f}")
    log(f"  Δ PCR  = [L={pcr[0] - pcr_lb[0]:+.5f} M={pcr[1] - pcr_lb[1]:+.5f} H={pcr[2] - pcr_lb[2]:+.5f}]")
    log(f"  Δ errs = {int(errs) - int(errs_lb):+d}")

    log("=== vs R10 (8c with 4-stack) ===")
    r10_oof = np.load(ART / "oof_sklearn_rf_meta_natural_r10_with_tier1b.npy").astype(np.float32)
    bias_r10, tuned_r10 = tune_log_bias(r10_oof, y, prior)
    pred_r10 = (safelog(r10_oof) + bias_r10).argmax(1)
    pcr_r10 = per_class_recall(y, pred_r10)
    errs_r10 = (pred_r10 != y).sum()
    log(f"  R10: tuned={tuned_r10:.5f}  errs={int(errs_r10)}")
    log(f"  Δ vs R10 tuned = {tuned - tuned_r10:+.5f}")
    log(f"  Δ vs R10 PCR  = [L={pcr[0] - pcr_r10[0]:+.5f} M={pcr[1] - pcr_r10[1]:+.5f} H={pcr[2] - pcr_r10[2]:+.5f}]")

    test_pred_r12 = (safelog(test_pred) + bias).argmax(1)
    test_pred_lb = (safelog(lb_test) + bias_lb).argmax(1)
    rows_diff = (test_pred_r12 != test_pred_lb).sum()
    net_h = int(((test_pred_r12 == 2) & (test_pred_lb != 2)).sum() -
                ((test_pred_lb == 2) & (test_pred_r12 != 2)).sum())
    churn_h = int(((test_pred_r12 == 2) ^ (test_pred_lb == 2)).sum())
    log(f"  test rows diff vs LB-best: {int(rows_diff)}/{n_te}")
    log(f"  test net_H flip: {net_h:+d}  churn_H: {churn_h}")

    out_csv = SUB / f"submission_sklearn_rf_meta_natural{SUFFIX}_standalone.csv"
    sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_pred_r12]})
    sub.to_csv(out_csv, index=False)
    log(f"wrote candidate {out_csv}")
    log(f"  class counts: {sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
