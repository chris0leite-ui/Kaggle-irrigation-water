"""Variant C: v1 RF natural meta with router OOF added as 1-d bank feature.

Mechanism: idea 4's router has AUC 0.895 distinguishing v1-wins vs raw-wins
on disagreement rows. Hard routing failed on the macro-recall Pareto frontier;
soft blending failed on the same axis. But the router's signal is structurally
orthogonal to the existing 7 prob-vector components — encoding it as a 1-d
feature lets the meta-stacker learn when to trust it WITHOUT forcing a hard
decision.

Architecture: same as v1 (7-component bank, sklearn RF natural-cal config),
plus 1 extra column = router OOF P(raw_wins).

Output: oof_rf_natural_v1_router.npy + test + submission CSV at tuned bias.
"""
from __future__ import annotations

import json
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

NATURAL_BANK_V1 = [
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
        m = y == k
        if m.sum():
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    log("=== RF natural v1 + router-as-feature ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = {}
    for name in NATURAL_BANK_V1:
        oof = _normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        tst = _normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        pool[name] = (oof, tst)
    log(f"loaded {len(pool)}/7 components")

    # Router OOF + test (1-d each)
    router_oof = np.load(ART / "oof_router_predictions.npy").astype(np.float32).reshape(-1, 1)
    router_test = np.load(ART / "test_router_decisions.npy").astype(np.float32).reshape(-1, 1)
    log(f"router OOF range: [{router_oof.min():.3f}, {router_oof.max():.3f}] "
        f"nonzero={(router_oof != 0).sum()}/{n_tr}")
    log(f"router test range: [{router_test.min():.3f}, {router_test.max():.3f}]")

    # Build dist features
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    for d in (tr_d, te_d):
        d["min_boundary_dist"] = d[["sm_abs", "rf_abs", "tc_abs", "ws_abs"]].min(axis=1)
        d["min_axis_abs"] = d[["sm_abs", "rf_abs", "tc_abs", "ws_abs"]].min(axis=1)
        d["score_dist_low_mid"] = (d["dgp_score"].astype(np.float32) - 3.5).abs()
        d["score_dist_mid_high"] = (d["dgp_score"].astype(np.float32) - 6.5).abs()
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    # Concat: meta_dist + sorted log-prob blocks + router (1-d)
    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]
    X_tr = np.concatenate([meta_tr] + log_tr + [router_oof], axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te + [router_test], axis=1).astype(np.float32)
    log(f"X_tr={X_tr.shape}  X_te={X_te.shape}  (last col = router)")

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    rf_params = dict(
        n_estimators=500, max_depth=12,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=42,
        class_weight=None, verbose=0,
    )
    log(f"RF: n_est=500 max_depth=12 class_weight=None features={X_tr.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_tr_s, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/5  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / 5
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"=== overall argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR = [L={pcr[0]:.4f}  M={pcr[1]:.4f}  H={pcr[2]:.4f}]")

    natural_bias = -np.log(prior)
    drift = bias - natural_bias
    log(f"  drift = {drift.round(4).tolist()}  |max|={float(np.abs(drift).max()):.3f}")

    np.save(ART / "oof_rf_natural_v1_router.npy", oof)
    np.save(ART / "test_rf_natural_v1_router.npy", test_pred)
    log("saved oof/test rf_natural_v1_router")

    test_logits = safelog(test_pred) + bias
    test_argmax = test_logits.argmax(1)
    sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_argmax]})
    sub_path = SUB / "submission_rf_natural_v1_router_standalone.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    # Compare vs LB-best v1 (LB 0.98129)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_logits = safelog(v1_test) + np.array([0.43, 0.87, 3.20])
    v1_argmax = v1_logits.argmax(1)
    diff = int((test_argmax != v1_argmax).sum())
    log(f"test diff vs v1: {diff}/{n_te} ({100*diff/n_te:.3f}%)")
    for k in range(3):
        d = int((test_argmax == k).sum() - (v1_argmax == k).sum())
        log(f"  class {IDX2CLS[k]}: count delta = {d:+d}")

    # PCR delta vs v1 LB-best (use idx2cls)
    pcr_v1 = per_class_recall(y, np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy")
                              .astype(np.float32).argmax(1))
    # Use v1's bias for fair compare
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    bias_v1, _ = tune_log_bias(v1_oof, y, prior)
    v1_oof_pred = (safelog(v1_oof) + bias_v1).argmax(1)
    pcr_v1 = per_class_recall(y, v1_oof_pred)
    log(f"v1 PCR @ tuned: [L={pcr_v1[0]:.4f} M={pcr_v1[1]:.4f} H={pcr_v1[2]:.4f}]")
    log(f"variant_C PCR delta vs v1: "
        f"L{pcr[0]-pcr_v1[0]:+.5f} M{pcr[1]-pcr_v1[1]:+.5f} H{pcr[2]-pcr_v1[2]:+.5f}")

    summary = dict(
        bank=NATURAL_BANK_V1, bank_size=len(pool),
        n_features=X_tr.shape[1],
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall), tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        drift=drift.tolist(),
        max_drift_magnitude=float(np.abs(drift).max()),
        per_class_recall=pcr.tolist(),
        v1_per_class_recall=pcr_v1.tolist(),
        pcr_delta_vs_v1=[float(pcr[k] - pcr_v1[k]) for k in range(3)],
        test_diff_vs_v1=diff,
    )
    with open(ART / "rf_natural_v1_router_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote rf_natural_v1_router_results.json")


if __name__ == "__main__":
    main()
