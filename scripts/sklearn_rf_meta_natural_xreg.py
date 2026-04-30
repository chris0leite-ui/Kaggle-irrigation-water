"""Cross-regime bank extension experiment.

Tests whether the LB-best 0.98129 RF natural meta-stacker can compound
across calibration regimes by adding the prior PRIMARY (4-stack
tier1b_greedy_meta, LB 0.98094) as the 8th bank component.

Bank comparison:
  v1 (LB 0.98129, 7-component, all natural-cal regime):
    rawashishsin_2600, recipe_full_te_catboost_natural,
    recipe_full_te_catboost, recipe_full_te, realmlp, xgb_corn,
    xgb_dist_digits
  v_xreg (this experiment, 8-component):
    v1 + tier1b_greedy_meta  (LB 0.98094, recipe-bias regime [+1.4, +1.5, +3.4])

Mechanism: the v1 bank is all NEGATIVE-bias natural-cal regime. Adding the
4-stack (POSITIVE-bias recipe regime) tests whether RF can absorb cross-
regime diversity. UNTESTED in any prior bank-extension on RF natural.

Pipeline:
  1. Reconstruct tier1b_greedy_meta = 0.7 * LB-3-stack + 0.3 * meta_iso
     (matches LB-best 4-stack composition). Save as oof+test bank artifact.
  2. Train sklearn RF (class_weight=None, n_est=500, max_depth=12,
     bootstrap=True) on 8-component bank with 14 dist meta features.
  3. 5-fold StratifiedKFold(seed=42) for OOF alignment.
  4. Tune log-bias + run 4-gate vs LB-best v1 (LB 0.98129).

Outputs:
  scripts/artifacts/oof_tier1b_greedy_meta.npy + test (the recon)
  scripts/artifacts/oof_sklearn_rf_meta_natural_xreg.npy + test
  scripts/artifacts/sklearn_rf_meta_natural_xreg_results.json
  scripts/artifacts/blend_gate_rf_natural_xreg_results.json (custom, vs v1 LB-best)

Wall budget: ~10 min CPU.
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
from common import add_distance_features, log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import build_lbbest_stack, iso_cal, normed, ART, BIAS  # noqa: E402

SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
SEED = 42

# v1 LB-validated 7-component bank (LB 0.98129)
V1_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
]
# Cross-regime addition: tier1b_greedy_meta (LB-validated 0.98094, recipe-bias)
XREG_ADD = "tier1b_greedy_meta"
XREG_BANK = V1_BANK + [XREG_ADD]

META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]

DATA = Path("data")


def log(m):
    print(f"[xreg {time.strftime('%H:%M:%S')}] {m}", flush=True)


def reconstruct_4stack(y):
    """Rebuild tier1b_greedy_meta = 0.7 × LB-best 3-stack + 0.3 × meta_iso."""
    log("reconstructing tier1b_greedy_meta (4-stack PRIMARY)")
    s2_o, s2_t = build_lbbest_stack(y)  # 3-stack
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o, meta_t = iso_cal(meta_o, meta_t, y)
    fourstack_o = log_blend([s2_o, meta_o], np.array([0.7, 0.3]))
    fourstack_t = log_blend([s2_t, meta_t], np.array([0.7, 0.3]))
    # Sanity: should match documented LB-best at OOF 0.98084 @ recipe bias
    p = (np.log(np.clip(fourstack_o, 1e-12, 1)) + BIAS).argmax(1)
    bal = balanced_accuracy_score(y, p)
    log(f"  4-stack OOF @ recipe bias = {bal:.5f} (should be ~0.98084)")
    np.save(ART / f"oof_{XREG_ADD}.npy", fourstack_o.astype(np.float32))
    np.save(ART / f"test_{XREG_ADD}.npy", fourstack_t.astype(np.float32))
    log(f"  saved oof_{XREG_ADD}.npy + test_{XREG_ADD}.npy")
    return fourstack_o, fourstack_t


def main():
    t0 = time.time()
    log("loading train + test")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].to_numpy()
    log(f"  train={len(train):,} test={len(test):,}")

    # Reconstruct + save 4-stack
    reconstruct_4stack(y)

    # Load all 8 bank components
    log(f"loading {len(XREG_BANK)}-component cross-regime bank")
    pool = {}
    for name in XREG_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not (oof_p.exists() and test_p.exists()):
            log(f"  MISSING: {name}")
            continue
        o = normed(np.load(oof_p).astype(np.float32))
        t = normed(np.load(test_p).astype(np.float32))
        pool[name] = (o, t)
        log(f"  loaded {name}: oof={o.shape} test={t.shape}")
    assert len(pool) == 8, f"need all 8 components, got {len(pool)}"

    # Build meta features (per-row distance/rule features)
    log("building distance meta features")
    train_meta = add_distance_features(train.copy())
    test_meta = add_distance_features(test.copy())
    X_meta_tr = train_meta[META_COLS].to_numpy(dtype=np.float32)
    X_meta_te = test_meta[META_COLS].to_numpy(dtype=np.float32)

    # Stack: log-probs from each component + meta features
    component_names = sorted(pool.keys())
    log(f"  components in order: {component_names}")

    def safelog(p):
        return np.log(np.clip(p, 1e-12, 1)).astype(np.float32)

    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]

    X_tr = np.hstack(log_tr + [X_meta_tr])
    X_te = np.hstack(log_te + [X_meta_te])
    log(f"  X_tr={X_tr.shape} X_te={X_te.shape}")

    # Standardize
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    # 5-fold RF training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_probs = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    rf_params = dict(
        n_estimators=500, max_depth=12,
        max_features="sqrt", bootstrap=True,
        class_weight=None,  # natural-cal: no class upweight
        random_state=SEED, n_jobs=-1,
    )
    log(f"RF params: {rf_params}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_s, y), 1):
        t1 = time.time()
        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_tr_s[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X_tr_s[va_idx])
        test_probs += clf.predict_proba(X_te_s) / 5
        fold_argmax = balanced_accuracy_score(y[va_idx], clf.predict(X_tr_s[va_idx]))
        fold_scores.append(fold_argmax)
        log(f"  fold {fold}/5: argmax_bal_acc = {fold_argmax:.5f} wall={time.time()-t1:.0f}s")

    # Tune log-bias
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
    log(f"OOF: argmax={argmax_bal:.5f}  tuned={tuned:.5f}  bias={[round(b, 3) for b in bias.tolist()]}")

    # Save artifacts
    np.save(ART / "oof_sklearn_rf_meta_natural_xreg.npy", oof)
    np.save(ART / "test_sklearn_rf_meta_natural_xreg.npy", test_probs)
    out = {
        "bank": XREG_BANK,
        "n_components": len(XREG_BANK),
        "fold_scores": fold_scores,
        "oof_argmax": float(argmax_bal),
        "oof_tuned": float(tuned),
        "tuned_bias": [float(b) for b in bias.tolist()],
        "rf_params": {k: str(v) for k, v in rf_params.items()},
        "elapsed_seconds": time.time() - t0,
    }
    json_path = ART / "sklearn_rf_meta_natural_xreg_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"saved {json_path.name}")

    # Build standalone submission (RF natural xreg @ tuned bias)
    pred = (np.log(np.clip(test_probs, 1e-12, 1)) + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids, TARGET: [["Low", "Medium", "High"][p] for p in pred]})
    sub_path = SUB / "submission_sklearn_rf_meta_natural_xreg_standalone.csv"
    sub.to_csv(sub_path, index=False)
    log(f"saved {sub_path.name}")

    # 4-gate vs v1 LB-best (LB 0.98129)
    log("=" * 60)
    log("4-GATE vs v1 LB-best 0.98129")
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32)
    # v1's tuned bias (from prior session)
    V1_BIAS = np.array([0.43, 0.87, 3.20])

    def per_class_recall(p):
        return [float(((p == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in range(3)]

    # v1 standalone @ its bias
    v1_pred = (np.log(np.clip(v1_oof, 1e-12, 1)) + V1_BIAS).argmax(1)
    v1_bal = balanced_accuracy_score(y, v1_pred)
    v1_pcr = per_class_recall(v1_pred)
    log(f"  v1 anchor @ bias {V1_BIAS.tolist()}: bal={v1_bal:.5f} PCR=[L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}]")

    # xreg standalone @ its bias
    xreg_pred = (np.log(np.clip(oof, 1e-12, 1)) + bias).argmax(1)
    xreg_bal = balanced_accuracy_score(y, xreg_pred)
    xreg_pcr = per_class_recall(xreg_pred)
    log(f"  xreg standalone @ tuned bias: bal={xreg_bal:.5f} PCR=[L={xreg_pcr[0]:.4f} M={xreg_pcr[1]:.4f} H={xreg_pcr[2]:.4f}]")
    log(f"  Δ PCR (xreg - v1): L={xreg_pcr[0]-v1_pcr[0]:+.4f} M={xreg_pcr[1]-v1_pcr[1]:+.4f} H={xreg_pcr[2]-v1_pcr[2]:+.4f}")

    # Test-side disagreement
    v1_test_pred = (np.log(np.clip(v1_test, 1e-12, 1)) + V1_BIAS).argmax(1)
    xreg_test_pred = (np.log(np.clip(test_probs, 1e-12, 1)) + bias).argmax(1)
    n_disagree = (xreg_test_pred != v1_test_pred).sum()
    log(f"  Test disagreement: {n_disagree:,} / {len(test):,} ({100*n_disagree/len(test):.2f}%)")

    # Bias drift summary
    drift = bias - (-np.log(prior))
    log(f"  xreg bias drift from -log(prior): {[round(d, 3) for d in drift.tolist()]}")
    log(f"  (v1 documented drift = [-0.10, -0.10, -0.20]; recipe-bias drift = [+0.50, +0.40, -0.10])")

    # Verdict
    log("=" * 60)
    g1 = xreg_bal > v1_bal + 3e-4  # G1: standalone Δ ≥ +3e-4
    g2 = all(xreg_pcr[c] >= v1_pcr[c] - 5e-4 for c in range(3))  # G2: PCR within -5e-4
    log(f"G1 (Δ standalone ≥ +3e-4): {xreg_bal - v1_bal:+.5f}  {'PASS' if g1 else 'FAIL'}")
    log(f"G2 (PCR ≥ v1 - 5e-4):     [{xreg_pcr[0]-v1_pcr[0]:+.5f}, {xreg_pcr[1]-v1_pcr[1]:+.5f}, {xreg_pcr[2]-v1_pcr[2]:+.5f}]  {'PASS' if g2 else 'FAIL'}")
    if g1 and g2:
        log("OVERALL: 2/2 PASS — LB probe candidate at projected ~0.98129+OOF_Δ")
    else:
        log("OVERALL: FAIL — likely 33rd saturation, lock final selection")

    out_gate = {
        "v1_bal_acc": float(v1_bal),
        "xreg_bal_acc": float(xreg_bal),
        "delta_standalone": float(xreg_bal - v1_bal),
        "v1_pcr": v1_pcr,
        "xreg_pcr": xreg_pcr,
        "test_disagreement": int(n_disagree),
        "g1_pass": bool(g1),
        "g2_pass": bool(g2),
        "verdict": "PASS" if (g1 and g2) else "FAIL",
        "v1_bias": V1_BIAS.tolist(),
        "xreg_bias": [float(b) for b in bias.tolist()],
    }
    gate_path = ART / "blend_gate_rf_natural_xreg_results.json"
    gate_path.write_text(json.dumps(out_gate, indent=2))
    log(f"saved {gate_path.name}")
    log(f"total wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
