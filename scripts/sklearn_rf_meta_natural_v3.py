"""v3 RF natural meta-stacker — extends v2's 10-component bank with LGBM-skte.

Mirror of `sklearn_rf_meta_natural.py` (main, LB 0.98129 v1; in-flight v2)
with one difference: adds `recipe_full_te_lgbm_skte` as the 11th component.

Bank composition (target 11 components when all artefacts on disk):
  1.  rawashishsin_2600                  (LB 0.98109 anchor, bias_H=0.00)
  2.  recipe_full_te_catboost_natural    (Phase 1, drift 0.90 PARTIAL)
  3.  recipe_full_te_catboost_skte       (Pick 2b A1, drift 1.50 FAIL)
  4.  recipe_full_te_xgb_skte            (XGB clone A1, drift TBD)
  5.  recipe_full_te_catboost            (LB 0.97935 gap +0.00001)
  6.  recipe_full_te                     (recipe XGB, LB 0.97939)
  7.  realmlp                            (NN diversity)
  8.  xgb_corn                           (Frank-Hall ordinal)
  9.  xgb_dist_digits                    (LB 0.97468 digit extraction)
  10. xgb_dist_routed_v3                 (routing lever)
  11. recipe_full_te_lgbm_skte           ← THIS BRANCH ADDS (drift 1.00 PARTIAL)

LGBM-skte's contribution:
  - Different model class (LGBM leaf-wise growth) vs XGB level-wise,
    CB ordered-boosting, RealMLP, etc.
  - Bias drift profile structurally distinct: NEGATIVE drift on High
    (-0.30) where recipe + rawashishsin have ~0. Could provide
    ADD-High asymmetry the bank otherwise lacks.
  - Standalone tuned 0.97862 (competitive but below recipe).

Same RF HPs as v2: max_depth=12, min_samples_leaf=20, max_features='sqrt',
bootstrap=True, class_weight=None, n_estimators=500.
Same 5-fold StratifiedKFold(seed=42).

Auto-tolerant of missing components: gracefully skips any bank entry
whose oof/test files are missing (so this script can run BEFORE main's
v2 chain completes — produces a v3-with-partial-bank result, useful as
an early-peek diagnostic).

Outputs (separate namespace from v1/v2 to avoid clobbering):
  scripts/artifacts/oof_sklearn_rf_meta_natural_v3.npy
  scripts/artifacts/test_sklearn_rf_meta_natural_v3.npy
  scripts/artifacts/sklearn_rf_meta_natural_v3_results.json
  scripts/artifacts/blend_gate_rf_natural_v3_results.json
  submissions/submission_rf_natural_v3_blend_*.csv (only on gate pass)

Run AFTER main's v2 lands for full 11-component bank. Run BEFORE for
early diagnostic on whatever subset is loadable.
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

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = os.environ.get("V3_SUFFIX", "_v3")  # change to compare variants

NATURAL_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost_skte",
    "recipe_full_te_xgb_skte",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
    "xgb_dist_routed_v3",
    "recipe_full_te_lgbm_skte",  # ← v3 adds (this branch's contribution)
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


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def load_bank(y, n_tr, n_te):
    log(f"loading natural-cal bank (target {len(NATURAL_BANK)} components)")
    pool = {}
    missing = []
    for name in NATURAL_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  SKIP {name}: missing")
            missing.append(name)
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != n_tr:
            log(f"  SKIP {name}: shape {o.shape}")
            missing.append(name)
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            missing.append(name)
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(NATURAL_BANK)}  missing={missing}")
    return pool, missing


def build_features(pool, train, test):
    log("constructing distance / rule meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    feature_names = list(META_COLS)
    for n in component_names:
        feature_names += [f"{n}_logL", f"{n}_logM", f"{n}_logH"]

    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  feature matrix: train={X_tr.shape}  test={X_te.shape}")
    return X_tr, X_te, feature_names, component_names


def main():
    log(f"v3 RF natural meta-stacker — SUFFIX={SUFFIX}")
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool, missing = load_bank(y, n_tr, n_te)
    if "rawashishsin_2600" not in pool:
        log("ERROR: rawashishsin_2600 missing — abort (anchor required)")
        return
    if "recipe_full_te_lgbm_skte" not in pool:
        log("WARN: recipe_full_te_lgbm_skte missing — running v2-equivalent bank "
            "(this branch's contribution absent)")
    if len(missing) > 0:
        log(f"WARN: {len(missing)} components missing; running with reduced "
            f"bank size {len(pool)}")

    X_tr, X_te, feature_names, component_names = build_features(pool, train, test)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight=None,
        verbose=0,
    )
    log(f"RF: n_est={n_est} max_depth={max_depth} class_weight=None bootstrap=True")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    drift = bias - (-np.log(prior))
    drift_max = float(np.abs(drift).max())
    log(f"  bias drift from -log(prior): {drift.round(4).tolist()}  max {drift_max:.3f}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / f"oof_sklearn_rf_meta_natural{SUFFIX}.npy", oof)
    np.save(ART / f"test_sklearn_rf_meta_natural{SUFFIX}.npy", test_pred)

    summary = dict(
        n_folds=n_folds, smoke=SMOKE, seed=SEED, suffix=SUFFIX,
        n_estimators=n_est, max_depth=max_depth,
        bank_target=NATURAL_BANK, bank_loaded=sorted(pool.keys()),
        bank_missing=missing,
        feature_count=len(feature_names),
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        bias_drift=drift.tolist(),
        drift_max=drift_max,
        per_class_recall=pcr.tolist(),
    )
    with open(ART / f"sklearn_rf_meta_natural{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote sklearn_rf_meta_natural{SUFFIX}_results.json")

    if SMOKE:
        log("SMOKE — skipping blend gate")
        return
    run_blend_gate(oof, test_pred, bias, y, test_ids)


def run_blend_gate(oof, test_pred, bias, y, test_ids):
    log("=== 4-gate analysis vs rawashishsin v3 + vs v1 LB-best 0.98129 ===")
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))
    raw_argmax = balanced_accuracy_score(y, raw_oof.argmax(1))
    log(f"  rawashishsin anchor argmax = {raw_argmax:.5f}")

    # Also load v1 LB-validated RF natural meta (LB 0.98129) as comparison.
    v1_p = ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy"
    if v1_p.exists():
        v1_oof = _normed(np.load(v1_p).astype(np.float32))
        v1_bal = balanced_accuracy_score(y, v1_oof.argmax(1))
        log(f"  v1 LB-best (LB 0.98129) anchor argmax = {v1_bal:.5f}")
    else:
        v1_oof, v1_bal = None, None
        log(f"  v1 LB-best preserved artifact missing — skipping comparison")

    results = {}
    anchors = [("rawashishsin", raw_oof, raw_test, raw_argmax)]

    for anchor_name, anchor_oof, anchor_test, anchor_bal in anchors:
        log(f"  -- vs {anchor_name} anchor --")
        a_pred = anchor_oof.argmax(1)
        a_pcr = per_class_recall(y, a_pred)
        sweep = []
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            blend_oof = log_blend([anchor_oof, _normed(oof)],
                                   np.array([1.0 - alpha, alpha]))
            bal = balanced_accuracy_score(y, blend_oof.argmax(1))
            pcr = per_class_recall(y, blend_oof.argmax(1))
            d_class = (pcr - a_pcr).tolist()
            sweep.append({
                "alpha": alpha,
                "bal_acc": float(bal),
                "delta": float(bal - anchor_bal),
                "pcr_delta": d_class,
            })
        # Pick best alpha that passes per-class guardrail (G2: PCR ≥ -5e-4)
        best = None
        for s in sweep:
            if all(d >= -5e-4 for d in s["pcr_delta"]):
                if best is None or s["bal_acc"] > best["bal_acc"]:
                    best = s
        log(f"    sweep: " + " ".join(
            f"a={s['alpha']:.2f}:{s['delta']:+.5f}{'(g)' if all(d>=-5e-4 for d in s['pcr_delta']) else ''}"
            for s in sweep))
        if best is not None:
            log(f"    best gate-pass alpha={best['alpha']:.2f}  delta={best['delta']:+.5f}")
            blend_test = log_blend([anchor_test, _normed(test_pred)],
                                    np.array([1.0 - best["alpha"], best["alpha"]]))
            a_test_pred = anchor_test.argmax(1)
            b_test_pred = blend_test.argmax(1)
            net_h = int(((b_test_pred == 2) & (a_test_pred != 2)).sum() -
                        ((a_test_pred == 2) & (b_test_pred != 2)).sum())
            churn_h = int(((b_test_pred == 2) ^ (a_test_pred == 2)).sum())
            g4_ratio = abs(net_h) / max(churn_h, 1)
            g4_pass = (net_h > 0) and (g4_ratio >= 0.5)
            log(f"    test diff vs {anchor_name}: net_H={net_h:+d}  churn_H={churn_h}  ratio={g4_ratio:.2f}  G4_pass={g4_pass}")
            sub_path = SUB / f"submission_rf_natural{SUFFIX}_blend_{anchor_name}_a{int(best['alpha']*100):03d}.csv"
            sub = pd.DataFrame({
                "id": test_ids,
                TARGET: [IDX2CLS[i] for i in b_test_pred],
            })
            sub.to_csv(sub_path, index=False)
            log(f"    wrote {sub_path}  (no LB submit — gate result documented)")
            results[anchor_name] = dict(
                anchor_bal=float(anchor_bal), best=best,
                net_H=net_h, churn_H=churn_h, g4_ratio=g4_ratio,
                g4_pass=g4_pass, sub_path=str(sub_path),
            )
        else:
            log(f"    no gate-pass alpha")
            results[anchor_name] = dict(anchor_bal=float(anchor_bal),
                                         best=None, sweep=sweep)

    out_p = ART / f"blend_gate_rf_natural{SUFFIX}_results.json"
    out_p.write_text(json.dumps(results, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
