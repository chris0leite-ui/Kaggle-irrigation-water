"""Phase 3: sklearn RF meta-stacker on a natural-calibration bank.

Mirror of sklearn_rf_meta.py with the natural-cal pattern:
  - Drops `class_weight='balanced'` (the LB regression cause for LR-meta).
  - Reduces max_depth 14 -> 12 (less prone to fitting calibration noise).
  - Bank curated to NATURALLY-CALIBRATED components (rawashishsin v3 +
    Phase 1 cb_natural + LB-tight cb + realmlp + recipe_full_te +
    xgb_corn + xgb_dist_digits).

Mechanism: bagging + bootstrap + per-fold OOF + small reg gives the
"+0.00010 OOF→LB gap" calibration tightness. On a natural-cal bank,
the meta should stack monotonically rather than overfitting the
recipe-family bias-retune leak.

Per-fold StratifiedKFold(seed=42) aligned with all saved OOFs.

Outputs:
  scripts/artifacts/oof_sklearn_rf_meta_natural.npy
  scripts/artifacts/test_sklearn_rf_meta_natural.npy
  scripts/artifacts/sklearn_rf_meta_natural_results.json
  scripts/artifacts/blend_gate_rf_natural_results.json
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
from common import add_distance_features, fast_bal_acc, log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

SMOKE = os.environ.get("SMOKE") == "1"
META_SUFFIX = os.environ.get("META_SUFFIX", "")  # "" = original; "_a1lgbm" etc

# Natural-calibration bank: bias_H near 0 OR tight OOF→LB gap.
# Excludes recipe-family stacks with bias_H = +3.40 + leak channel.
# A1 expansion: + Pick 2b CB (sklearn TE), + XGB clone (rawashishsin parity),
# + xgb_dist_routed_v3 (routing lever).
NATURAL_BANK = [
    "rawashishsin_2600",                  # LB 0.98109 anchor (bias_H=0.00)
    "recipe_full_te_catboost_natural",    # Phase 1 output
    "recipe_full_te_catboost_skte",       # Pick 2b output (sklearn TE CB) — A1
    "recipe_full_te_xgb_skte",            # XGB clone on recipe FE — A1
    "recipe_full_te_lgbm_skte",           # LightGBM family-diversity — Option 1
    "recipe_full_te_catboost",            # LB 0.97935 gap +0.00001
    "recipe_full_te",                     # LB 0.97939 gap +0.00028 (recipe XGB)
    "realmlp",                            # 3-stack lift, NN diversity
    "xgb_corn",                           # Frank-Hall ordinal
    "xgb_dist_digits",                    # LB 0.97468 digit extraction
    "xgb_dist_routed_v3",                 # routing lever, naturally-cal-friendly — A1
]

# Distance / rule meta features (same as cuml_meta_input.npz)
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
    log(f"loading natural-cal bank ({len(NATURAL_BANK)} components)")
    pool = {}
    for name in NATURAL_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  SKIP {name}: missing")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != n_tr:
            log(f"  SKIP {name}: shape {o.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(NATURAL_BANK)} components")
    return pool


def build_features(pool, train, test, y):
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
    return X_tr, X_te, feature_names


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(y, n_tr, n_te)
    if "rawashishsin_2600" not in pool:
        log("ERROR: rawashishsin_2600 missing — abort")
        return
    if "recipe_full_te_catboost_natural" not in pool:
        log("WARN: Phase 1 cb_natural missing — running with reduced bank")

    X_tr, X_te, feature_names = build_features(pool, train, test, y)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12   # natural-cal: 14 -> 12
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight=None,             # natural-cal: drop balanced
        verbose=0,
    )
    log(f"RF (natural): n_est={n_est} max_depth={max_depth} class_weight=None bootstrap=True")

    if SMOKE:
        sub_idx = np.arange(50_000)
        X_tr_s_use = X_tr_s[sub_idx]
        y_use = y[sub_idx]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(X_tr_s_use, y_use))
        oof = np.zeros((len(y_use), 3), dtype=np.float32)
        test_pred = np.zeros((n_te, 3), dtype=np.float32)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(X_tr_s, y))
        oof = np.zeros((n_tr, 3), dtype=np.float32)
        test_pred = np.zeros((n_te, 3), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        if SMOKE:
            rf.fit(X_tr_s_use[tr_idx], y_use[tr_idx])
            p_va = rf.predict_proba(X_tr_s_use[va_idx]).astype(np.float32)
            p_te = rf.predict_proba(X_te_s).astype(np.float32)
            oof[va_idx] = p_va
            test_pred += p_te / n_folds
            bal = balanced_accuracy_score(y_use[va_idx], p_va.argmax(1))
        else:
            rf.fit(X_tr_s[tr_idx], y[tr_idx])
            p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
            p_te = rf.predict_proba(X_te_s).astype(np.float32)
            oof[va_idx] = p_va
            test_pred += p_te / n_folds
            bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t0:.1f}s")

    y_eval = y_use if SMOKE else y
    overall = balanced_accuracy_score(y_eval, oof.argmax(1))
    prior = np.bincount(y_eval, minlength=3) / len(y_eval)
    bias, tuned = tune_log_bias(oof, y_eval, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")
    bias_h = float(bias[2])

    if -0.5 <= bias_h <= 1.5:
        cal_verdict = "PASS — natural calibration in target band"
    elif bias_h <= 2.5:
        cal_verdict = "PARTIAL — bias_H below recipe family but above target"
    else:
        cal_verdict = "FAIL — bias_H still high"
    log(f"natural-cal verdict: {cal_verdict}")

    # Per-class recall
    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y_eval, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / f"oof_sklearn_rf_meta_natural{META_SUFFIX}.npy", oof)
    np.save(ART / f"test_sklearn_rf_meta_natural{META_SUFFIX}.npy", test_pred)

    summary = dict(
        n_folds=n_folds, smoke=SMOKE, seed=SEED,
        n_estimators=n_est, max_depth=max_depth,
        bank=NATURAL_BANK, bank_loaded=sorted(pool.keys()),
        feature_count=len(feature_names),
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        bias_H=bias_h,
        cal_verdict=cal_verdict,
        per_class_recall=pcr.tolist(),
    )
    with open(ART / f"sklearn_rf_meta_natural{META_SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/sklearn_rf_meta_natural{META_SUFFIX}_results.json")

    # Blend gate (only on full production)
    if SMOKE:
        log("SMOKE — skipping blend gate")
        return
    run_blend_gate(oof, test_pred, bias, y, test_ids)


def run_blend_gate(oof, test_pred, bias, y, test_ids):
    log("=== blend gate vs rawashishsin v3 + vs Phase 2 geomean blend ===")
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)
    raw_argmax = balanced_accuracy_score(y, raw_oof.argmax(1))

    geom_p = ART / "oof_blend_natural_geomean.npy"
    geom_t = ART / "test_blend_natural_geomean.npy"
    have_geom = geom_p.exists() and geom_t.exists()
    if have_geom:
        geom_oof = np.load(geom_p).astype(np.float32)
        geom_test = np.load(geom_t).astype(np.float32)
        geom_bal = balanced_accuracy_score(y, geom_oof.argmax(1))
        log(f"  geomean anchor OOF argmax = {geom_bal:.5f}")
    else:
        geom_oof = None
        geom_test = None
        geom_bal = None

    log(f"  rawashishsin anchor argmax = {raw_argmax:.5f}")

    # Blend RF natural (no retune) into rawashishsin
    results = {}
    for anchor_name, anchor_oof, anchor_test, anchor_bal in [
        ("rawashishsin", raw_oof, raw_test, raw_argmax),
        ("geomean", geom_oof, geom_test, geom_bal),
    ]:
        if anchor_oof is None:
            log(f"  skip {anchor_name} — missing")
            continue
        log(f"  -- vs {anchor_name} anchor --")
        a_pred = anchor_oof.argmax(1)
        a_pcr = per_class_recall(y, a_pred)
        sweep = []
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            blend_oof = log_blend([anchor_oof, oof],
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
        # Pick best alpha that passes per-class guardrail
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
            # Net rare-class flips for G4
            blend_test = log_blend([anchor_test, test_pred],
                                    np.array([1.0 - best["alpha"], best["alpha"]]))
            a_test_pred = anchor_test.argmax(1)
            b_test_pred = blend_test.argmax(1)
            net_h = int(((b_test_pred == 2) & (a_test_pred != 2)).sum() -
                        ((a_test_pred == 2) & (b_test_pred != 2)).sum())
            churn_h = int(((b_test_pred == 2) ^ (a_test_pred == 2)).sum())
            log(f"    test diff vs {anchor_name}: net_H={net_h:+d}  churn_H={churn_h}")
            sub_path = SUB / f"submission_rf_natural{META_SUFFIX}_blend_{anchor_name}_a{int(best['alpha']*100):03d}.csv"
            sub = pd.DataFrame({
                "id": test_ids,
                TARGET: [IDX2CLS[i] for i in b_test_pred],
            })
            sub.to_csv(sub_path, index=False)
            log(f"    wrote {sub_path}  (no LB submit — gate result documented)")
            results[anchor_name] = dict(
                anchor_bal=float(anchor_bal), best=best,
                net_H=net_h, churn_H=churn_h, sub_path=str(sub_path),
            )
        else:
            log(f"    no gate-pass alpha")
            results[anchor_name] = dict(anchor_bal=float(anchor_bal),
                                         best=None, sweep=sweep)

    out_p = ART / "blend_gate_rf_natural_results.json"
    out_p.write_text(json.dumps(results, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
