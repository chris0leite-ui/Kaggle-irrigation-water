"""Path 2 — RF natural META seed-bag.

Mechanism: variance reduction at the META level on the EXACT v1 7-component
bank. Different from N1 (which bagged the rawashishsin INPUT and lost
quality variance) — here we bag the meta itself.

5 sklearn RandomForestClassifier seeds × 5 folds × 7-component bank.
Geomean OOFs and test probs across seeds. Per-seed checkpoint files for
rehydrate resilience.

Hypothesis: RF stochastic variance is non-trivial because bootstrap=True
and max_features='sqrt'. Geomean across seeds should produce slightly
smoother decision surface that may transfer cleaner to LB.

Usage:
  SMOKE=1 python scripts/path2_rf_meta_seedbag.py    # 1 seed, 2 folds, ~30s
  python scripts/path2_rf_meta_seedbag.py            # 5 seeds, 5 folds, ~110 min

Outputs:
  scripts/artifacts/oof_rf_meta_seedbag.npy
  scripts/artifacts/test_rf_meta_seedbag.npy
  scripts/artifacts/rf_meta_seedbag_results.json
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
FOLD_SEED = 42

SMOKE = os.environ.get("SMOKE") == "1"

# v1 LB-best 7-component bank (LB 0.98129).  EXACTLY as-committed in fe99b4e.
V1_BANK = [
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

RF_SEEDS = [42] if SMOKE else [42, 7, 123]  # 3-seed for time budget; geomean still meaningful


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


def load_bank(n_tr, n_te):
    log(f"loading v1 bank ({len(V1_BANK)} components)")
    pool = {}
    for name in V1_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  MISSING: {name}")
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
    log(f"  loaded {len(pool)}/{len(V1_BANK)} components")
    return pool


def build_features(pool, train, test):
    log("constructing distance / rule meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]

    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  feature matrix: train={X_tr.shape}  test={X_te.shape}")
    return X_tr, X_te, component_names


def train_one_seed(X_tr_s, X_te_s, y, n_te, rf_seed, n_folds=5, n_est=500, max_depth=12):
    """Train one RF natural meta at given seed.  Returns (oof, test_pred)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=FOLD_SEED)
    splits = list(skf.split(X_tr_s, y))
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_leaf=20, max_features="sqrt",
            bootstrap=True, n_jobs=-1, random_state=rf_seed,
            class_weight=None, verbose=0,
        )
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"    seed={rf_seed} fold {fold}/{n_folds} argmax={bal:.5f} wall={time.time()-t0:.1f}s")
    return oof, test_pred, fold_scores


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(n_tr, n_te)
    if len(pool) != 7:
        log(f"FATAL: expected 7 components, got {len(pool)}")
        return

    X_tr, X_te, component_names = build_features(pool, train, test)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    if SMOKE:
        sub_idx = np.arange(50_000)
        X_tr_s_use = X_tr_s[sub_idx]
        y_use = y[sub_idx]
        n_folds, n_est, max_depth = 2, 100, 8
        log(f"SMOKE: 1 seed × 2 folds × n_est={n_est}")
    else:
        X_tr_s_use = X_tr_s
        y_use = y
        n_folds, n_est, max_depth = 5, 500, 12
        log(f"PROD: {len(RF_SEEDS)} seeds × {n_folds} folds × n_est={n_est}")

    # Per-seed checkpoint loop
    all_oofs = []
    all_tests = []
    seed_summary = {}
    for rf_seed in RF_SEEDS:
        oof_p = ART / f"oof_rf_meta_seedbag_seed{rf_seed}.npy"
        test_p = ART / f"test_rf_meta_seedbag_seed{rf_seed}.npy"
        if oof_p.exists() and test_p.exists() and not SMOKE:
            log(f"=== seed={rf_seed}: cached, loading ===")
            oof = np.load(oof_p).astype(np.float32)
            tp = np.load(test_p).astype(np.float32)
            argmax = balanced_accuracy_score(y_use, oof.argmax(1))
            log(f"  cached argmax={argmax:.5f}")
        else:
            log(f"=== seed={rf_seed}: training ===")
            t_seed = time.time()
            oof, tp, fold_scores = train_one_seed(
                X_tr_s_use, X_te_s, y_use, n_te,
                rf_seed=rf_seed, n_folds=n_folds, n_est=n_est, max_depth=max_depth,
            )
            argmax = balanced_accuracy_score(y_use, oof.argmax(1))
            log(f"  seed={rf_seed} OOF argmax={argmax:.5f} wall={time.time()-t_seed:.1f}s")
            np.save(oof_p, oof)
            np.save(test_p, tp)
            log(f"  saved {oof_p.name} + {test_p.name}")
            seed_summary[str(rf_seed)] = dict(
                argmax=float(argmax), fold_scores=fold_scores)
        all_oofs.append(oof)
        all_tests.append(tp)

    # Geomean across seeds
    log("=== geomean across seeds ===")
    log_oofs = np.stack([safelog(_normed(o)) for o in all_oofs], axis=0)
    log_tests = np.stack([safelog(_normed(t)) for t in all_tests], axis=0)
    bag_oof = _normed(np.exp(log_oofs.mean(axis=0)))
    bag_test = _normed(np.exp(log_tests.mean(axis=0)))

    overall = balanced_accuracy_score(y_use, bag_oof.argmax(1))
    prior = np.bincount(y_use, minlength=3) / len(y_use)
    bias, tuned = tune_log_bias(bag_oof, y_use, prior)
    log(f"BAG OOF argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    drift = (bias - (-np.log(prior))).round(4).tolist()
    log(f"BAG drift from -log(prior) = {drift}  (max |drift| = {max(abs(d) for d in drift):.3f})")

    pred_at_bias = (safelog(bag_oof) + bias).argmax(1)
    pcr_bag = per_class_recall(y_use, pred_at_bias)
    log(f"BAG PCR=[L={pcr_bag[0]:.4f} M={pcr_bag[1]:.4f} H={pcr_bag[2]:.4f}]")

    # Compare vs v1 LB-best
    v1_oof_path = ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy"
    if v1_oof_path.exists() and not SMOKE:
        v1_oof = np.load(v1_oof_path).astype(np.float32)
        v1_test_path = ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy"
        v1_test = np.load(v1_test_path).astype(np.float32) if v1_test_path.exists() else None
        v1_overall = balanced_accuracy_score(y, v1_oof.argmax(1))
        v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
        log(f"V1 LB-best (saved): argmax={v1_overall:.5f}  tuned={v1_tuned:.5f}  bias={v1_bias.round(4).tolist()}")
        v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
        v1_pcr = per_class_recall(y, v1_pred)
        log(f"V1 PCR=[L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}]")
        delta_tuned = float(tuned - v1_tuned)
        delta_pcr = (pcr_bag - v1_pcr).round(5).tolist()
        log(f"BAG vs V1: Δ tuned = {delta_tuned:+.5f}  Δ PCR = {delta_pcr}")
    else:
        v1_oof = None
        v1_test = None
        v1_tuned = None
        delta_tuned = None
        delta_pcr = None

    # Save bag artifacts
    np.save(ART / "oof_rf_meta_seedbag.npy", bag_oof)
    np.save(ART / "test_rf_meta_seedbag.npy", bag_test)

    # 4-gate test vs v1 PRIMARY (only if v1 available)
    gate_results = None
    if v1_oof is not None and v1_test is not None and not SMOKE:
        gate_results = run_4gate(bag_oof, bag_test, v1_oof, v1_test, y, test_ids, prior)

    summary = dict(
        method="path2_rf_meta_seedbag",
        rf_seeds=RF_SEEDS,
        n_folds=n_folds, n_est=n_est, max_depth=max_depth,
        bank=V1_BANK,
        bag_overall_argmax=float(overall),
        bag_tuned=float(tuned),
        bag_bias=bias.tolist(),
        bag_drift=drift,
        bag_pcr=pcr_bag.tolist(),
        v1_tuned=float(v1_tuned) if v1_tuned else None,
        delta_tuned=delta_tuned,
        delta_pcr=delta_pcr,
        seed_summary=seed_summary,
        gate=gate_results,
    )
    out_p = ART / "rf_meta_seedbag_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")


def run_4gate(bag_oof, bag_test, v1_oof, v1_test, y, test_ids, prior):
    """Test bag as candidate vs v1 PRIMARY anchor."""
    log("=== 4-gate filter: bag-meta as candidate, anchor = v1 LB-best ===")
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred)
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)

    sweep = []
    for alpha in [0.10, 0.20, 0.30, 0.40, 0.50]:
        # Log-blend bag into v1 with v1's bias fixed
        blend_oof = log_blend([v1_oof, bag_oof], np.array([1.0 - alpha, alpha]))
        oof_pred_at_v1bias = (safelog(blend_oof) + v1_bias).argmax(1)
        bal = balanced_accuracy_score(y, oof_pred_at_v1bias)
        pcr = per_class_recall(y, oof_pred_at_v1bias)
        delta = float(bal - v1_tuned)
        d_class = (pcr - v1_pcr).tolist()

        # Test-side asymmetry for G4
        blend_test = log_blend([v1_test, bag_test], np.array([1.0 - alpha, alpha]))
        b_test_pred = (safelog(blend_test) + v1_bias).argmax(1)
        add_h = int(((b_test_pred == 2) & (v1_test_pred != 2)).sum())
        rem_h = int(((b_test_pred != 2) & (v1_test_pred == 2)).sum())
        net_h = add_h - rem_h
        churn = add_h + rem_h
        ratio = abs(net_h) / max(1, churn)

        g1 = delta >= 2e-4
        g2 = all(d >= -5e-4 for d in d_class)
        g4 = (net_h > 0) and (ratio >= 0.5)

        sweep.append(dict(
            alpha=alpha, delta=delta, pcr_delta=d_class,
            add_h=add_h, rem_h=rem_h, net_h=net_h, churn=churn, ratio=ratio,
            G1=bool(g1), G2=bool(g2), G4=bool(g4),
            all_pass=bool(g1 and g2 and g4),
        ))
        log(f"  α={alpha:.2f} Δ={delta:+.5f} PCR=[L{d_class[0]:+.5f} M{d_class[1]:+.5f} H{d_class[2]:+.5f}] "
            f"net_H={net_h:+d}/churn={churn}  G1={g1} G2={g2} G4={g4}")

    passing = [s for s in sweep if s["all_pass"]]
    log(f"=== {len(passing)}/{len(sweep)} alphas pass all 3 gates ===")
    if passing:
        best = max(passing, key=lambda s: s["delta"])
        log(f"BEST gate-pass α={best['alpha']:.2f} Δ={best['delta']:+.5f}")
        # Emit submission candidate
        alpha = best["alpha"]
        blend_test = log_blend([v1_test, bag_test], np.array([1.0 - alpha, alpha]))
        b_pred = (safelog(blend_test) + v1_bias).argmax(1)
        sub_path = SUB / f"submission_path2_seedbag_a{int(alpha*100):03d}.csv"
        pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in b_pred]}).to_csv(sub_path, index=False)
        log(f"  emitted {sub_path}")
        # Pure-bag submission (α=1.0 effectively, but use bag's own bias)
    return dict(sweep=sweep, n_passing=len(passing))


if __name__ == "__main__":
    main()
