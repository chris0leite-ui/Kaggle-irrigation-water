"""#5 RF natural meta HP search (simplified NSGA-II → random search).

Sweep 5 HP configs around v1's (n_est=500, max_depth=12, max_features=sqrt,
min_samples_leaf=20). Each gets full 5-fold seed=42 + per-class recall +
macro-recall. Pick Pareto-non-dominated configs.

Goal: find a different RF configuration that:
  - Beats v1 OOF (0.98063) by >+5e-4 OR
  - Has materially different per-class recall profile (different OTHERS
    pool member for override mechanism)
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
from common import tune_log_bias, add_distance_features  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
SEED = 42

# Bank components for natural-cal RF meta-stacker (same as v1)
NATURAL_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
]
META_COLS = [
    "dgp_score", "sm_dist", "rf_dist", "tc_dist", "ws_dist",
    "sm_abs", "rf_abs", "tc_abs", "ws_abs",
    "min_axis_abs", "min_boundary_dist",
    "norain", "windy", "rule_pred",
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def _normed(a, eps=1e-9):
    return a / np.clip(a.sum(1, keepdims=True), eps, None)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)
    n_tr, n_te = len(train), len(test)

    # Load bank
    pool = {}
    for name in NATURAL_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  SKIP {name}")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape != (n_tr, 3) or t.shape != (n_te, 3):
            log(f"  SKIP {name}: shape mismatch")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")

    log(f"Loaded {len(pool)}/{len(NATURAL_BANK)} components")
    if "rawashishsin_2600" not in pool:
        log("ERROR: rawashishsin missing")
        return

    # Build features
    log("Building features")
    tr_d = add_distance_features(train.drop(columns=[TARGET]))
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)
    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    log(f"Feature matrix: train={X_tr_s.shape}  test={X_te_s.shape}")

    # HP grid (5 configs around v1: n_est=500, max_depth=12, max_features=sqrt, min_samples_leaf=20)
    configs = [
        dict(name="v1_baseline",  n_estimators=500, max_depth=12, max_features="sqrt", min_samples_leaf=20),
        dict(name="deeper",       n_estimators=500, max_depth=16, max_features="sqrt", min_samples_leaf=20),
        dict(name="more_trees",   n_estimators=800, max_depth=12, max_features="sqrt", min_samples_leaf=20),
        dict(name="wider_feats",  n_estimators=500, max_depth=12, max_features=0.5,    min_samples_leaf=20),
        dict(name="tight_leaf",   n_estimators=500, max_depth=12, max_features="sqrt", min_samples_leaf=5),
    ]
    log(f"Sweeping {len(configs)} configs")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))

    results = []
    for cfg in configs:
        log(f"\n=== {cfg['name']}: {cfg} ===")
        params = dict(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            max_features=cfg["max_features"],
            min_samples_leaf=cfg["min_samples_leaf"],
            bootstrap=True, n_jobs=-1, random_state=SEED,
            class_weight=None, verbose=0,
        )
        oof = np.zeros((n_tr, 3), dtype=np.float32)
        test_pred = np.zeros((n_te, 3), dtype=np.float32)
        for fold, (tr, va) in enumerate(splits):
            t0 = time.time()
            rf = RandomForestClassifier(**params)
            rf.fit(X_tr_s[tr], y[tr])
            oof[va] = rf.predict_proba(X_tr_s[va]).astype(np.float32)
            test_pred += rf.predict_proba(X_te_s).astype(np.float32) / 5.0
            log(f"  fold {fold+1}: {time.time()-t0:.1f}s")
        bias, tuned = tune_log_bias(oof, y, prior)
        argm = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(1)
        pcr = per_class_recall(y, argm)
        log(f"  OOF tuned: {tuned:.5f}  bias={bias.round(3).tolist()}")
        log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

        # Save artifacts
        np.save(ART / f"oof_n5_rf_{cfg['name']}.npy", oof)
        np.save(ART / f"test_n5_rf_{cfg['name']}.npy", test_pred)

        results.append(dict(
            name=cfg["name"], cfg=cfg, tuned=float(tuned),
            bias=bias.tolist(), pcr=pcr.tolist(),
            oof_path=f"oof_n5_rf_{cfg['name']}.npy",
        ))

    # Find Pareto winners
    log(f"\n=== Pareto frontier (macro-recall, min-class-recall) ===")
    print(f"{'name':<15}{'tuned':>10}{'min_pcr':>10}{'pcr_L':>9}{'pcr_M':>9}{'pcr_H':>9}")
    for r in sorted(results, key=lambda x: -x["tuned"]):
        min_pcr = min(r["pcr"])
        print(f"  {r['name']:<13}{r['tuned']:>10.5f}{min_pcr:>10.4f}"
              f"{r['pcr'][0]:>9.4f}{r['pcr'][1]:>9.4f}{r['pcr'][2]:>9.4f}")

    # Best by tuned OOF
    best = max(results, key=lambda x: x["tuned"])
    log(f"\nBest: {best['name']} tuned={best['tuned']:.5f}")
    if best["tuned"] > 0.98063 + 5e-4:
        log(f"  PASSES gate (>+5e-4 over v1 0.98063)")
    else:
        log(f"  Sub-gate ({best['tuned']-0.98063:+.5f} vs v1)")

    # Save submission of best
    bias = np.array(best["bias"])
    test_p = np.load(ART / f"test_n5_rf_{best['name']}.npy")
    test_pred = (np.log(np.clip(test_p, 1e-9, 1.0)) + bias).argmax(1)
    path = SUB / f"submission_n5_rf_{best['name']}.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in test_pred]}).to_csv(path, index=False)
    log(f"Saved: {path}")

    # Compare to v1
    v1_pred = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")[TARGET].map(CLS2IDX).to_numpy()
    diff = int((test_pred != v1_pred).sum())
    log(f"Test diff vs v1: {diff} rows")

    with open(ART / "n5_rf_hp_search_results.json", "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    log(f"Saved summary")


if __name__ == "__main__":
    main()
