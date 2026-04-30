"""#5 RF natural meta HP search — QUICK version.

Single-fold (val-only) screening of 4 HP configs to find any that
materially differs from v1's PCR profile. Full 5-fold only on the winner.

Each config: train on 80% of train, evaluate on 20% val. ~3-5 min per config.
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
def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))
def _normed(a, eps=1e-9): return a / np.clip(a.sum(1, keepdims=True), eps, None)


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

    pool = {}
    for name in NATURAL_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape != (n_tr, 3):
            continue
        pool[name] = (_normed(o), _normed(t))
    log(f"Loaded {len(pool)}/{len(NATURAL_BANK)}")

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

    # Use first fold of 5-fold seed=42 for quick screening
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_tr_s, y))
    tr_idx, va_idx = splits[0]
    log(f"Quick screen: train {len(tr_idx)}, val {len(va_idx)}")

    configs = [
        dict(name="v1_baseline",  n_estimators=500, max_depth=12, max_features="sqrt", min_samples_leaf=20),
        dict(name="deeper",       n_estimators=500, max_depth=16, max_features="sqrt", min_samples_leaf=20),
        dict(name="wider_feats",  n_estimators=500, max_depth=12, max_features=0.5,    min_samples_leaf=20),
        dict(name="tight_leaf",   n_estimators=500, max_depth=12, max_features="sqrt", min_samples_leaf=5),
    ]

    results = []
    for cfg in configs:
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            max_features=cfg["max_features"],
            min_samples_leaf=cfg["min_samples_leaf"],
            bootstrap=True, n_jobs=-1, random_state=42,
            class_weight=None, verbose=0,
        )
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        proba = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        # Tune bias on this val (not perfect but quick)
        bias, tuned = tune_log_bias(proba, y[va_idx], prior)
        argm = (np.log(np.clip(proba, 1e-9, 1.0)) + bias).argmax(1)
        bal = balanced_accuracy_score(y[va_idx], argm)
        pcr = per_class_recall(y[va_idx], argm)
        elapsed = time.time() - t0
        results.append(dict(cfg=cfg, bal=float(bal), tuned=float(tuned), bias=bias.tolist(),
                            pcr=pcr.tolist(), elapsed=elapsed))
        log(f"  {cfg['name']}: tuned {tuned:.5f}  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]  ({elapsed:.0f}s)")

    log(f"\n=== Ranking ===")
    for r in sorted(results, key=lambda x: -x["tuned"]):
        log(f"  {r['cfg']['name']:<15} tuned={r['tuned']:.5f}  PCR=[L={r['pcr'][0]:.4f} M={r['pcr'][1]:.4f} H={r['pcr'][2]:.4f}]")

    best = max(results, key=lambda x: x["tuned"])
    log(f"\nBest: {best['cfg']['name']} tuned={best['tuned']:.5f}")
    # v1 fold-0 reference: ~0.98060 on first fold
    log(f"  v1 fold-0 reference is around 0.98050-0.98070")

    with open(ART / "n5_rf_hp_search_quick_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved")


if __name__ == "__main__":
    main()
