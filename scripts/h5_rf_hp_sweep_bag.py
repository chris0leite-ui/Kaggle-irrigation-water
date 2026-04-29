"""H5 — RF HP sweep + bag.

v1 was (n_estimators=500, max_depth=12). Run 4 HP variants on v1's
exact 7-component bank, all class_weight=None, then geomean-bag
the 4 OOFs/test. HP-axis variance reduction.

Variants:
  V1: (n_est=300, max_depth=10)   [under-capacity]
  V2: (n_est=500, max_depth=14)   [v1 + deeper trees]
  V3: (n_est=800, max_depth=12)   [more trees, v1 depth]
  V4: (n_est=500, max_depth=16)   [v1 trees, much deeper]

Output: per-variant OOF + final geomean-bag OOF/test.
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

VARIANTS = [
    ("V1", dict(n_estimators=300, max_depth=10)),
    ("V2", dict(n_estimators=500, max_depth=14)),
    ("V3", dict(n_estimators=800, max_depth=12)),
    ("V4", dict(n_estimators=500, max_depth=16)),
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))
def _normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = {}
    for name in V1_BANK:
        oof = _normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        tt = _normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        pool[name] = (oof, tt)
        log(f"  + {name}")

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in names]
    log_te = [safelog(pool[n][1]) for n in names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    n_folds = 2 if SMOKE else 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))

    per_variant = {}
    oofs, tests = [], []
    variants = VARIANTS[:1] if SMOKE else VARIANTS
    for tag, hp in variants:
        log(f"=== {tag}  hp={hp} ===")
        oof = np.zeros((n_tr, 3), dtype=np.float32)
        test_pred = np.zeros((n_te, 3), dtype=np.float32)
        fs = []
        rf_params = dict(
            min_samples_leaf=20, max_features="sqrt",
            bootstrap=True, n_jobs=-1, random_state=SEED,
            class_weight=None, verbose=0, **hp,
        )
        if SMOKE:
            rf_params["n_estimators"] = 100
            rf_params["max_depth"] = 6
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            t0 = time.time()
            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_tr_s[tr_idx], y[tr_idx])
            p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
            p_te = rf.predict_proba(X_te_s).astype(np.float32)
            oof[va_idx] = p_va
            test_pred += p_te / n_folds
            bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
            fs.append(float(bal))
            log(f"  {tag} fold {fold} bal={bal:.5f} wall={time.time()-t0:.1f}s")
        bal = balanced_accuracy_score(y, oof.argmax(1))
        prior = np.bincount(y, minlength=3) / len(y)
        bias, tuned = tune_log_bias(_normed(oof), y, prior)
        pcr = per_class_recall(y, (safelog(_normed(oof)) + bias).argmax(1))
        log(f"  {tag} argmax={bal:.5f} tuned={tuned:.5f} bias={bias.round(4).tolist()}")
        log(f"  {tag} PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")
        per_variant[tag] = dict(
            hp=hp, fold_scores=fs, argmax=float(bal),
            tuned=float(tuned), bias=bias.tolist(), pcr=pcr.tolist(),
        )
        np.save(ART / f"oof_h5_{tag}.npy", _normed(oof))
        np.save(ART / f"test_h5_{tag}.npy", _normed(test_pred))
        oofs.append(_normed(oof))
        tests.append(_normed(test_pred))

    log("=== H5 geomean bag ===")
    log_oofs = np.stack([safelog(o) for o in oofs], axis=0)
    log_tests = np.stack([safelog(t) for t in tests], axis=0)
    bag_oof = _normed(np.exp(log_oofs.mean(axis=0)))
    bag_test = _normed(np.exp(log_tests.mean(axis=0)))

    bag_argmax = balanced_accuracy_score(y, bag_oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bag_bias, bag_tuned = tune_log_bias(bag_oof, y, prior)
    pcr = per_class_recall(y, (safelog(bag_oof) + bag_bias).argmax(1))
    log(f"BAG argmax={bag_argmax:.5f} tuned={bag_tuned:.5f} bias={bag_bias.round(4).tolist()}")
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / "oof_h5_hp_bag.npy", bag_oof)
    np.save(ART / "test_h5_hp_bag.npy", bag_test)

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    bag_test_pred = (safelog(bag_test) + bag_bias).argmax(1)
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)
    diff = int((bag_test_pred != v1_test_pred).sum())

    summary = dict(smoke=SMOKE, per_variant=per_variant,
                   bag_argmax=float(bag_argmax),
                   bag_tuned=float(bag_tuned),
                   bag_bias=bag_bias.tolist(), bag_pcr=pcr.tolist(),
                   v1_tuned=float(v1_tuned),
                   delta_tuned_vs_v1=float(bag_tuned - v1_tuned),
                   bag_vs_v1_test_diff=diff)
    with open(ART / "h5_hp_bag_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"wrote {ART}/h5_hp_bag_results.json")

    sub_path = SUB / "submission_h5_hp_bag_standalone.csv"
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in bag_test_pred]})
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")


if __name__ == "__main__":
    main()
