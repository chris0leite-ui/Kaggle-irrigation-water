"""A2: Multi-seed bag of RF natural meta-stacker.

Mirror of sklearn_rf_meta_natural.py architecture (LB 0.98129) — same 7-component
bank, same XGB hyperparams. Only random_state varies across seeds {42, 7, 123}.
Reuses existing seed=42 OOF on disk; trains seeds 7 and 123 fresh.

Bag = log-mean of the three OOF / test posteriors. Tune log-bias on the bag.

SMOKE=1: 50k rows, 2 folds, n_est=100, max_depth=8, single seed (42).

Outputs:
  scripts/artifacts/oof_sklearn_rf_meta_natural_seed{42,7,123}.npy
  scripts/artifacts/test_sklearn_rf_meta_natural_seed{42,7,123}.npy
  scripts/artifacts/oof_sklearn_rf_meta_natural_bag.npy
  scripts/artifacts/test_sklearn_rf_meta_natural_bag.npy
  scripts/artifacts/sklearn_rf_meta_natural_bag_results.json
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
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
FOLD_SEED = 42
SEEDS = [42, 7, 123]
SMOKE = os.environ.get("SMOKE") == "1"

# LB 0.98129 bank — frozen 7 components (NOT the post-LB A1-expansion list)
LB_BEST_BANK = [
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
    log(f"loading natural-cal LB-best bank ({len(LB_BEST_BANK)} components)")
    pool = {}
    for name in LB_BEST_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  MISSING {name}")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape != (n_tr, 3) or t.shape != (n_te, 3):
            log(f"  SKIP {name}: shape mismatch oof={o.shape} test={t.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (normed(o), normed(t))
        log(f"  + {name}")
    return pool


def build_features(pool, train, test):
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in names]
    log_te = [safelog(pool[n][1]) for n in names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  feature matrix: train={X_tr.shape}  test={X_te.shape}")
    return X_tr, X_te, names


def train_one_seed(seed, X_tr, X_te, y, n_tr, n_te, n_folds, n_est, max_depth):
    """Train 5-fold RF at the given random_state. Returns (oof, test_pred)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=FOLD_SEED)
    splits = list(skf.split(X_tr, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_leaf=20, max_features="sqrt",
            bootstrap=True, n_jobs=-1, random_state=seed,
            class_weight=None, verbose=0,
        )
        rf.fit(X_tr[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"    seed={seed} fold {fold}/{n_folds} argmax={bal:.5f}  wall={time.time()-t0:.1f}s")
    return oof, test_pred, fold_scores


def get_or_train_seed(seed, X_tr, X_te, y, n_tr, n_te, n_folds, n_est, max_depth, smoke=False):
    """Reuse on-disk artifacts if present; else train + save."""
    suffix = "_smoke" if smoke else ""
    oof_p = ART / f"oof_sklearn_rf_meta_natural_seed{seed}{suffix}.npy"
    test_p = ART / f"test_sklearn_rf_meta_natural_seed{seed}{suffix}.npy"
    # seed=42 may exist as the unsuffixed LB 0.98129 result (production only)
    if seed == 42 and not smoke and not oof_p.exists():
        legacy_oof = ART / "oof_sklearn_rf_meta_natural.npy"
        legacy_test = ART / "test_sklearn_rf_meta_natural.npy"
        if legacy_oof.exists() and legacy_test.exists():
            log(f"  seed=42: copying legacy LB 0.98129 artifacts to seed-suffixed names")
            np.save(oof_p, np.load(legacy_oof))
            np.save(test_p, np.load(legacy_test))
    if oof_p.exists() and test_p.exists():
        oof = np.load(oof_p).astype(np.float32)
        test_pred = np.load(test_p).astype(np.float32)
        if oof.shape == (n_tr, 3) and test_pred.shape == (n_te, 3) and not (oof.sum(1) < 1e-3).any():
            log(f"  seed={seed}: reusing on-disk artifacts")
            return oof, test_pred, None
        log(f"  seed={seed}: on-disk artifacts have wrong shape/zeros, retraining")
    log(f"  seed={seed}: training fresh")
    oof, test_pred, fold_scores = train_one_seed(seed, X_tr, X_te, y, n_tr, n_te, n_folds, n_est, max_depth)
    np.save(oof_p, oof)
    np.save(test_p, test_pred)
    log(f"  seed={seed}: saved {oof_p.name} + {test_p.name}")
    return oof, test_pred, fold_scores


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    n_tr, n_te = len(train), len(test)

    pool = load_bank(n_tr, n_te)
    if "rawashishsin_2600" not in pool:
        log("ERROR: rawashishsin_2600 missing — abort")
        return
    if len(pool) != len(LB_BEST_BANK):
        log(f"ERROR: pool size {len(pool)} != expected {len(LB_BEST_BANK)} — abort to keep LB-best architecture")
        return

    X_tr, X_te, names = build_features(pool, train, test)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    del X_tr, X_te
    import gc; gc.collect()

    if SMOKE:
        sub_idx = np.arange(50_000)
        X_tr_s = X_tr_s[sub_idx]
        y = y[sub_idx]
        n_tr = len(sub_idx)
        n_folds = 2
        n_est = 100
        max_depth = 8
        seeds = [42]
    else:
        n_folds = 5
        n_est = 500
        max_depth = 12
        seeds = SEEDS
    log(f"config: n_folds={n_folds} n_est={n_est} max_depth={max_depth} seeds={seeds}")

    per_seed_oof = {}
    per_seed_test = {}
    per_seed_scores = {}
    for seed in seeds:
        log(f"=== seed {seed} ===")
        oof, test_pred, fold_scores = get_or_train_seed(
            seed, X_tr_s, X_te_s, y, n_tr, n_te, n_folds, n_est, max_depth, smoke=SMOKE)
        per_seed_oof[seed] = oof
        per_seed_test[seed] = test_pred
        per_seed_scores[seed] = fold_scores
        seed_argmax = balanced_accuracy_score(y, oof.argmax(1))
        log(f"  seed={seed} OOF argmax = {seed_argmax:.5f}")

    # Log-mean bag
    log("=== log-mean bag ===")
    bag_oof_log = np.mean([safelog(per_seed_oof[s]) for s in seeds], axis=0)
    bag_test_log = np.mean([safelog(per_seed_test[s]) for s in seeds], axis=0)
    bag_oof = np.exp(bag_oof_log - bag_oof_log.max(1, keepdims=True))
    bag_oof = bag_oof / bag_oof.sum(1, keepdims=True)
    bag_test = np.exp(bag_test_log - bag_test_log.max(1, keepdims=True))
    bag_test = bag_test / bag_test.sum(1, keepdims=True)

    bag_argmax = balanced_accuracy_score(y, bag_oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(bag_oof, y, prior)
    log(f"  bag OOF argmax = {bag_argmax:.5f}  tuned = {tuned:.5f}")
    log(f"  bag log-bias = {bias.round(4).tolist()}")

    pred_at_bias = (safelog(bag_oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Per-seed standalone tuned for ladder comparison
    seed_tuned = {}
    for s in seeds:
        bias_s, tuned_s = tune_log_bias(per_seed_oof[s], y, prior)
        seed_tuned[s] = (bias_s.tolist(), float(tuned_s))
        log(f"  seed={s}: tuned={tuned_s:.5f}  bias={bias_s.round(4).tolist()}")

    # Save bag artifacts
    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_sklearn_rf_meta_natural_bag{suffix}.npy", bag_oof.astype(np.float32))
    np.save(ART / f"test_sklearn_rf_meta_natural_bag{suffix}.npy", bag_test.astype(np.float32))

    summary = dict(
        smoke=SMOKE, n_folds=n_folds, n_est=n_est, max_depth=max_depth,
        seeds=seeds, bank=LB_BEST_BANK, fold_seed=FOLD_SEED,
        per_seed_fold_scores={str(k): v for k, v in per_seed_scores.items()},
        per_seed_tuned={str(k): v for k, v in seed_tuned.items()},
        bag_overall_argmax=float(bag_argmax),
        bag_tuned=float(tuned),
        bag_log_bias=bias.tolist(),
        bag_bias_H=float(bias[2]),
        bag_per_class_recall=pcr.tolist(),
    )
    out_p = ART / f"sklearn_rf_meta_natural_bag{suffix}_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
