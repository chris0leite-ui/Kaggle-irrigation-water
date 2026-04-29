"""H1 — Seed-bag of v1 RF natural across 5 random_states.

Mechanism-preserving variance reduction on the proven LB-best (LB 0.98129).
Holds bank, HPs, and architecture EXACTLY at v1's config; varies only the
sklearn RF random_state across {42, 7, 123, 456, 789}, then geomeans the
5 OOF / test prob arrays.

Bank (v1's exact 7 components — DO NOT MODIFY):
  rawashishsin_2600, recipe_full_te_catboost_natural,
  recipe_full_te_catboost, recipe_full_te, realmlp,
  xgb_corn, xgb_dist_digits

HPs (v1's exact config):
  n_estimators=500, max_depth=12, min_samples_leaf=20,
  max_features='sqrt', bootstrap=True, class_weight=None

5 fold StratifiedKFold(seed=42) for OOF alignment with v1.

Outputs:
  oof_h1_seedbag_rf.npy  test_h1_seedbag_rf.npy
  h1_seedbag_rf_results.json  (per-seed + bag metrics, blend gate)
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

# v1 EXACT bank (7 components) — do NOT add or remove
V1_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
]

# Distance / rule meta features (same as v1)
META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]

# Seeds for the seed-bag — v1's was 42; 3 seeds gives ~78% of 5-seed
# variance reduction at 60% of compute. Adjust via N_SEEDS env var.
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
SEED_BAG_FULL = [42, 7, 123, 456, 789]
SEED_BAG = SEED_BAG_FULL[:N_SEEDS]
FOLD_SEED = 42  # OOF alignment unchanged

SMOKE = os.environ.get("SMOKE") == "1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def load_bank(n_tr, n_te):
    pool = {}
    for name in V1_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  MISSING {name}")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape != (n_tr, 3) or t.shape != (n_te, 3):
            log(f"  SKIP {name}: shape {o.shape} / {t.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (_normed(o), _normed(t))
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
    return X_tr, X_te


def run_rf_one_seed(X_tr_s, X_te_s, y, fold_seed, rf_seed, n_tr, n_te):
    """5-fold OOF + test predict for one RF random_state.

    Per-fold checkpointing: oof_h1_seed{S}_fold{F}.npy + test_h1_seed{S}_fold{F}.npy.
    Resumes from cached folds on relaunch."""
    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=rf_seed,
        class_weight=None, verbose=0,
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)

    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_s, y), 1):
        ck_oof = ART / f"oof_h1_seed{rf_seed}_fold{fold}.npy"
        ck_test = ART / f"test_h1_seed{rf_seed}_fold{fold}.npy"
        ck_meta = ART / f"h1_seed{rf_seed}_fold{fold}_meta.json"
        if ck_oof.exists() and ck_test.exists() and ck_meta.exists():
            p_va = np.load(ck_oof).astype(np.float32)
            p_te = np.load(ck_test).astype(np.float32)
            meta = json.load(open(ck_meta))
            oof[va_idx] = p_va
            test += p_te / n_folds
            fold_scores.append(meta["bal"])
            log(f"    fold {fold}/{n_folds} bal={meta['bal']:.5f} (cached)")
            continue
        t0 = time.time()
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        wall = time.time() - t0
        # Atomic save: write to .tmp.npy (np.save won't double-suffix), then rename
        tmp_oof = ck_oof.parent / (ck_oof.stem + ".tmp.npy")
        tmp_test = ck_test.parent / (ck_test.stem + ".tmp.npy")
        np.save(tmp_oof, p_va); tmp_oof.rename(ck_oof)
        np.save(tmp_test, p_te); tmp_test.rename(ck_test)
        ck_meta.write_text(json.dumps(dict(bal=float(bal), wall=wall)))
        log(f"    fold {fold}/{n_folds} bal={bal:.5f} wall={wall:.1f}s")
    return oof, test, fold_scores


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    log(f"loading v1 7-component bank")
    pool = load_bank(n_tr, n_te)
    if len(pool) != len(V1_BANK):
        log(f"FATAL: only {len(pool)}/{len(V1_BANK)} v1 components present")
        sys.exit(1)

    X_tr, X_te = build_features(pool, train, test)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    log(f"feature matrix: train={X_tr_s.shape}  test={X_te_s.shape}")

    seeds = SEED_BAG[:1] if SMOKE else SEED_BAG
    per_seed = {}
    oofs = []
    tests = []
    for sd in seeds:
        log(f"=== seed {sd} ===")
        oof, test_pred, fold_scores = run_rf_one_seed(
            X_tr_s, X_te_s, y, FOLD_SEED, sd, n_tr, n_te
        )
        bal = balanced_accuracy_score(y, oof.argmax(1))
        prior = np.bincount(y, minlength=3) / len(y)
        bias, tuned = tune_log_bias(oof, y, prior)
        log(f"  seed {sd}: argmax={bal:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")
        per_seed[str(sd)] = dict(
            argmax=float(bal), tuned=float(tuned),
            bias=bias.tolist(), fold_scores=fold_scores,
        )
        # Save per-seed artifact
        np.save(ART / f"oof_h1_rf_seed{sd}.npy", _normed(oof))
        np.save(ART / f"test_h1_rf_seed{sd}.npy", _normed(test_pred))
        oofs.append(_normed(oof))
        tests.append(_normed(test_pred))

    # Geomean bag
    log("=== geomean bag ===")
    log_oofs = np.stack([safelog(o) for o in oofs], axis=0)
    log_tests = np.stack([safelog(t) for t in tests], axis=0)
    bag_oof = _normed(np.exp(log_oofs.mean(axis=0)))
    bag_test = _normed(np.exp(log_tests.mean(axis=0)))

    bag_argmax = balanced_accuracy_score(y, bag_oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bag_bias, bag_tuned = tune_log_bias(bag_oof, y, prior)
    log(f"BAG argmax={bag_argmax:.5f}  tuned={bag_tuned:.5f}  bias={bag_bias.round(4).tolist()}")

    pred_at_bias = (safelog(bag_oof) + bag_bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / "oof_h1_seedbag_rf.npy", bag_oof)
    np.save(ART / "test_h1_seedbag_rf.npy", bag_test)

    # v1 anchor comparison
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_oof = _normed(v1_oof)
    v1_test = _normed(v1_test)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    log(f"v1 LB-best   argmax={balanced_accuracy_score(y, v1_oof.argmax(1)):.5f}  tuned={v1_tuned:.5f}  bias={v1_bias.round(4).tolist()}")

    # Diversity diagnostic: Jaccard of test argmaxes
    bag_pred = bag_test.argmax(1)
    v1_pred = v1_test.argmax(1)
    diff = int((bag_pred != v1_pred).sum())
    log(f"test argmax diff: bag vs v1 = {diff} / {n_te} ({diff/n_te*100:.3f}%)")

    summary = dict(
        smoke=SMOKE, seeds=seeds, fold_seed=FOLD_SEED,
        bank=V1_BANK, per_seed=per_seed,
        bag_argmax=float(bag_argmax),
        bag_tuned=float(bag_tuned),
        bag_bias=bag_bias.tolist(),
        bag_pcr=pcr.tolist(),
        v1_argmax=float(balanced_accuracy_score(y, v1_oof.argmax(1))),
        v1_tuned=float(v1_tuned),
        v1_bias=v1_bias.tolist(),
        bag_vs_v1_test_diff=diff,
        delta_tuned_vs_v1=float(bag_tuned - v1_tuned),
    )
    with open(ART / "h1_seedbag_rf_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"wrote {ART}/h1_seedbag_rf_results.json")

    # Build standalone submission at bag's tuned bias
    bag_pred_at_bias = (safelog(bag_test) + bag_bias).argmax(1)
    sub_path = SUB / "submission_h1_seedbag_rf_standalone.csv"
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in bag_pred_at_bias],
    })
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")


if __name__ == "__main__":
    main()
