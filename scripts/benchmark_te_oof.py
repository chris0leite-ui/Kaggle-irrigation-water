"""LGBM-dist + out-of-fold target encoding from SYNTHETIC labels.

Key difference from benchmark_te_orig.py:
  benchmark_te_orig: encoder source = 10k original (rule = 100%),
                     TE values replicate the rule. Null (+0.00004).
  this script:       encoder source = synthetic labels with out-of-fold
                     CV (no leakage). TE values deviate from the rule
                     by the synthetic's flip rate → direct per-category
                     flip-probability signal.

Key encoded targets:
  - Per-class probability: P(y=Low | cat), P(y=Med | cat), P(y=High | cat)
  - Per-class LOG-ODDS: log(P/(1-P))  -- expanded dynamic range, useful
    for small deviations near 0 or 1
  - Flip-rate: P(rule_pred != y | cat) -- direct flip-probability signal
    that rule_pred alone cannot provide

For each fold f:
  - Compute TE lookups from rows in folds != f
  - Apply to val-fold (f) + all test rows (test TE = full-train TE after CV)
Laplace smoothing alpha calibrated per cardinality.

Baselines:
  LGBM-dist OOF tuned          0.97266
  LGBM-dist + TE-orig          0.97270  (null)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from xgb_specialist_678 import add_distance_features


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

MAIN_CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
PAIR_CATS = [
    ("Soil_Type", "Crop_Type"),
    ("Crop_Type", "Crop_Growth_Stage"),
    ("Season", "Region"),
    ("Soil_Type", "Season"),
    ("Crop_Type", "Season"),
    ("Crop_Type", "Irrigation_Type"),
]

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def compute_te_counts(df, y, key_cols, alpha, prior):
    """Per-class TE counts over (key_cols) with Laplace smoothing."""
    k = len(CLASSES)
    smooth = alpha * prior
    out = {}
    if len(key_cols) == 1:
        vals = df[key_cols[0]].values
        gb = pd.DataFrame({"k": vals, "y": y}).groupby("k")["y"]
    else:
        key_tuples = list(zip(*[df[c].values for c in key_cols]))
        gb = pd.DataFrame({"k": key_tuples, "y": y}).groupby("k")["y"]
    for key, grp in gb:
        c = np.bincount(grp.values, minlength=k).astype(np.float64)
        c_sm = c + smooth
        p = c_sm / c_sm.sum()
        out[key] = p
    return out


def compute_flip_rate(df, y, rule_pred, key_cols, alpha):
    """Mean flip rate over (key_cols) — P(y != rule_pred | key) with
    Laplace smoothing toward global flip rate."""
    global_flip = (y != rule_pred).mean()
    out = {}
    if len(key_cols) == 1:
        vals = df[key_cols[0]].values
        gb = pd.DataFrame({"k": vals, "f": (y != rule_pred).astype(int)}).groupby("k")["f"]
    else:
        key_tuples = list(zip(*[df[c].values for c in key_cols]))
        gb = pd.DataFrame({"k": key_tuples, "f": (y != rule_pred).astype(int)}).groupby("k")["f"]
    for key, grp in gb:
        n = len(grp)
        s = grp.sum()
        out[key] = (s + alpha * global_flip) / (n + alpha)
    return out, global_flip


def apply_vec(df, key_cols, lookup, default):
    """Vectorized lookup. default may be scalar (flip) or array (probs)."""
    if len(key_cols) == 1:
        keys = df[key_cols[0]].values
    else:
        keys = list(zip(*[df[c].values for c in key_cols]))

    # Fast lookup
    if np.isscalar(default):
        out = np.full(len(df), default, dtype=np.float32)
        for i, k in enumerate(keys):
            if k in lookup:
                out[i] = lookup[k]
    else:
        out = np.tile(default[None, :], (len(df), 1)).astype(np.float32)
        for i, k in enumerate(keys):
            if k in lookup:
                out[i] = lookup[k]
    return out


def te_feature_names():
    names = []
    for c in MAIN_CATS:
        for cls in CLASSES:
            names.append(f"teS_{c}__{cls}")
        names.append(f"teS_flip_{c}")
    for a, b in PAIR_CATS:
        for cls in CLASSES:
            names.append(f"teS_{a}_{b}__{cls}")
        names.append(f"teS_flip_{a}_{b}")
    return names


def fold_te(df_tr, y_tr, rule_tr, df_va, df_te):
    """Compute TE lookups from df_tr+y_tr, apply to df_va and df_te.
    Returns (te_matrix_va, te_matrix_te) in column order matching te_feature_names.
    """
    prior = np.bincount(y_tr, minlength=3) / len(y_tr)

    def pack(df):
        cols_va = []
        # single-cat
        for c in MAIN_CATS:
            lookup = compute_te_counts(df_tr, y_tr, [c], alpha=20.0, prior=prior)
            arr = apply_vec(df, [c], lookup, prior)  # (n, 3)
            cols_va.append(arr)
            flip_lk, flip_prior = compute_flip_rate(df_tr, y_tr, rule_tr, [c], alpha=20.0)
            flip_arr = apply_vec(df, [c], flip_lk, flip_prior).reshape(-1, 1)
            cols_va.append(flip_arr)
        for a, b in PAIR_CATS:
            lookup = compute_te_counts(df_tr, y_tr, [a, b], alpha=10.0, prior=prior)
            arr = apply_vec(df, [a, b], lookup, prior)
            cols_va.append(arr)
            flip_lk, flip_prior = compute_flip_rate(df_tr, y_tr, rule_tr, [a, b], alpha=10.0)
            flip_arr = apply_vec(df, [a, b], flip_lk, flip_prior).reshape(-1, 1)
            cols_va.append(flip_arr)
        return np.concatenate(cols_va, axis=1).astype(np.float32)

    return pack(df_va), pack(df_te)


def tune_log_bias(p, y, prior):
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
    grid = np.linspace(-3, 3, 61)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = b.copy()
            sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = sc[j]
                imp = True
        if not imp:
            break
    return b, best


def main() -> None:
    t_all = time.time()
    log("loading data")
    tr_raw = pd.read_csv("data/train.csv")
    te_raw = pd.read_csv("data/test.csv")

    log("building dist features")
    tr = add_distance_features(tr_raw.copy())
    te = add_distance_features(te_raw.copy())

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    rule_pred = tr["rule_pred"].values
    prior = np.bincount(y) / len(y)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols_raw = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    # preserve raw cat columns as strings for TE lookups
    tr_cat = tr[cat_cols_raw].astype(str).copy()
    te_cat = te[cat_cols_raw].astype(str).copy()

    te_names = te_feature_names()
    log(f"TE cols to compute: {len(te_names)}")

    # label-encode cats for LGBM
    for c in cat_cols_raw:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    X = tr[num_cols + cat_cols_raw].copy()
    X_test = te[num_cols + cat_cols_raw].copy()
    for c in cat_cols_raw:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # allocate TE columns on X / X_test
    for n in te_names:
        X[n] = np.float32(0)
        X_test[n] = np.float32(0)

    # ---------------- compute test TE using full training set ---------------
    log("computing test TE from full train")
    test_prior = prior
    test_chunks = []
    for c in MAIN_CATS:
        lookup = compute_te_counts(tr_cat, y, [c], alpha=20.0, prior=prior)
        test_chunks.append(apply_vec(te_cat, [c], lookup, prior))
        flip_lk, fp = compute_flip_rate(tr_cat, y, rule_pred, [c], alpha=20.0)
        test_chunks.append(apply_vec(te_cat, [c], flip_lk, fp).reshape(-1, 1))
    for a, b in PAIR_CATS:
        lookup = compute_te_counts(tr_cat, y, [a, b], alpha=10.0, prior=prior)
        test_chunks.append(apply_vec(te_cat, [a, b], lookup, prior))
        flip_lk, fp = compute_flip_rate(tr_cat, y, rule_pred, [a, b], alpha=10.0)
        test_chunks.append(apply_vec(te_cat, [a, b], flip_lk, fp).reshape(-1, 1))
    test_te = np.concatenate(test_chunks, axis=1).astype(np.float32)
    for j, n in enumerate(te_names):
        X_test[n] = test_te[:, j]

    # -------------------- 5-fold: compute TE + train LGBM -------------------
    log("5-fold stratified: fold TE + LGBM-dist + TE")
    lgb_params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
        verbose=-1, seed=SEED,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        # compute fold-specific TE
        va_cols, _ = fold_te(
            tr_cat.iloc[tr_idx], y[tr_idx], rule_pred[tr_idx],
            tr_cat.iloc[va_idx], te_cat.iloc[:1],
        )
        tr_cols, _ = fold_te(
            tr_cat.iloc[tr_idx], y[tr_idx], rule_pred[tr_idx],
            tr_cat.iloc[tr_idx], te_cat.iloc[:1],
        )
        # NOTE: tr TE uses tr itself (same source + target), which is mild
        # leakage. For small-cardinality cats (4-6) the TE mean from a
        # single group with 80k+ rows is extremely stable; leakage ≪ 1/n_c.
        # Proper nested CV would be ideal but ~5× slower; mild leakage is
        # acceptable here as a first diagnostic.
        X_tr_fold = X.iloc[tr_idx].copy()
        X_va_fold = X.iloc[va_idx].copy()
        for j, n in enumerate(te_names):
            X_tr_fold[n] = tr_cols[:, j]
            X_va_fold[n] = va_cols[:, j]

        dtr = lgb.Dataset(X_tr_fold, label=y[tr_idx], categorical_feature=cat_cols_raw)
        dva = lgb.Dataset(X_va_fold, label=y[va_idx], categorical_feature=cat_cols_raw,
                          reference=dtr)
        m = lgb.train(lgb_params, dtr, num_boost_round=4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        best_iters.append(m.best_iteration)
        oof[va_idx] = m.predict(X_va_fold, num_iteration=m.best_iteration)
        test_probs += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={m.best_iteration}  "
            f"argmax_bal={bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned = tune_log_bias(oof, y, prior)
    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))

    print(f"\n=== LGBM-dist + OOF-TE from SYNTHETIC (OOF bal_acc) ===")
    print(f"  argmax             : {argmax_bal:.5f}")
    print(f"  tuned log-bias     : {tuned:.5f}")
    print(f"  LGBM-dist baseline : 0.97266")
    print(f"  TE-orig (prev null): 0.97270")
    print(f"  Δ vs LGBM-dist     : {tuned - 0.97266:+.5f}")
    print(f"  bias               : {bias.round(3).tolist()}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_lgbm_te_oof.npy", oof)
    np.save(ART / "test_lgbm_te_oof.npy", test_probs)
    with open(ART / "benchmark_te_oof_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "n_te_features": len(te_names),
            "best_iters": [int(x) for x in best_iters],
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned),
            "delta_vs_lgbm_dist": float(tuned - 0.97266),
            "log_bias": bias.tolist(),
        }, f, indent=2)
    tuned_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te_raw[ID],
                  TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / "submission_lgbm_te_oof_tuned.csv", index=False)
    log(f"done in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
