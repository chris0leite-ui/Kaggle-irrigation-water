"""Tree-leaf OTE meta-stacker — SMOKE.

Lever J1: train a base XGB, extract per-tree leaf indices for every row,
treat each tree's leaf as a high-card categorical, OrderedTE-encode
(3 classes), feed the resulting OTE features to a meta XGB. The meta
sees TREE-SPACE (per-tree partition memberships) instead of PROB-SPACE.

Smoke goal: validate the chain end-to-end on a 20k subsample / 2 folds.
Pass criteria:
  - Pipeline completes without error.
  - Meta tuned bal_acc > base tuned bal_acc by any margin (signal in leaves).
  - Meta-vs-base error Jaccard < 0.97 (meta is not a relabeled base).
If pass: scale to full 5-fold w/ recipe FE in a separate script.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
SMOKE = os.environ.get("SMOKE", "1") == "1"
N_FOLDS = 2 if SMOKE else 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train[TARGET] = train[TARGET].map(CLS_MAP).astype(np.int32)
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  loaded: train={len(train)} test={len(test)} nums={len(nums)} cats={len(cats)}")
    for c in cats:
        combined = pd.concat([train[c], test[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train)
        train[c] = codes[:s].astype(np.int32)
        test[c] = codes[s:].astype(np.int32)
    return train, test


def extract_leaves(model: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Return shape (n, total_trees) int leaf indices. For multi:softprob,
    total_trees = n_estimators * num_class (one tree per class per round)."""
    dm = xgb.DMatrix(X)
    leaves = model.predict(dm, pred_leaf=True)
    return leaves.astype(np.int32)


def main() -> None:
    log(f"SMOKE={SMOKE} N_FOLDS={N_FOLDS}")
    train, test = load()
    y = train[TARGET].to_numpy()
    feat = [c for c in test.columns]
    Xtr_full = train[feat].to_numpy(dtype=np.float32)
    Xte_full = test[feat].to_numpy(dtype=np.float32)

    base_params = dict(
        objective="multi:softprob", num_class=3, max_depth=4, max_leaves=16,
        learning_rate=0.1, reg_alpha=1.0, reg_lambda=1.0,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        verbosity=0, seed=SEED, n_estimators=60,
    )
    # XGB with multi:softprob fits ONE tree PER CLASS per boosting round, so
    # actual n_trees = n_estimators * num_class. We discover this via the
    # leaf array's column count after the first extraction.

    base_oof = np.zeros((len(train), 3), dtype=np.float32)
    meta_oof = np.zeros_like(base_oof)
    base_test = np.zeros((len(test), 3), dtype=np.float32)
    meta_test = np.zeros_like(base_test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(Xtr_full, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        Xtr, Xva = Xtr_full[tr], Xtr_full[va]
        ytr = y[tr]

        base = xgb.XGBClassifier(**base_params)
        t0 = time.time()
        base.fit(Xtr, ytr)
        log(f"  base fit {time.time()-t0:.1f}s")

        booster = base.get_booster()
        leaves_tr = extract_leaves(booster, Xtr)
        leaves_va = extract_leaves(booster, Xva)
        leaves_te = extract_leaves(booster, Xte_full)
        n_trees = leaves_tr.shape[1]
        log(f"  leaves: tr={leaves_tr.shape}  va={leaves_va.shape}  te={leaves_te.shape}  "
            f"n_trees={n_trees}")

        leaf_cols = [f"L{j}" for j in range(n_trees)]
        df_tr = pd.DataFrame(leaves_tr, columns=leaf_cols)
        df_tr[TARGET] = ytr
        df_va = pd.DataFrame(leaves_va, columns=leaf_cols)
        df_te = pd.DataFrame(leaves_te, columns=leaf_cols)

        rng = np.random.default_rng(SEED + fold)
        df_tr_shuf = df_tr.iloc[rng.permutation(len(df_tr))].reset_index(drop=True)
        ote = OrderedTE(a=1.0)
        t0 = time.time()
        df_tr_te = ote.fit(df_tr_shuf, leaf_cols, TARGET)
        df_va_te = ote.transform(df_va)
        df_te_te = ote.transform(df_te)
        log(f"  OTE on {n_trees} leaves done in {time.time()-t0:.1f}s")
        te_cols = ote.te_col_names()

        Xtr_meta = df_tr_te[te_cols].to_numpy(dtype=np.float32)
        Xva_meta = df_va_te[te_cols].to_numpy(dtype=np.float32)
        Xte_meta = df_te_te[te_cols].to_numpy(dtype=np.float32)
        ytr_shuf = df_tr_te[TARGET].to_numpy()

        meta_params = dict(
            objective="multi:softprob", num_class=3, max_depth=4, max_leaves=30,
            learning_rate=0.1, reg_alpha=5.0, reg_lambda=5.0,
            subsample=0.9, colsample_bytree=0.7, tree_method="hist",
            verbosity=0, seed=SEED, n_estimators=200,
        )
        meta = xgb.XGBClassifier(**meta_params)
        t0 = time.time()
        meta.fit(Xtr_meta, ytr_shuf)
        log(f"  meta fit {time.time()-t0:.1f}s")

        base_oof[va] = base.predict_proba(Xva)
        meta_oof[va] = meta.predict_proba(Xva_meta)
        base_test += base.predict_proba(Xte_full) / N_FOLDS
        meta_test += meta.predict_proba(Xte_meta) / N_FOLDS

    if SMOKE and N_FOLDS == 2:
        mask = base_oof.sum(axis=1) > 0
        log(f"  evaluating on {mask.sum()}/{len(y)} held-out rows (full coverage)")

    prior = np.bincount(y, minlength=3) / len(y)
    base_argmax = balanced_accuracy_score(y, base_oof.argmax(1))
    meta_argmax = balanced_accuracy_score(y, meta_oof.argmax(1))
    base_bias, base_tuned = tune_log_bias(base_oof, y, prior, coarse=True)
    meta_bias, meta_tuned = tune_log_bias(meta_oof, y, prior, coarse=True)
    base_pred = (np.log(np.clip(base_oof, 1e-9, 1)) + base_bias).argmax(1)
    meta_pred = (np.log(np.clip(meta_oof, 1e-9, 1)) + meta_bias).argmax(1)
    base_err = base_pred != y
    meta_err = meta_pred != y
    inter = (base_err & meta_err).sum()
    union = (base_err | meta_err).sum()
    jaccard = inter / max(union, 1)

    log("================ smoke summary ================")
    log(f"  base argmax = {base_argmax:.5f}   tuned = {base_tuned:.5f}   "
        f"errs = {int(base_err.sum())}")
    log(f"  meta argmax = {meta_argmax:.5f}   tuned = {meta_tuned:.5f}   "
        f"errs = {int(meta_err.sum())}")
    log(f"  meta-vs-base Jaccard = {jaccard:.4f}")
    delta = meta_tuned - base_tuned
    log(f"  Δ tuned (meta - base) = {delta:+.5f}")
    if delta > 0 and jaccard < 0.97:
        log("  GATE PASS — chain works, meta extracts signal beyond base argmax.")
    else:
        log("  GATE FAIL — meta is redundant with base or weaker.")


if __name__ == "__main__":
    main()
