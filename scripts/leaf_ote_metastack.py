"""Tree-leaf OTE meta-stacker — PRODUCTION (lever J1).

Train a base XGB on raw factorized features, extract per-tree leaf
indices for every row, OTE-encode each tree's leaf as a 3-class
target-smoothed cat, train a meta XGB on the resulting OTE matrix.
The meta sees TREE-SPACE (per-tree partition memberships) instead of
PROB-SPACE — structurally orthogonal to the 63-component meta-stacker
bank that produced LB 0.98094.

5-fold StratifiedKFold(seed=42) aligned with every other OOF on disk.
Outputs:
  scripts/artifacts/oof_leaf_ote_meta.npy
  scripts/artifacts/test_leaf_ote_meta.npy
  scripts/artifacts/leaf_ote_meta_results.json

Wall safety: aborts if fold 1 exceeds 18 min (caps total wall ~90 min).
"""
from __future__ import annotations

import json
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
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SUFFIX = "_smoke" if SMOKE else ""

BASE_NEST = 50  # ~150 trees total under multi:softprob (3 cls × 50 rounds)
MAX_FOLD_WALL_S = 18 * 60


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
    for c in cats:
        combined = pd.concat([train[c], test[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train)
        train[c] = codes[:s].astype(np.int32)
        test[c] = codes[s:].astype(np.int32)
    log(f"loaded: train={len(train)} test={len(test)} feats={len(test.columns)}")
    return train, test


def main() -> None:
    log(f"SMOKE={SMOKE}  N_FOLDS={N_FOLDS}  BASE_NEST={BASE_NEST}")
    train, test = load()
    y = train[TARGET].to_numpy()
    feat = list(test.columns)
    Xtr_full = train[feat].to_numpy(dtype=np.float32)
    Xte_full = test[feat].to_numpy(dtype=np.float32)

    base_params = dict(
        objective="multi:softprob", num_class=3, max_depth=4, max_leaves=16,
        learning_rate=0.1, reg_alpha=1.0, reg_lambda=1.0,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        verbosity=0, seed=SEED, n_estimators=BASE_NEST,
    )
    meta_params = dict(
        objective="multi:softprob", num_class=3, max_depth=4, max_leaves=30,
        learning_rate=0.1, reg_alpha=5.0, reg_lambda=5.0,
        subsample=0.9, colsample_bytree=0.7, tree_method="hist",
        verbosity=0, seed=SEED, n_estimators=400,
    )

    base_oof = np.zeros((len(train), 3), dtype=np.float32)
    meta_oof = np.zeros_like(base_oof)
    base_test = np.zeros((len(test), 3), dtype=np.float32)
    meta_test = np.zeros_like(base_test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    t_total = time.time()
    for fold, (tr, va) in enumerate(skf.split(Xtr_full, y), 1):
        t_fold = time.time()
        log(f"=== fold {fold}/{N_FOLDS} ===")
        Xtr, Xva = Xtr_full[tr], Xtr_full[va]
        ytr = y[tr]

        base = xgb.XGBClassifier(**base_params)
        t0 = time.time()
        base.fit(Xtr, ytr)
        log(f"  base fit {time.time()-t0:.1f}s")

        booster = base.get_booster()
        t0 = time.time()
        leaves_tr = booster.predict(xgb.DMatrix(Xtr), pred_leaf=True).astype(np.int32)
        leaves_va = booster.predict(xgb.DMatrix(Xva), pred_leaf=True).astype(np.int32)
        leaves_te = booster.predict(xgb.DMatrix(Xte_full), pred_leaf=True).astype(np.int32)
        n_trees = leaves_tr.shape[1]
        log(f"  leaf extract {time.time()-t0:.1f}s  n_trees={n_trees}")

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
        log(f"  OTE fit+transform {time.time()-t0:.1f}s  te_cols={len(ote.te_col_names())}")
        te_cols = ote.te_col_names()

        Xtr_meta = df_tr_te[te_cols].to_numpy(dtype=np.float32)
        Xva_meta = df_va_te[te_cols].to_numpy(dtype=np.float32)
        Xte_meta = df_te_te[te_cols].to_numpy(dtype=np.float32)
        ytr_shuf = df_tr_te[TARGET].to_numpy()

        meta = xgb.XGBClassifier(**meta_params)
        t0 = time.time()
        meta.fit(Xtr_meta, ytr_shuf)
        log(f"  meta fit {time.time()-t0:.1f}s")

        base_oof[va] = base.predict_proba(Xva)
        meta_oof[va] = meta.predict_proba(Xva_meta)
        base_test += base.predict_proba(Xte_full) / N_FOLDS
        meta_test += meta.predict_proba(Xte_meta) / N_FOLDS

        fold_wall = time.time() - t_fold
        log(f"  fold {fold} wall {fold_wall:.1f}s")
        if fold == 1 and fold_wall > MAX_FOLD_WALL_S:
            log(f"  ABORT: fold 1 exceeded {MAX_FOLD_WALL_S}s budget — exiting.")
            sys.exit(2)

    prior = np.bincount(y, minlength=3) / len(y)
    base_argmax = balanced_accuracy_score(y, base_oof.argmax(1))
    meta_argmax = balanced_accuracy_score(y, meta_oof.argmax(1))
    base_bias, base_tuned = tune_log_bias(base_oof, y, prior)
    meta_bias, meta_tuned = tune_log_bias(meta_oof, y, prior)
    log("================ summary ================")
    log(f"  base argmax={base_argmax:.5f}  tuned={base_tuned:.5f}  bias={base_bias.tolist()}")
    log(f"  meta argmax={meta_argmax:.5f}  tuned={meta_tuned:.5f}  bias={meta_bias.tolist()}")
    log(f"  total wall {time.time()-t_total:.1f}s")

    np.save(ART / f"oof_leaf_ote_meta{SUFFIX}.npy", meta_oof)
    np.save(ART / f"test_leaf_ote_meta{SUFFIX}.npy", meta_test)
    np.save(ART / f"oof_leaf_ote_base{SUFFIX}.npy", base_oof)
    np.save(ART / f"test_leaf_ote_base{SUFFIX}.npy", base_test)
    results = dict(
        smoke=SMOKE, n_folds=N_FOLDS, base_n_est=BASE_NEST, n_trees=int(n_trees),
        base_argmax=float(base_argmax), base_tuned=float(base_tuned),
        meta_argmax=float(meta_argmax), meta_tuned=float(meta_tuned),
        base_bias=base_bias.tolist(), meta_bias=meta_bias.tolist(),
        wall_s=float(time.time() - t_total),
    )
    (ART / f"leaf_ote_meta{SUFFIX}_results.json").write_text(json.dumps(results, indent=2))
    log(f"saved oof/test/results to {ART}/")


if __name__ == "__main__":
    main()
