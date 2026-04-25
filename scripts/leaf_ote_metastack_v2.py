"""J1-v2: tree-leaf OTE meta-stacker with RECIPE-FEATURE BASE.

J1-v1 (19-raw-feature base) PROVED orthogonality (Jaccard 0.56 vs LB-best
4-stack — lowest ever) but FAILED magnitude gate: standalone tuned 0.96564
vs anchor 0.98084. Errs +15%, High recall crashed -0.045.

J1-v2 hypothesis: a stronger base (recipe FE + per-fold OrderedTE,
matching recipe_full_te's own pipeline) lifts standalone close enough
to the anchor that the orthogonal signal can compensate for residual
magnitude. Risk: recipe-base trees may produce leaves too similar to
recipe_full_te's tree partitions (already in the meta-stacker bank),
making the leaf-OTE encoding redundant.

Pipeline:
  - Load recipe FE via recipe_full_te.load_and_engineer() (~92 numerics +
    cats/combos/digits/num_as_cat/tres for OTE source).
  - Per fold: fit OrderedTE on cats → train base XGB on 443-feat matrix
    (n_est=80 cap, max_leaves=30, depth=4). Extract leaf indices.
  - OrderedTE the leaves (240 leaves × 3 cls = 720 OTE features per row).
  - Train meta XGB on the leaf-OTE matrix.

Wall budget: ~40-50 min CPU. Aborts if fold 1 exceeds 12 min.
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
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
TARGET = "Irrigation_Need"
ART = Path("scripts/artifacts")
SUFFIX = "_v2_smoke" if SMOKE else "_v2"
MAX_FOLD_WALL_S = (3 if SMOKE else 12) * 60
BASE_NEST = 80  # ~240 trees under multi:softprob
OTE_ALPHA = 1.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"SMOKE={SMOKE}  N_FOLDS={N_FOLDS}  BASE_NEST={BASE_NEST}")
    train, test, info, _ = load_and_engineer()
    y = train[TARGET].to_numpy()
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    log(f"feature inventory: numeric={len(numeric_feats)}  te_cols={len(info['te_cols'])}")

    base_params = dict(
        objective="multi:softprob", num_class=3, max_depth=4, max_leaves=30,
        learning_rate=0.1, reg_alpha=5.0, reg_lambda=5.0,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        max_bin=1024, tree_method="hist",
        verbosity=0, seed=SEED, n_estimators=BASE_NEST,
    )
    meta_params = dict(
        objective="multi:softprob", num_class=3, max_depth=4, max_leaves=30,
        learning_rate=0.1, reg_alpha=5.0, reg_lambda=5.0,
        subsample=0.8, colsample_bytree=0.6, tree_method="hist",
        verbosity=0, seed=SEED, n_estimators=400,
    )

    base_oof = np.zeros((len(train), 3), dtype=np.float32)
    meta_oof = np.zeros_like(base_oof)
    base_test = np.zeros((len(test), 3), dtype=np.float32)
    meta_test = np.zeros_like(base_test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    t_total = time.time()
    for fold, (tr, va) in enumerate(skf.split(train, y), 1):
        t_fold = time.time()
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr].copy().reset_index(drop=True)
        X_va = train.iloc[va].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE on raw cats")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te_cat = OrderedTE(a=OTE_ALPHA)
        X_tr_shuf = te_cat.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te_cat.transform(X_va)
        X_te = te_cat.transform(X_te)
        log(f"    cat-OTE done in {time.time()-t0:.1f}s ({len(te_cat.te_col_names())} TE cols)")

        feat_cols = numeric_feats + te_cat.te_col_names()
        Xtr_arr = X_tr[feat_cols].to_numpy(dtype=np.float32)
        Xva_arr = X_va[feat_cols].to_numpy(dtype=np.float32)
        Xte_arr = X_te[feat_cols].to_numpy(dtype=np.float32)
        ytr = y[tr]
        sw = compute_sample_weight("balanced", ytr)

        log(f"  training base XGB on {len(feat_cols)} features  {len(Xtr_arr):,} rows")
        t0 = time.time()
        base = xgb.XGBClassifier(**base_params)
        base.fit(Xtr_arr, ytr, sample_weight=sw)
        log(f"    base fit {time.time()-t0:.1f}s")

        booster = base.get_booster()
        t0 = time.time()
        leaves_tr = booster.predict(xgb.DMatrix(Xtr_arr), pred_leaf=True).astype(np.int32)
        leaves_va = booster.predict(xgb.DMatrix(Xva_arr), pred_leaf=True).astype(np.int32)
        leaves_te = booster.predict(xgb.DMatrix(Xte_arr), pred_leaf=True).astype(np.int32)
        n_trees = leaves_tr.shape[1]
        log(f"    leaf extract {time.time()-t0:.1f}s  n_trees={n_trees}")

        leaf_cols = [f"L{j}" for j in range(n_trees)]
        df_tr = pd.DataFrame(leaves_tr, columns=leaf_cols)
        df_tr[TARGET] = ytr
        df_va = pd.DataFrame(leaves_va, columns=leaf_cols)
        df_te = pd.DataFrame(leaves_te, columns=leaf_cols)

        log("  fitting leaf-OTE")
        t0 = time.time()
        rng2 = np.random.default_rng(SEED + 100 + fold)
        perm2 = rng2.permutation(len(df_tr))
        df_tr_shuf = df_tr.iloc[perm2].reset_index(drop=True)
        leaf_ote = OrderedTE(a=OTE_ALPHA)
        df_tr_te = leaf_ote.fit(df_tr_shuf, leaf_cols, TARGET)
        df_va_te = leaf_ote.transform(df_va)
        df_te_te = leaf_ote.transform(df_te)
        leaf_te_cols = leaf_ote.te_col_names()
        log(f"    leaf-OTE done in {time.time()-t0:.1f}s ({len(leaf_te_cols)} TE cols)")

        Xtr_meta = df_tr_te[leaf_te_cols].to_numpy(dtype=np.float32)
        Xva_meta = df_va_te[leaf_te_cols].to_numpy(dtype=np.float32)
        Xte_meta = df_te_te[leaf_te_cols].to_numpy(dtype=np.float32)
        ytr_shuf = df_tr_te[TARGET].to_numpy()
        sw_shuf = compute_sample_weight("balanced", ytr_shuf)

        log(f"  training meta XGB on {len(leaf_te_cols)} leaf-OTE features")
        t0 = time.time()
        meta = xgb.XGBClassifier(**meta_params)
        meta.fit(Xtr_meta, ytr_shuf, sample_weight=sw_shuf)
        log(f"    meta fit {time.time()-t0:.1f}s")

        base_oof[va] = base.predict_proba(Xva_arr)
        meta_oof[va] = meta.predict_proba(Xva_meta)
        base_test += base.predict_proba(Xte_arr) / N_FOLDS
        meta_test += meta.predict_proba(Xte_meta) / N_FOLDS

        fold_wall = time.time() - t_fold
        log(f"  fold {fold} wall {fold_wall:.1f}s  cumulative {time.time()-t_total:.1f}s")
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
        n_features_base=len(feat_cols), n_features_meta=len(leaf_te_cols),
        base_argmax=float(base_argmax), base_tuned=float(base_tuned),
        meta_argmax=float(meta_argmax), meta_tuned=float(meta_tuned),
        base_bias=base_bias.tolist(), meta_bias=meta_bias.tolist(),
        wall_s=float(time.time() - t_total),
    )
    (ART / f"leaf_ote_meta{SUFFIX}_results.json").write_text(json.dumps(results, indent=2))
    log(f"saved oof/test/results to {ART}/")


if __name__ == "__main__":
    main()
