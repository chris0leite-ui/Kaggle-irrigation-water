"""LightGBM-base on V10 recipe FE WITH per-fold OTE + native categorical
handling.

Distinct from existing v1 LGBM components:
  - lgbm_te_orig: no OTE, just original-data features
  - lgbm_dist_digits: no OTE, dist+digits set
  - lgbm_dist_digits_ote: small custom OTE on dist+digits
  - recipe_full_te_lgbm: same recipe FE but factorized cats only (no
    native categorical handling)

This variant uses LightGBM's NATIVE categorical handling
(categorical_feature= argument to fit) on the 8 base cat columns +
combos. LightGBM's split-finding for categoricals is fundamentally
different from XGB's (Chen-Guestrin algorithm: groups categories by
class proportion). Should give different errors.

5-fold StratifiedKFold(seed=42) for v1 alignment. Per-fold OTE matches
recipe pipeline. Class-balanced sample weights.

Outputs:
  scripts/artifacts/oof_recipe_lgbm_native.npy
  scripts/artifacts/test_recipe_lgbm_native.npy
  scripts/artifacts/recipe_lgbm_native_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from _recipe_helpers import build_fe, ART, SUB, DATA, TARGET, SEED, N_FOLDS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402


def log(m): print(f"[{time.strftime('%H:%M:%S')}] LGBM: {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    t0 = time.time()
    log("loading + engineering V10 recipe FE")
    train, test, info, te_keys, static_cols = build_fe()
    y = train[TARGET].to_numpy().astype(np.int32)
    log(f"  train {train.shape}  test {test.shape}  te_keys {len(te_keys)}  static {len(static_cols)}")

    # LightGBM native cat handling: 8 base cat columns (in info["cats"])
    # + cat-pair combos (info["combos"]). These are factorized integers
    # in train/test after recipe FE — pass column NAMES to LGBM.
    base_cats = info.get("cats", [])
    combos = info.get("combos", [])
    cat_feat_names = [c for c in (base_cats + combos) if c in train.columns]
    log(f"  native cat features: {len(cat_feat_names)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_preds = []
    best_iters = []
    fold_argmaxes = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        ckpt_oof  = ART / f"_recipe_lgbm_fold{fold}_oof.npy"
        ckpt_te   = ART / f"_recipe_lgbm_fold{fold}_test.npy"
        ckpt_meta = ART / f"_recipe_lgbm_fold{fold}_meta.json"
        if ckpt_oof.exists() and ckpt_te.exists() and ckpt_meta.exists():
            log(f"  fold {fold+1}/{N_FOLDS} resuming from checkpoint")
            vp = np.load(ckpt_oof); tp = np.load(ckpt_te)
            mi = json.loads(ckpt_meta.read_text())
            oof[va_idx] = vp.astype(np.float32)
            test_preds.append(tp)
            best_iters.append(mi["best_iter"])
            fold_argmaxes.append(mi["argmax_bal"])
            log(f"    val_argmax={mi['argmax_bal']:.5f} (cached)")
            continue

        t1 = time.time()
        log(f"  fold {fold+1}/{N_FOLDS} fitting OTE")
        # Per-fold OTE: shuffle tr rows, fit OTE, then transform tr/val/test
        tr_df = train.iloc[tr_idx].reset_index(drop=True).copy()
        va_df = train.iloc[va_idx].reset_index(drop=True).copy()
        tr_shuf = tr_df.sample(frac=1.0, random_state=SEED + fold).reset_index(drop=True)
        ote = OrderedTE(a=1.0)
        tr_with_te = ote.fit(tr_shuf, te_keys, TARGET)
        # Map back from shuffled order to original tr order
        tr_with_te_sorted = tr_with_te.sort_index()  # preserves original tr_df order
        # Actually we need to sort back to tr_df.iloc match — easier path: use ote.transform on tr_df
        tr_with_te = ote.transform(tr_df)
        va_with_te = ote.transform(va_df)
        te_with_te = ote.transform(test)

        # Build feature matrix: static cols + OTE cols + categorical cols (already factorized in train/test)
        ote_cols = ote.te_col_names()
        feat_cols = static_cols + ote_cols + cat_feat_names
        # Deduplicate while preserving order
        seen = set(); feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]

        X_tr = tr_with_te[feat_cols].astype(np.float32).to_numpy()
        X_va = va_with_te[feat_cols].astype(np.float32).to_numpy()
        X_te = te_with_te[feat_cols].astype(np.float32).to_numpy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Categorical features by name → LGBM
        cat_idxs = [feat_cols.index(c) for c in cat_feat_names if c in feat_cols]
        sw_tr = compute_sample_weight("balanced", y_tr).astype(np.float32)

        log(f"    feat_dim={X_tr.shape[1]} cats={len(cat_idxs)}")

        params = dict(
            objective="multiclass", num_class=3, metric="multi_logloss",
            learning_rate=0.05, num_leaves=30, max_depth=4,
            min_child_samples=20, feature_fraction=0.9, bagging_fraction=0.9,
            bagging_freq=5, lambda_l1=5.0, lambda_l2=5.0,
            verbosity=-1, seed=SEED, num_threads=-1,
        )
        model = lgb.LGBMClassifier(**params, n_estimators=3000)
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_idxs,
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        bi = model.best_iteration_ or model.n_estimators
        best_iters.append(bi)
        vp = model.predict_proba(X_va, num_iteration=bi).astype(np.float32)
        tp = model.predict_proba(X_te, num_iteration=bi).astype(np.float32)
        oof[va_idx] = vp
        test_preds.append(tp)
        argmax_bal = balanced_accuracy_score(y_va, vp.argmax(1))
        fold_argmaxes.append(argmax_bal)
        np.save(ckpt_oof, vp); np.save(ckpt_te, tp)
        ckpt_meta.write_text(json.dumps({"best_iter": int(bi), "argmax_bal": float(argmax_bal)}))
        log(f"    fold {fold+1} val_argmax={argmax_bal:.5f} it={bi} wall={time.time()-t1:.1f}s [ckpt]")

    test_pred = np.mean(test_preds, axis=0).astype(np.float32)
    np.save(ART / "oof_recipe_lgbm_native.npy", oof)
    np.save(ART / "test_recipe_lgbm_native.npy", test_pred)

    oof_argmax = balanced_accuracy_score(y, oof.argmax(1))
    BIAS = np.array([1.4324, 1.4689, 3.4008])
    pred_tuned = (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1)
    tuned_bal = balanced_accuracy_score(y, pred_tuned)
    log(f"\n=== recipe-LGBM-native standalone ===")
    log(f"  argmax = {oof_argmax:.5f}")
    log(f"  @recipe-bias = {tuned_bal:.5f}")

    out = dict(
        n_features=int(X_tr.shape[1]),
        cat_features=cat_idxs if cat_idxs else [],
        n_cat_features=len(cat_idxs),
        oof_argmax=float(oof_argmax),
        oof_tuned_recipe_bias=float(tuned_bal),
        per_fold_argmax=[float(x) for x in fold_argmaxes],
        best_iters=[int(b) for b in best_iters],
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / "recipe_lgbm_native_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {json_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
