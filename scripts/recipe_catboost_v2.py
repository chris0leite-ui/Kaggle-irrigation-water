"""CatBoost-base v2: distinct config from existing v1 CatBoost components.

Existing v1 CatBoost variants:
  recipe_full_te_catboost: depth=4, l2_leaf_reg=10, iter=2000, Bernoulli bootstrap (CPU)
  catboost_recipe_gpu:     depth=4, GPU, Bayesian bootstrap
  catboost_optuna:         Optuna-tuned

This v2 uses DISTINCTLY DIFFERENT HPs:
  depth=6  (vs all 3 existing at 4)
  ordered boosting (vs Plain in v1)
  l2_leaf_reg=5  (lighter than 10)
  learning_rate=0.03  (slower than 0.05)
  iterations=4000  (longer than 2000)
  bootstrap=Bayesian  (different randomness from Bernoulli)

Should produce structurally different errors via the depth+ordered
combo. CatBoost handles categoricals natively (no factorization).

5-fold StratifiedKFold(seed=42). Per-fold OTE matches recipe pipeline.

Outputs:
  scripts/artifacts/oof_recipe_catboost_v2.npy
  scripts/artifacts/test_recipe_catboost_v2.npy
  scripts/artifacts/recipe_catboost_v2_results.json
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
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).parent))
from _recipe_helpers import build_fe, ART, SUB, DATA, TARGET, SEED, N_FOLDS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402


def log(m): print(f"[{time.strftime('%H:%M:%S')}] CB2: {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    t0 = time.time()
    log("loading + engineering V10 recipe FE")
    train, test, info, te_keys, static_cols = build_fe()
    y = train[TARGET].to_numpy().astype(np.int32)
    log(f"  train {train.shape}  test {test.shape}  te_keys {len(te_keys)}  static {len(static_cols)}")

    # CatBoost native cats
    base_cats = info.get("cats", [])
    combos = info.get("combos", [])
    cat_feat_names = [c for c in (base_cats + combos) if c in train.columns]
    log(f"  CatBoost native cats: {len(cat_feat_names)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_preds = []
    best_iters = []
    fold_argmaxes = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        ckpt_oof  = ART / f"_recipe_cb2_fold{fold}_oof.npy"
        ckpt_te   = ART / f"_recipe_cb2_fold{fold}_test.npy"
        ckpt_meta = ART / f"_recipe_cb2_fold{fold}_meta.json"
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
        tr_df = train.iloc[tr_idx].reset_index(drop=True).copy()
        va_df = train.iloc[va_idx].reset_index(drop=True).copy()
        tr_shuf = tr_df.sample(frac=1.0, random_state=SEED + fold).reset_index(drop=True)
        ote = OrderedTE(a=1.0)
        ote.fit(tr_shuf, te_keys, TARGET)
        tr_with_te = ote.transform(tr_df)
        va_with_te = ote.transform(va_df)
        te_with_te = ote.transform(test)

        ote_cols = ote.te_col_names()
        feat_cols = static_cols + ote_cols + cat_feat_names
        seen = set(); feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]

        X_tr = tr_with_te[feat_cols].astype(np.float32).to_numpy()
        X_va = va_with_te[feat_cols].astype(np.float32).to_numpy()
        X_te = te_with_te[feat_cols].astype(np.float32).to_numpy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        cat_idxs = [feat_cols.index(c) for c in cat_feat_names if c in feat_cols]
        sw_tr = compute_sample_weight("balanced", y_tr).astype(np.float32)

        log(f"    feat_dim={X_tr.shape[1]} cats={len(cat_idxs)}")

        # CatBoost wants string categoricals natively, but we have factorized
        # ints in those columns post-recipe-FE. CatBoost can still treat
        # ints as categorical via cat_features param. Use Pool with cat_features.
        # But the factorized ints range up to ~1e5 unique values for combos —
        # CatBoost handles that fine.
        train_pool = Pool(X_tr, y_tr, cat_features=cat_idxs, weight=sw_tr)
        val_pool = Pool(X_va, y_va, cat_features=cat_idxs)

        # DISTINCT config: depth=6, ordered, lighter reg, slower LR, more iters
        model = CatBoostClassifier(
            iterations=4000,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            boosting_type="Ordered",
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=300,
            thread_count=-1,
            loss_function="MultiClass",
            classes_count=3,
        )
        model.fit(train_pool, eval_set=val_pool)
        bi = int(model.tree_count_)
        best_iters.append(bi)
        vp = model.predict_proba(X_va).astype(np.float32)
        tp = model.predict_proba(X_te).astype(np.float32)
        oof[va_idx] = vp
        test_preds.append(tp)
        argmax_bal = balanced_accuracy_score(y_va, vp.argmax(1))
        fold_argmaxes.append(argmax_bal)
        np.save(ckpt_oof, vp); np.save(ckpt_te, tp)
        ckpt_meta.write_text(json.dumps({"best_iter": int(bi), "argmax_bal": float(argmax_bal)}))
        log(f"    fold {fold+1} val_argmax={argmax_bal:.5f} it={bi} wall={time.time()-t1:.1f}s [ckpt]")

    test_pred = np.mean(test_preds, axis=0).astype(np.float32)
    np.save(ART / "oof_recipe_catboost_v2.npy", oof)
    np.save(ART / "test_recipe_catboost_v2.npy", test_pred)

    oof_argmax = balanced_accuracy_score(y, oof.argmax(1))
    BIAS = np.array([1.4324, 1.4689, 3.4008])
    pred_tuned = (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1)
    tuned_bal = balanced_accuracy_score(y, pred_tuned)
    log(f"\n=== recipe-CatBoost v2 standalone ===")
    log(f"  argmax = {oof_argmax:.5f}")
    log(f"  @recipe-bias = {tuned_bal:.5f}")

    out = dict(
        config="depth=6, ordered, l2=5, lr=0.03, iter=4000, Bayesian",
        oof_argmax=float(oof_argmax),
        oof_tuned_recipe_bias=float(tuned_bal),
        per_fold_argmax=[float(x) for x in fold_argmaxes],
        best_iters=[int(b) for b in best_iters],
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / "recipe_catboost_v2_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {json_path}")


if __name__ == "__main__":
    main()
