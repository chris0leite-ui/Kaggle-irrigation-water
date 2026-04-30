"""C — ordinal cumulative-link XGB on recipe FE (15th bank component candidate).

Distinct from 14 existing bank components: all 14 use multinomial 3-way
softmax. Cumulative-link reformulates as two binary tasks:
  clf1: P(y > L) = P(y in {M, H})    binary, prevalence ~41%
  clf2: P(y > M) = P(y == H)         binary, prevalence ~3.3%
Reconstruct: P(L) = 1 - P(>L); P(M) = P(>L) - P(>M); P(H) = P(>M)
Floor at 0 and renormalize.

Smoke mode (SMOKE=1): 1 fold, 200 rounds per head.
Production: 5 folds, 3000 rounds with early-stopping(200).
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
from _recipe_helpers import build_fe, ART, DATA, TARGET, SEED, N_FOLDS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] C: {m}", flush=True)


def reconstruct_3class(p_gt_L: np.ndarray, p_gt_M: np.ndarray) -> np.ndarray:
    p_L = 1.0 - p_gt_L
    p_M = p_gt_L - p_gt_M
    p_H = p_gt_M
    p = np.stack([p_L, p_M, p_H], axis=1)
    p = np.clip(p, 1e-6, None)
    p = p / p.sum(1, keepdims=True)
    return p.astype(np.float32)


def fit_binary(X_tr, y_bin_tr, X_va, y_bin_va, X_te, n_rounds, seed):
    sw = compute_sample_weight("balanced", y_bin_tr).astype(np.float32)
    dtr = xgb.DMatrix(X_tr, label=y_bin_tr, weight=sw)
    dva = xgb.DMatrix(X_va, label=y_bin_va)
    dte = xgb.DMatrix(X_te)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 5.0,
        "reg_lambda": 5.0,
        "seed": seed,
        "nthread": -1,
        "verbosity": 0,
    }
    booster = xgb.train(
        params, dtr, num_boost_round=n_rounds,
        evals=[(dva, "val")],
        early_stopping_rounds=200 if not SMOKE else 50,
        verbose_eval=False,
    )
    bi = booster.best_iteration + 1
    pv = booster.predict(dva, iteration_range=(0, bi))
    pt = booster.predict(dte, iteration_range=(0, bi))
    return pv, pt, bi


def main():
    t0 = time.time()
    log(f"loading recipe FE  SMOKE={SMOKE}")
    train, test, info, te_keys, static_cols = build_fe()
    y = train[TARGET].to_numpy().astype(np.int32)
    log(f"  train {train.shape} test {test.shape}  L={(y==0).sum()} M={(y==1).sum()} H={(y==2).sum()}")

    base_cats = info.get("cats", [])
    combos = info.get("combos", [])
    cat_feat_names = [c for c in (base_cats + combos) if c in train.columns]
    log(f"  native cat features: {len(cat_feat_names)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_p_gtL = []
    test_p_gtM = []
    metas = []

    n_rounds = 200 if SMOKE else 3000
    folds_to_do = 1 if SMOKE else N_FOLDS

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        if fold >= folds_to_do:
            break
        t1 = time.time()
        log(f"  fold {fold+1}/{folds_to_do} OTE fit")
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
        seen = set()
        feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]

        X_tr = tr_with_te[feat_cols].astype(np.float32).to_numpy()
        X_va = va_with_te[feat_cols].astype(np.float32).to_numpy()
        X_te = te_with_te[feat_cols].astype(np.float32).to_numpy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        log(f"    X_tr {X_tr.shape}  feat_dim={X_tr.shape[1]}")

        # Head 1: y > L  i.e.  y in {M, H}
        log(f"    fitting head1 (y>L)  prev={(y_tr>=1).mean():.4f}")
        pv1, pt1, bi1 = fit_binary(X_tr, (y_tr >= 1).astype(np.int8),
                                   X_va, (y_va >= 1).astype(np.int8),
                                   X_te, n_rounds, SEED + 100 + fold)
        # Head 2: y > M  i.e.  y == H
        log(f"    fitting head2 (y>M)  prev={(y_tr==2).mean():.4f}")
        pv2, pt2, bi2 = fit_binary(X_tr, (y_tr == 2).astype(np.int8),
                                   X_va, (y_va == 2).astype(np.int8),
                                   X_te, n_rounds, SEED + 200 + fold)

        # Enforce monotonicity: P(y>M) <= P(y>L)
        pv2 = np.minimum(pv2, pv1)
        pt2 = np.minimum(pt2, pt1)

        oof[va_idx] = reconstruct_3class(pv1, pv2)
        test_p_gtL.append(pt1)
        test_p_gtM.append(pt2)
        metas.append({"fold": fold, "bi1": bi1, "bi2": bi2,
                      "fold_argmax_bal": float(balanced_accuracy_score(y_va, oof[va_idx].argmax(1)))})
        log(f"    fold done bi1={bi1} bi2={bi2} fold_argmax_bal={metas[-1]['fold_argmax_bal']:.5f}  ({time.time()-t1:.1f}s)")

    if SMOKE:
        # For smoke, we only have 1 fold; compute partial macro
        smoke_va_macro = metas[0]["fold_argmax_bal"]
        log(f"\nSMOKE fold-1 val macro: {smoke_va_macro:.5f}")
        out = ART / "C_ordinal_xgb_smoke_results.json"
        out.write_text(json.dumps({"smoke": True, "metas": metas,
                                   "smoke_fold_macro": smoke_va_macro,
                                   "wall_seconds": time.time() - t0}, indent=2))
        log(f"wrote {out}")
        return

    test_p_gtL_mean = np.stack(test_p_gtL, 0).mean(0)
    test_p_gtM_mean = np.stack(test_p_gtM, 0).mean(0)
    test_p_gtM_mean = np.minimum(test_p_gtM_mean, test_p_gtL_mean)
    test_probs = reconstruct_3class(test_p_gtL_mean, test_p_gtM_mean)

    np.save(ART / "oof_C_ordinal_xgb.npy", oof)
    np.save(ART / "test_C_ordinal_xgb.npy", test_probs)

    full_macro = balanced_accuracy_score(y, oof.argmax(1))
    log(f"\nFULL OOF argmax balanced-acc: {full_macro:.6f}")
    out = ART / "C_ordinal_xgb_results.json"
    out.write_text(json.dumps({"smoke": False, "metas": metas,
                               "full_oof_macro_argmax": full_macro,
                               "wall_seconds": time.time() - t0}, indent=2))
    log(f"wrote artifacts + {out}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
