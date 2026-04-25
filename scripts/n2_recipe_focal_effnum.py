"""N2: Recipe XGB with class-balanced effective-number focal loss.

Textbook fix to the prior focal nulls (recipe_focal_invfreq.py with α=invfreq
=[1.0, 1.55, 17.6] gave OOF tuned 0.97683, monotone-negative on every blend
gate). Per Cui et al. CVPR 2019:

    α_c ∝ (1 − β) / (1 − β^{n_c})

This is the "effective number of samples" reweighting. At our class counts
(Low=370k / Med=239k / High=21k), β=0.99999 produces α ≈ [1.0, 1.07, 5.15]
— a 3.4× milder rare-class boost than invfreq's 17.6, which is precisely
the "too aggressive on a tuned XGB" failure mode the prior nulls hit.

Plus two corrective levers from the prior-null portmortem:
  - γ=1 (not 2): gentler focal modulation `(1-p)^1` halves the gradient
    starvation on easy boundary rows that γ=2 caused.
  - early_stopping_rounds=400 (not 200): lets the rare-class surface
    self-correct before the majority-class objective declares convergence.

Same FE+OTE+5-fold seed=42 as recipe_full_te.py. Outputs:
    oof_recipe_focal_effnum.npy + test + results.json
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402
from focal_loss_common import (  # noqa: E402
    make_focal_obj, make_hard_val_metric, margin_to_prob,
)
from recipe_full_te import (  # noqa: E402
    CLS_MAP, IDX2CLS, TARGET, load_and_engineer,
)
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

# Hyperparameters tuned to address the prior focal nulls' failure modes.
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "1.0"))
FOCAL_BETA = float(os.environ.get("FOCAL_BETA", "0.99999"))

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def effective_number_alpha(y: np.ndarray, beta: float, n_class: int = 3) -> np.ndarray:
    n = np.bincount(y, minlength=n_class).astype(np.float64)
    eff = (1.0 - beta) / np.clip(1.0 - np.power(beta, n), 1e-30, None)
    alpha = eff / eff[0]  # normalise so alpha_Low = 1
    return alpha.astype(np.float32)


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           alpha: np.ndarray, gamma: float) -> dict:
    y = train[TARGET].to_numpy().astype(np.int32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores: list[float] = []

    base_params = dict(
        max_depth=4, max_leaves=30,
        eta=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        tree_method="hist", num_class=3, verbosity=0,
    )
    num_round = 300 if SMOKE else 4000  # extra headroom past recipe's 3000
    esr = 50 if SMOKE else 400           # 2x recipe's 200, per portmortem

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"  OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        dtrain = xgb.DMatrix(X_tr[feat_cols].to_numpy(dtype=np.float32),
                             label=y_tr.astype(np.float32))
        dval = xgb.DMatrix(X_va[feat_cols].to_numpy(dtype=np.float32),
                           label=y_va.astype(np.float32))
        dtest = xgb.DMatrix(X_te[feat_cols].to_numpy(dtype=np.float32))

        obj = make_focal_obj(y_tr, alpha=alpha, gamma=gamma, n_class=3)
        val_metric = make_hard_val_metric(y_va, n_class=3)

        log(f"  training XGB on {len(feat_cols)} feats  N_tr={len(X_tr):,}  "
            f"γ={gamma}  α={alpha.round(3).tolist()}  esr={esr}")
        t0 = time.time()
        booster = xgb.train(
            base_params, dtrain,
            num_boost_round=num_round,
            obj=obj, custom_metric=val_metric,
            evals=[(dval, "val")], maximize=False,
            early_stopping_rounds=esr,
            verbose_eval=500,
        )
        oof[va_idx] = margin_to_prob(booster.predict(dval, output_margin=True))
        test_pred += margin_to_prob(booster.predict(dtest, output_margin=True)) / N_FOLDS
        fold_bal = fast_bal_acc(y_va, oof[va_idx].argmax(1))
        fold_scores.append(fold_bal)
        log(f"  fold {fold} argmax_bal = {fold_bal:.5f}  "
            f"best_iter={booster.best_iteration}  wall={time.time()-t0:.1f}s")

    overall = fast_bal_acc(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    log(f"N2 effective-number focal: γ={FOCAL_GAMMA}  β={FOCAL_BETA}  smoke={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy().astype(np.int32)
    alpha = effective_number_alpha(y, beta=FOCAL_BETA)
    log(f"  effective-number alpha = {alpha.round(4).tolist()}  "
        f"(invfreq for ref = {(1/(np.bincount(y)/len(y)) / (1/(np.bincount(y)/len(y))[0])).round(2).tolist()})")

    res = run_cv(train, test, info, alpha=alpha, gamma=FOCAL_GAMMA)

    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(res["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_recipe_focal_effnum{suffix}.npy", res["oof"])
    np.save(ART / f"test_recipe_focal_effnum{suffix}.npy", res["test"])

    eps = 1e-9
    test_log = np.log(np.clip(res["test"], eps, 1.0))
    pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in pred_idx]})
    sub_path = SUB / f"submission_recipe_focal_effnum{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE,
        focal_gamma=FOCAL_GAMMA, focal_beta=FOCAL_BETA,
        alpha=alpha.tolist(),
        fold_scores_argmax=[float(s) for s in res["fold_scores"]],
        overall_argmax_bal_acc=res["overall_argmax"],
        tuned_log_bias_bal_acc=tuned, log_bias=bias.tolist(),
        n_features=len(res["feat_cols"]),
    )
    with open(ART / f"recipe_focal_effnum{suffix}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
