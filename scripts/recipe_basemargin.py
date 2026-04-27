"""Phase B — base-margin residualization on top of recipe FE.

Mechanism: pass `base_margin = K * one_hot(rule_pred) - K/2` per row to
xgb.train via DMatrix. XGBoost adds base_margin to each tree's logit before
applying softmax — so trees start from "rule_pred says class C with K-margin
confidence" and only learn residual corrections.

This is structurally novel: the closed-form 6-feature DGP rule becomes an
explicit prior, freeing tree capacity for boundary-flip residuals. Distinct
from every prior tree experiment which started boosting from logit=0.

K_MARGIN env var (default 4.0) controls prior strength. K=4 → softmax of
[+4, -2, -2] = [≈0.984, ≈0.008, ≈0.008] before any tree contribution; trees
need to overcome ~6 logit-units to flip a prediction. K=2 ≈ [0.79, 0.10, 0.10],
K=6 ≈ [0.998, ≈0.001, ≈0.001]. Sweet spot likely 2-4.

Output paths suffixed _basemargin_K{val}. SMOKE=1 → 20k train, 2 folds.
RUN_FOLD=N → fold-N-only for rehydrate-resilient sequencing.
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
from recipe_full_te import (  # noqa: E402
    CLS_MAP, IDX2CLS, TARGET, load_and_engineer,
)
from recipe_ote import OrderedTE  # noqa: E402
from residual_te_helpers import compute_rule_pred_score  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2
RUN_FOLD = int(os.environ.get("RUN_FOLD", "0"))
K_MARGIN = float(os.environ.get("K_MARGIN", "4.0"))

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)
SUFFIX = f"_basemargin_K{K_MARGIN:g}".replace(".", "")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_base_margin(rule_pred: np.ndarray, K: float) -> np.ndarray:
    """Return (n, 3) base-margin matrix. rule_pred class gets +K, others get -K/2.
    Logit sum = +K - K = 0 so softmax stays normalized; the predicted class has
    a +1.5K logit advantage over each other class.
    """
    n = rule_pred.shape[0]
    bm = np.full((n, 3), -K / 2.0, dtype=np.float32)
    bm[np.arange(n), rule_pred.astype(np.int64)] = K
    return bm


def main() -> None:
    log(f"Phase B base-margin  smoke={SMOKE}  K={K_MARGIN}  run_fold={RUN_FOLD or 'all'}")
    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()

    # Compute rule_pred for train + test from raw CSVs (recipe factorized cats).
    raw_train = pd.read_csv("data/train.csv")
    raw_test = pd.read_csv("data/test.csv")
    if SMOKE:
        raw_train = raw_train.sample(20_000, random_state=SEED).reset_index(drop=True)
        raw_test = raw_test.sample(10_000, random_state=SEED).reset_index(drop=True)
    assert len(raw_train) == len(train), (len(raw_train), len(train))
    assert len(raw_test) == len(test), (len(raw_test), len(test))
    _, rule_train = compute_rule_pred_score(raw_train)
    _, rule_test = compute_rule_pred_score(raw_test)
    bm_train = make_base_margin(rule_train, K_MARGIN)
    bm_test = make_base_margin(rule_test, K_MARGIN)
    log(f"  base-margin computed: train rule prior "
        f"L={(rule_train==0).mean():.3f} M={(rule_train==1).mean():.3f} "
        f"H={(rule_train==2).mean():.3f}")
    rule_acc = float((rule_train == y).mean())
    log(f"  rule baseline raw acc on train = {rule_acc:.5f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores: list[float] = []

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    xgb_params = dict(
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", num_class=3,
        tree_method="hist", eval_metric="mlogloss",
        nthread=-1, seed=SEED, verbosity=0,
    )
    n_rounds = 300 if SMOKE else 3000
    es_rounds = 50 if SMOKE else 200

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if RUN_FOLD and fold != RUN_FOLD:
            continue
        log(f"=== fold {fold}/{N_FOLDS} ===")
        ck_oof = ART / f"oof_recipe_full_te{SUFFIX}_fold{fold}.npy"
        ck_te = ART / f"test_recipe_full_te{SUFFIX}_fold{fold}.npy"
        if ck_oof.exists() and ck_te.exists():
            vp = np.load(ck_oof); tp = np.load(ck_te)
            if vp.shape[0] == len(va_idx) and tp.shape[0] == len(test):
                oof[va_idx] = vp; test_pred += tp / N_FOLDS
                bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
                fold_scores.append(bal)
                log(f"  fold {fold} CACHED bal={bal:.5f}")
                continue
            log(f"  fold {fold} checkpoint shape mismatch; re-running")
            ck_oof.unlink(); ck_te.unlink()

        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        ote = OrderedTE(a=1.0)
        X_tr_shuf = ote.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = ote.transform(X_va); X_te = ote.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + ote.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx])

        dtr = xgb.DMatrix(X_tr[feat_cols].values, label=y[tr_idx],
                          weight=sw, base_margin=bm_train[tr_idx].ravel(),
                          feature_names=feat_cols)
        dva = xgb.DMatrix(X_va[feat_cols].values, label=y[va_idx],
                          base_margin=bm_train[va_idx].ravel(),
                          feature_names=feat_cols)
        dte = xgb.DMatrix(X_te[feat_cols].values,
                          base_margin=bm_test.ravel(),
                          feature_names=feat_cols)

        log(f"  training xgb.train on {len(feat_cols)} feats, {len(X_tr):,} rows")
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=n_rounds,
            evals=[(dva, "val")], early_stopping_rounds=es_rounds,
            verbose_eval=500,
        )
        # multi:softprob with base_margin: predict_proba already includes base.
        vp = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1)).astype(np.float32)
        tp = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)).astype(np.float32)
        # Reshape: predict returns (n, num_class) for softprob.
        vp = vp.reshape(-1, 3); tp = tp.reshape(-1, 3)

        np.save(ck_oof, vp); np.save(ck_te, tp)
        oof[va_idx] = vp; test_pred += tp / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} bal={bal:.5f} best_iter={booster.best_iteration}")

    if RUN_FOLD:
        log(f"RUN_FOLD={RUN_FOLD} — partial run, exiting")
        return

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"OOF argmax={overall:.5f} tuned={tuned:.5f} bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_recipe_full_te{SUFFIX}.npy", oof)
    np.save(ART / f"test_recipe_full_te{SUFFIX}.npy", test_pred)
    eps = 1e-9
    test_idx = (np.log(np.clip(test_pred, eps, 1.0)) + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_idx]})
    sub_path = SUB / f"submission_recipe_full_te{SUFFIX}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    summary = dict(
        n_folds=N_FOLDS, smoke=SMOKE, k_margin=K_MARGIN,
        rule_acc_train=rule_acc,
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
    )
    with open(ART / f"recipe_full_te{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote results JSON")


if __name__ == "__main__":
    main()
