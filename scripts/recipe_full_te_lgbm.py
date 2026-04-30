"""LightGBM on the full recipe feature set.

Mirrors recipe_full_te.py (~440 features: cat-pair combos + digits +
num-as-cat + threshold flags + LR-formula logits + FREQ + ORIG mean/std
+ OrderedTE on ~117 cats) but swaps XGBClassifier -> LGBMClassifier.

Hypothesis: LGBM's leaf-wise splits produce a different decision surface
than XGB's level-wise on this 500-col feature set. Our earlier LGBM-digits
null was on a 137-feature subset; the recipe's larger OTE/ORIG/FREQ
block may give LGBM structured features XGB doesn't exploit the same way.

HPs chosen to mirror recipe's XGB philosophy:
  num_leaves=16       (≈ max_depth=4 with 2**4=16 leaves)
  min_data_in_leaf=20 (heavier than dist+digits' 200; matches recipe's
                      min_child_weight=2 philosophy × smaller leaves)
  learning_rate=0.1
  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5
  lambda_l1=5, lambda_l2=5   (match recipe alpha/reg_lambda)
  n_estimators=3000, early_stopping=200

Same 5-fold StratifiedKFold(seed=42), same per-fold OrderedTE, same
class-balanced sample weights. Saves distinct artifacts so it coexists
with the XGB recipe.

Wall-time budget: ~30-45 min on CPU for 5 folds.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_features import (  # noqa: E402
    add_cat_pair_combos, add_digit_features, add_freq_features,
    add_lr_formula_logits, add_num_as_cat, add_orig_mean_std,
    add_threshold_flags,
)
from recipe_ote import OrderedTE  # noqa: E402


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

# LGBM_BOOSTING knob: 'gbdt' (default, preserves prior LB-validated artefact),
# 'goss' (gradient-one-sided sampling — different gradient utilization than gbdt),
# 'dart' (tree dropout). Output paths get suffix when non-default.
LGBM_BOOSTING = os.environ.get("LGBM_BOOSTING", "gbdt")
assert LGBM_BOOSTING in ("gbdt", "goss", "dart"), \
    f"LGBM_BOOSTING must be gbdt|goss|dart, got {LGBM_BOOSTING!r}"
SUFFIX = "" if LGBM_BOOSTING == "gbdt" else f"_{LGBM_BOOSTING}"

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_engineer():
    log("loading train / test / orig")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        log("SMOKE=1 — subsampling 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:10_000]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)}  cats={len(cats)}  "
        f"train={len(train)}  test={len(test)}  orig={len(orig)}")

    log("adding threshold flags + LR-formula logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)

    log("adding cat x cat pair combos")
    combos = add_cat_pair_combos(train, test, orig, cats)
    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)
    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)
    log("adding ORIG mean/std per col")
    orig_stats = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: "
        f"cats={len(cats)} combos={len(combos)} digits={len(digits)} "
        f"num_as_cat={len(num_as_cat)} tres={len(tres)} logits={len(logits)} "
        f"freq={len(freq)} orig_stats={len(orig_stats)} "
        f"te_cols={len(info['te_cols'])}")
    return train, test, info, test_ids


def run_cv(train, test, info, a_ote=1.0):
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    best_iters = []

    lgbm_params = dict(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        learning_rate=0.1,
        num_leaves=16,                 # ~ max_depth=4
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=5.0,                 # matches recipe XGB alpha=5
        lambda_l2=5.0,                 # matches recipe XGB reg_lambda=5
        verbose=-1,
        seed=SEED,
    )
    if LGBM_BOOSTING == "goss":
        # GOSS: gradient-one-sided sampling. Replaces bagging entirely;
        # samples by gradient magnitude (top_rate large-grad rows kept,
        # other_rate small-grad rows kept). bagging must be disabled.
        lgbm_params["boosting"] = "goss"
        lgbm_params["top_rate"] = 0.2
        lgbm_params["other_rate"] = 0.1
        lgbm_params.pop("bagging_fraction")
        lgbm_params.pop("bagging_freq")
    elif LGBM_BOOSTING == "dart":
        lgbm_params["boosting"] = "dart"
        lgbm_params["drop_rate"] = 0.1
        lgbm_params["skip_drop"] = 0.5
        # DART doesn't honour early stopping in older LGBMs; keep the
        # callback path but cap rounds at 1500 to bound wall.

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a_ote)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx])

        log(f"  training LGBM on {len(feat_cols)} features")
        t_fit = time.time()
        dtr = lgb.Dataset(X_tr[feat_cols], label=y[tr_idx], weight=sw)
        dva = lgb.Dataset(X_va[feat_cols], label=y[va_idx], reference=dtr)
        num_rounds = 300 if SMOKE else 3000
        stop_rounds = 50 if SMOKE else 200
        booster = lgb.train(
            lgbm_params, dtr, num_boost_round=num_rounds,
            valid_sets=[dva], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=stop_rounds, verbose=False)],
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(X_va[feat_cols], num_iteration=bi).astype(np.float32)
        test_pred += booster.predict(X_te[feat_cols], num_iteration=bi).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold}  best_iter={bi}  argmax={bal:.5f}  "
            f"wall={time.time()-t_fit:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== overall OOF argmax = {overall:.5f}  "
        f"(fold mean {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols,
                best_iters=best_iters)


def main():
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_recipe_full_te_lgbm{SUFFIX}.npy", result["oof"])
    np.save(ART / f"test_recipe_full_te_lgbm{SUFFIX}.npy", result["test"])

    # Jaccard vs recipe XGB. Skip in SMOKE (row-count mismatch).
    recipe_res = json.loads((ART/'recipe_full_te_results.json').read_text())
    jacc = None
    lgbm_err_count = None
    rec_err_count = None
    if not SMOKE:
        recipe_oof = np.load(ART / "oof_recipe_full_te.npy")
        recipe_bias = np.array(recipe_res['log_bias'])
        rec_pred = (np.log(np.clip(recipe_oof, 1e-9, 1.0)) + recipe_bias).argmax(1)
        lgbm_pred = (np.log(np.clip(result["oof"], 1e-9, 1.0)) + bias).argmax(1)
        rec_err = rec_pred != y
        lgbm_err = lgbm_pred != y
        jacc = float((rec_err & lgbm_err).sum() / max(1, (rec_err | lgbm_err).sum()))
        lgbm_err_count = int(lgbm_err.sum())
        rec_err_count = int(rec_err.sum())
        log(f"errors:  lgbm={lgbm_err_count}  recipe_xgb={rec_err_count}")
        log(f"Jaccard vs recipe = {jacc:.4f}")

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub.to_csv(SUB / f"submission_recipe_full_te_lgbm{SUFFIX}.csv", index=False)
    log(f"wrote submission_recipe_full_te_lgbm{SUFFIX}.csv")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        best_iters=result["best_iters"],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
        recipe_oof_tuned=recipe_res["tuned_log_bias_bal_acc"],
        delta_vs_recipe=tuned - recipe_res["tuned_log_bias_bal_acc"],
        error_count_lgbm=lgbm_err_count,
        error_count_recipe=rec_err_count,
        jaccard_vs_recipe=jacc,
    )
    with open(ART / f"recipe_full_te_lgbm{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote scripts/artifacts/recipe_full_te_lgbm{SUFFIX}_results.json")


if __name__ == "__main__":
    main()
