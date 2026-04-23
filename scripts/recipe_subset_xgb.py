"""Recipe-subset XGBs for blend diversity (N1 from CLAUDE.md Hypothesis board).

After 4 tree-family nulls on recipe (XGB, LGBM, CatBoost CPU+GPU all
Jaccard 0.78-0.84), the pattern: further tree additions on the same
features are null. To break it, change the feature set.

This script trains recipe XGB on SUBSETS of the ~440 cols by dropping
one feature block at a time. Each subset produces an XGB with a
different decision surface. Blend with full recipe and if the subset's
Jaccard < 0.75 AND error count within +10%, the blend may lift.

Variants controlled via RECIPE_SUBSET env var:
  no_digits   drop 66 digit cols + their OTE contributions
  no_combos   drop 28 pair combos + their OTE contributions
  no_ote      drop ALL OTE features (numeric-only XGB: 85 features)
  no_orig     drop 38 ORIG mean/std features (OTE on orig stats stays)

Pipeline mirrors recipe_full_te.py (5-fold seed=42, OrderedTE with a=1,
class-balanced sample weights, heavy-reg XGB). Outputs distinct
artifacts per variant. Wall ~30-45 min each on CPU.
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

# RECIPE_SUBSET env var picks which block(s) to drop.
# Valid values: no_digits, no_combos, no_ote, no_orig
VARIANT = os.environ.get("RECIPE_SUBSET", "no_ote")
if VARIANT not in ("no_digits", "no_combos", "no_ote", "no_orig"):
    raise ValueError(
        f"Unknown RECIPE_SUBSET={VARIANT!r}. "
        f"Valid: no_digits, no_combos, no_ote, no_orig"
    )

SUFFIX = f"_{VARIANT}"

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

    # Skip feature blocks per VARIANT.
    combos = []
    digits = []
    num_as_cat = []
    orig_stats = []

    if VARIANT != "no_combos":
        log("adding cat x cat pair combos")
        combos = add_cat_pair_combos(train, test, orig, cats)
    else:
        log("SKIP combos (no_combos variant)")

    if VARIANT != "no_digits":
        log("adding digit features")
        digits = add_digit_features(train, test, orig, nums)
    else:
        log("SKIP digits (no_digits variant)")

    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)

    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)

    if VARIANT != "no_orig":
        log("adding ORIG mean/std per col")
        orig_stats = add_orig_mean_std(train, test, orig, nums + cats, TARGET)
    else:
        log("SKIP ORIG mean/std (no_orig variant)")

    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    # For no_ote: we still build the features but won't target-encode them;
    # for others, te_cols = cats + combos + digits + num_as_cat + tres.
    te_cols = cats + combos + digits + num_as_cat + tres

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats,
        te_cols=te_cols,
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

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        enable_categorical=False, n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        if VARIANT != "no_ote" and info["te_cols"]:
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
            feat_cols = numeric_feats + te.te_col_names()
            log(f"    OTE done in {time.time()-t0:.1f}s  ({len(te.te_col_names())} OTE cols)")
        else:
            feat_cols = numeric_feats
            log(f"  no OTE — using {len(feat_cols)} numeric features only")

        sw = compute_sample_weight("balanced", y[tr_idx])

        log(f"  training XGB on {len(feat_cols)} features")
        t_fit = time.time()
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y[tr_idx],
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        bi = int(model.best_iteration)
        best_iters.append(bi)
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
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
    log(f"=== RECIPE_SUBSET variant = {VARIANT} ===")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_recipe{SUFFIX}.npy", result["oof"])
    np.save(ART / f"test_recipe{SUFFIX}.npy", result["test"])

    # Jaccard vs full recipe (skip in SMOKE).
    recipe_res = json.loads((ART/'recipe_full_te_results.json').read_text())
    jacc = None
    subset_err_count = None
    recipe_err_count = None
    if not SMOKE:
        recipe_oof = np.load(ART / "oof_recipe_full_te.npy")
        recipe_bias = np.array(recipe_res['log_bias'])
        rec_pred = (np.log(np.clip(recipe_oof, 1e-9, 1.0)) + recipe_bias).argmax(1)
        sub_pred = (np.log(np.clip(result["oof"], 1e-9, 1.0)) + bias).argmax(1)
        rec_err = rec_pred != y
        sub_err = sub_pred != y
        jacc = float((rec_err & sub_err).sum() / max(1, (rec_err | sub_err).sum()))
        subset_err_count = int(sub_err.sum())
        recipe_err_count = int(rec_err.sum())
        log(f"errors:  subset={subset_err_count}  recipe={recipe_err_count}")
        log(f"Jaccard vs recipe = {jacc:.4f}")

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub.to_csv(SUB / f"submission_recipe{SUFFIX}_tuned.csv", index=False)
    log(f"wrote submission_recipe{SUFFIX}_tuned.csv")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        variant=VARIANT,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        best_iters=result["best_iters"],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
        recipe_oof_tuned=recipe_res["tuned_log_bias_bal_acc"],
        delta_vs_recipe=tuned - recipe_res["tuned_log_bias_bal_acc"],
        error_count_subset=subset_err_count,
        error_count_recipe=recipe_err_count,
        jaccard_vs_recipe=jacc,
    )
    with open(ART / f"recipe{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote scripts/artifacts/recipe{SUFFIX}_results.json")


if __name__ == "__main__":
    main()
