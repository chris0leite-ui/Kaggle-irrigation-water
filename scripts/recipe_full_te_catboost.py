"""CatBoost variant of recipe_full_te.py.

Swaps XGBClassifier -> CatBoostClassifier on the SAME ~440-feature set
(cat pair combos + digits + num-as-cat + threshold flags + LR-formula
logits + FREQ + ORIG mean/std + OrderedTE on ~117 cats). Novel model
family test: CatBoost's ordered boosting produces structurally different
splits than XGB on categorical-heavy data, so error footprint may be
orthogonal enough for a productive blend.

HPs chosen to mirror recipe's XGB philosophy:
  depth=4            (≈ max_depth=4)
  l2_leaf_reg=10     (heavy regularisation, matches alpha=5 + reg_lambda=5)
  iterations=2000    (≈ n_estimators=3000 cap)
  learning_rate=0.1
  loss_function=MultiClass
  class_weights=balanced or sample_weight=balanced
  od_type=Iter, od_wait=200  (early stopping)

Same 5-fold StratifiedKFold(seed=42), same per-fold OTE, same features.
Saves to distinct artifact names so it coexists with XGB recipe.

Blend diagnostic: if standalone OOF >= 0.975 and Jaccard(cat_preds, xgb_preds)
< 0.9, a productive blend with recipe_full_te is likely. If standalone
< 0.970 or Jaccard >= 0.95, null.

Wall-time budget: ~1h on CPU for 5 folds × ~1000-2000 iter.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

import os

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_engineer() -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
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
        log("SMOKE=1 — subsampling to 20k train, 10k test")
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
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

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
        orig_stats=orig_stats_cols,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: "
        f"cats={len(cats)} combos={len(combos)} digits={len(digits)} "
        f"num_as_cat={len(num_as_cat)} tres={len(tres)} logits={len(logits)} "
        f"freq={len(freq)} orig_stats={len(orig_stats_cols)} "
        f"te_cols={len(info['te_cols'])}")
    return train, test, info, test_ids


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           a_ote: float = 1.0) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    cb_params = dict(
        iterations=300 if SMOKE else 2000,
        depth=4,
        learning_rate=0.1,
        l2_leaf_reg=10.0,
        subsample=0.8,
        rsm=0.8,                # column subsample (≈ colsample_bytree)
        min_data_in_leaf=2,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=SEED,
        od_type="Iter",
        od_wait=50 if SMOKE else 200,
        bootstrap_type="Bernoulli",
        verbose=False,
        task_type="CPU",
        thread_count=-1,
        allow_writing_files=False,
    )

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

        log(f"  training CatBoost on {len(feat_cols)} features")
        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_tr[feat_cols], y[tr_idx],
            sample_weight=sw,
            eval_set=(X_va[feat_cols], y[va_idx]),
            use_best_model=True,
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter={model.tree_count_}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    # Quick orthogonality check vs recipe_full_te (LB-best). Skipped in SMOKE
    # (row counts wouldn't match).
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    jacc = None
    cat_err_count = None
    recipe_err_count = None
    if not SMOKE:
        recipe_oof = np.load(ART / "oof_recipe_full_te.npy")
        recipe_bias = np.array(recipe_res["log_bias"])
        recipe_pred = (np.log(np.clip(recipe_oof, 1e-9, 1.0)) + recipe_bias).argmax(1)
        cat_pred = (np.log(np.clip(result["oof"], 1e-9, 1.0)) + bias).argmax(1)
        recipe_err = recipe_pred != y
        cat_err = cat_pred != y
        jacc = float((recipe_err & cat_err).sum() / max(1, (recipe_err | cat_err).sum()))
        cat_err_count = int(cat_err.sum())
        recipe_err_count = int(recipe_err.sum())
        log(f"error count:  catboost={cat_err_count}  recipe_xgb={recipe_err_count}")
        log(f"Jaccard vs recipe_full_te = {jacc:.4f}")

    np.save(ART / "oof_recipe_full_te_catboost.npy", result["oof"])
    np.save(ART / "test_recipe_full_te_catboost.npy", result["test"])

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / "submission_recipe_full_te_catboost.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
        recipe_oof_tuned=float(recipe_res["tuned_log_bias_bal_acc"]),
        delta_vs_recipe=float(tuned - recipe_res["tuned_log_bias_bal_acc"]),
        error_count_catboost=cat_err_count,
        error_count_recipe=recipe_err_count,
        jaccard_vs_recipe=jacc,
    )
    with open(ART / "recipe_full_te_catboost_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote scripts/artifacts/recipe_full_te_catboost_results.json")


if __name__ == "__main__":
    main()
