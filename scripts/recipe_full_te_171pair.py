"""171-pair extension of the V10 recipe — Ali Afzal's pairwise-TE magic.

Difference from `recipe_full_te.py`:
  - Bins every numeric to N_BINS quantile bins (BIN_<num>; default 16).
  - Builds ALL C(19, 2) = 171 pair combos from (cats + BIN_<num>)
    instead of only C(8, 2) = 28 cat x cat combos.
  - Expanded TE feature set: ~171 combo OTEs vs ~28 in the V10 baseline.

Everything else identical: same digits / freq / orig_stats / tres / logits
/ num_as_cat blocks, same OrderedTE(a=1), same heavy-reg XGB, same 5-fold
StratifiedKFold(seed=42) split for OOF alignment.

Knobs (env vars):
  SMOKE=1   2-fold, 20k train, 10k test, capped XGB iters (~3-5 min wall)
  N_BINS=N  override the per-numeric bin count (default 16)

Outputs:
  scripts/artifacts/oof_recipe_171pair.npy
  scripts/artifacts/test_recipe_171pair.npy
  scripts/artifacts/recipe_171pair_results.json
  submissions/submission_recipe_171pair.csv
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
from recipe_pair_features import add_quantile_bins  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

SMOKE = os.environ.get("SMOKE") == "1"
N_BINS = int(os.environ.get("N_BINS", "16"))
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------- data + features
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

    # 171-pair extension: bin numerics, then build combos over (cats + bins).
    log(f"adding quantile-bin numeric cats (N_BINS={N_BINS})")
    bin_cols = add_quantile_bins(train, test, orig, nums, n_bins=N_BINS)

    log("adding all-pair combos over (cats + bin_cols)")
    pair_keys = cats + bin_cols  # 8 + 11 = 19 keys -> C(19,2) = 171 pairs
    combos = add_cat_pair_combos(train, test, orig, pair_keys)
    log(f"  built {len(combos)} pair combos (expected 171 if 8 cats + 11 bins)")

    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)

    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)

    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)

    log("adding ORIG mean/std per col")
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    # Factorize raw cats AFTER all FE that needs string values is done.
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    info = dict(
        nums=nums, cats=cats, bins=bin_cols, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats_cols,
        # bin_cols enter TE too (they're low-card categoricals built from nums)
        te_cols=cats + bin_cols + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: cats={len(cats)} bins={len(bin_cols)} "
        f"combos={len(combos)} digits={len(digits)} num_as_cat={len(num_as_cat)} "
        f"tres={len(tres)} logits={len(logits)} freq={len(freq)} "
        f"orig_stats={len(orig_stats_cols)} te_cols={len(info['te_cols'])}")
    return train, test, info, test_ids


# --------------------------------------------------------- training loop
def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           a_ote: float = 1.0) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

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
        log(f"    OTE done in {time.time()-t0:.1f}s "
            f"({len(te.te_col_names())} OTE cols)")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx])

        log(f"  training XGB on {len(feat_cols)} features")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y[tr_idx],
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={model.best_iteration}")

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

    suffix = "_smoke" if SMOKE else ""
    oof_path = ART / f"oof_recipe_171pair{suffix}.npy"
    test_path = ART / f"test_recipe_171pair{suffix}.npy"
    np.save(oof_path, result["oof"])
    np.save(test_path, result["test"])
    log(f"wrote {oof_path} + {test_path}")

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / f"submission_recipe_171pair{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")
    log(f"  pred dist: {dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, n_bins=N_BINS, smoke=SMOKE,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
        feature_group_sizes={k: len(v) if isinstance(v, list) else v
                             for k, v in info.items() if k != "te_cols"},
        te_col_count=len(info["te_cols"]),
    )
    with open(ART / f"recipe_171pair_results{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote scripts/artifacts/recipe_171pair_results{suffix}.json")


if __name__ == "__main__":
    main()
