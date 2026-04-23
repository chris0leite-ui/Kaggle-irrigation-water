"""V10 recipe extended to ALL 171 pair combos (Ali Afzal's "pairwise TE magic").

The V10 kernel + our current `recipe_full_te.py` use only C(8, 2)=28 cat×cat
pair combos. The Ali Afzal public kernel
(`aliafzal9323/s6e4-0-978-xgb-cat-pairwise-te-magic`, LB 0.97779) pairs all
C(19, 2)=171 columns — including cat×num and num×num — string-concats them,
factorizes to integer codes, and target-encodes each as a 3-class OTE.

This script mirrors `recipe_full_te.py` end-to-end except:
  - `add_cat_pair_combos()` replaced with `add_all_pair_combos()` which pairs
    every (c1, c2) in combinations(cats + nums, 2).
  - `feat_info["combos"]` now has 171 entries (was 28).
  - OTE runs on ~260 categorical sources (vs 117 in baseline), producing
    ~780 OTE features (vs 351).
  - FREQ per combo: 171 entries (was 28).
  - Total feature count ≈ 1150 (was 443).

Risk: most num×num pairs string-concat two floats → near-unique keys per row
→ OTE shrinks to prior on those rows → pure noise columns. Cat×num pairs
(where cat is low-cardinality) retain repetition and are most likely to carry
new signal. We include all 171 per the published recipe; XGB's
feature_fraction=0.8 + heavy reg (alpha=5, lambda=5) should suppress noise
columns without hurting convergence.

Expected wall: ~100-150 min on 504k train (OTE 2 min/fold + XGB 15-20
min/fold on 5-fold).

Outputs:
    scripts/artifacts/oof_recipe_allpairs.npy
    scripts/artifacts/test_recipe_allpairs.npy
    scripts/artifacts/recipe_allpairs_results.json
    submissions/submission_recipe_allpairs.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
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
    add_digit_features, add_freq_features, add_lr_formula_logits,
    add_num_as_cat, add_orig_mean_std, add_threshold_flags,
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

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_all_pair_combos(train: pd.DataFrame, test: pd.DataFrame,
                        orig: pd.DataFrame,
                        all_cols: list[str]) -> list[str]:
    """Pair every combination(all_cols, 2) as string-concat + factorize.

    171 pairs for all_cols = 19 original columns (8 cats + 11 nums).
    Each resulting col is an integer factorization shared across
    train+test+orig.
    """
    new_cols: list[str] = []
    for c1, c2 in combinations(all_cols, 2):
        col = f"COMBO_{c1}_{c2}"
        for df in (train, test, orig):
            df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
        combined = pd.concat([train[col], test[col], orig[col]])
        codes, _ = pd.factorize(combined)
        s = len(train)
        t = s + len(test)
        train[col] = codes[:s]
        test[col] = codes[s:t]
        orig[col] = codes[t:]
        new_cols.append(col)
    return new_cols


def load_and_engineer_allpairs():
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

    # EXPANDED: all C(19, 2) pair combos instead of C(8, 2).
    all_19 = cats + nums
    log(f"adding ALL-pair combos: C({len(all_19)},2)={len(list(combinations(all_19, 2)))} pairs")
    combos = add_all_pair_combos(train, test, orig, all_19)

    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)

    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)

    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)

    log("adding ORIG mean/std per col")
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    # Factorize raw cats AFTER FE that needs string values.
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
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx])

        log(f"  training XGB on {len(feat_cols)} features, "
            f"{len(X_tr)} train / {len(X_va)} val")
        t0 = time.time()
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
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter={model.best_iteration}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    train, test, info, test_ids = load_and_engineer_allpairs()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / "oof_recipe_allpairs.npy"
    test_path = ART / "test_recipe_allpairs.npy"
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
    sub_path = SUB / "submission_recipe_allpairs.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")
    log(f"  pred dist: {dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
        feature_group_sizes={k: len(v) if isinstance(v, list) else v
                             for k, v in info.items() if k != "te_cols"},
        te_col_count=len(info["te_cols"]),
    )
    with open(ART / "recipe_allpairs_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote scripts/artifacts/recipe_allpairs_results.json")


if __name__ == "__main__":
    main()
