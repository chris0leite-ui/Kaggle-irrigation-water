"""TTA pipeline on top of the full recipe_full_te FE + XGB.

Trains the same 5-fold heavy-reg XGB on the same 443-feature recipe.
At val + test inference, generates K perturbed copies of each row (noise
on the 4 rule-threshold numerics only), recomputes ONLY the threshold-
derived features (flags, LR logits, digit cols for those 4 nums),
predicts, log-averages across K. OTE / FREQ / num_as_cat / combos are
held fixed — perturbing them would create unknown keys and degenerate
to prior, adding noise rather than smoothing.

Iterates over sigmas in one run so training cost is paid once.

Env vars:
  SMOKE=1               20k train, 10k test, 2 folds, tiny XGB.
  TTA_K=<int>           perturbations per sigma (default 3).
  TTA_SIGMAS="0.02,0.05,0.10"   comma-sep sigmas (IQR-scaled).
  FOLD_SEED=<int>       override fold split (default 42 for alignment).
  MAX_BIN=<int>         XGB max_bin (default 512, dropped from recipe's
                         1024 to halve histogram memory — still gives
                         >99% split AUC per the recipe comment).

Outputs per sigma:
  scripts/artifacts/oof_tta_recipe_s{tag}.npy
  scripts/artifacts/test_tta_recipe_s{tag}.npy
Plus baseline (no TTA, same training path) for sanity:
  scripts/artifacts/oof_tta_recipe_baseline.npy
  scripts/artifacts/test_tta_recipe_baseline.npy
And a results JSON with all OOF scores + per-sigma deltas.
"""
from __future__ import annotations

import gc
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
from tta_helpers import (  # noqa: E402
    THRESHOLD_NUMS, compute_iqr, perturb, recompute_threshold_derived,
    apply_tta_override,
)

SEED = 42
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
FOLD_SEED = int(os.environ.get("FOLD_SEED", str(SEED)))
TTA_K = int(os.environ.get("TTA_K", "2" if SMOKE else "3"))
TTA_SIGMAS = [float(s) for s in os.environ.get(
    "TTA_SIGMAS", "0.05" if SMOKE else "0.02,0.05,0.10").split(",")]
MAX_BIN = int(os.environ.get("MAX_BIN", "256" if SMOKE else "512"))

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sigma_tag(s: float) -> str:
    return f"{s:.3f}".replace("0.", "").replace(".", "p")


# -------- FE is the same as recipe_full_te, but we keep the raw 4 threshold
#          numerics (pre-factorization) and Stage/Mulch strings aside for TTA.
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
        log("SMOKE=1 — subsampling to 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:10_000]

    # Raw arrays saved for TTA BEFORE any factorization / mutation. Stage
    # and mulch remain strings in the original DataFrames at this point.
    raw_train = {c: train[c].to_numpy().astype(np.float32) for c in THRESHOLD_NUMS}
    raw_test = {c: test[c].to_numpy().astype(np.float32) for c in THRESHOLD_NUMS}
    stage_train = train["Crop_Growth_Stage"].astype(str).to_numpy()
    stage_test = test["Crop_Growth_Stage"].astype(str).to_numpy()
    mulch_train = train["Mulching_Used"].astype(str).to_numpy()
    mulch_test = test["Mulching_Used"].astype(str).to_numpy()

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)} cats={len(cats)} train={len(train)} test={len(test)}")

    log("FE: threshold flags + LR logits + combos + digits + num_as_cat + FREQ + ORIG stats")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)
    combos = add_cat_pair_combos(train, test, orig, cats)
    digits = add_digit_features(train, test, orig, nums)
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    freq = add_freq_features(train, test, orig, cats + combos)
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]; test[c] = codes[s:t]; orig[c] = codes[t:]

    # Which digit cols survived the test-constant filter? Needed when we
    # overwrite digit cols during TTA (only replace cols that weren't dropped).
    surviving_digits = set(digits)

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats_cols,
        te_cols=cats + combos + digits + num_as_cat + tres,
        surviving_digits=surviving_digits,
    )
    raw = dict(train=raw_train, test=raw_test,
               stage_train=stage_train, stage_test=stage_test,
               mulch_train=mulch_train, mulch_test=mulch_test)
    # orig is no longer needed after FE — release before training.
    del orig
    gc.collect()
    return train, test, info, test_ids, raw


# -------- core: one fold train + K×|sigmas|+1 inference passes.
def train_fold_and_predict(X_tr, y_tr, X_va, X_te, feat_cols, xgb_params):
    sw = compute_sample_weight("balanced", y_tr)
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr[feat_cols], y_tr, sample_weight=sw,
              eval_set=[(X_va[feat_cols], None)] if False else None,
              verbose=False)
    return model


def _predict_with_override(model, X_base, feat_cols, tta_df, surv_digits):
    X = X_base[feat_cols].copy()
    apply_tta_override(X, tta_df, surv_digits)
    return model.predict_proba(X).astype(np.float32)


def run_cv(train, test, info, raw, test_ids):
    y = train[TARGET].to_numpy()
    iqr = compute_iqr(train)
    log(f"IQR per threshold num: {iqr}")
    log(f"TTA_K={TTA_K}  TTA_SIGMAS={TTA_SIGMAS}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    # Preallocate one OOF/test array per variant.
    variants = ["baseline"] + [f"s{sigma_tag(s)}" for s in TTA_SIGMAS]
    oofs = {v: np.zeros((len(train), 3), dtype=np.float32) for v in variants}
    tests = {v: np.zeros((len(test), 3), dtype=np.float32) for v in variants}

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=MAX_BIN,
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

        log("  fitting OTE")
        t0 = time.time()
        rng_shuf = np.random.default_rng(SEED + fold)
        perm = rng_shuf.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        log(f"  training XGB on {len(feat_cols)} features, {len(X_tr):,} rows")
        y_tr = y[tr_idx]
        sw = compute_sample_weight("balanced", y_tr)
        xgb_params["eval_metric"] = "mlogloss"
        model = xgb.XGBClassifier(**xgb_params)
        t0 = time.time()
        model.fit(X_tr[feat_cols], y_tr, sample_weight=sw,
                  eval_set=[(X_va[feat_cols], y[va_idx])], verbose=False)
        log(f"    XGB done in {time.time()-t0:.1f}s  best_iter={model.best_iteration}")

        # Baseline (no TTA).
        t0 = time.time()
        oofs["baseline"][va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        tests["baseline"] += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oofs["baseline"][va_idx].argmax(1))
        log(f"  fold {fold} baseline argmax_bal_acc = {bal:.5f}  inf_t={time.time()-t0:.1f}s")

        # TTA per (sigma, k).
        raw_va = {c: raw["train"][c][va_idx] for c in THRESHOLD_NUMS}
        raw_te = raw["test"]
        stage_va = raw["stage_train"][va_idx]
        mulch_va = raw["mulch_train"][va_idx]

        eps = 1e-12
        for s in TTA_SIGMAS:
            tag = f"s{sigma_tag(s)}"
            t0 = time.time()
            va_logs = np.zeros((len(va_idx), 3), dtype=np.float64)
            te_logs = np.zeros((len(test), 3), dtype=np.float64)
            for k in range(TTA_K):
                rng = np.random.default_rng(SEED + 1000 * fold + k)
                pert_va = perturb(raw_va, s, iqr, rng)
                pert_te = perturb(raw_te, s, iqr, rng)
                tta_va = recompute_threshold_derived(
                    pert_va, stage_va, mulch_va)
                tta_te = recompute_threshold_derived(
                    pert_te, raw["stage_test"], raw["mulch_test"])
                p_va = _predict_with_override(
                    model, X_va, feat_cols, tta_va, info["surviving_digits"])
                p_te = _predict_with_override(
                    model, X_te, feat_cols, tta_te, info["surviving_digits"])
                va_logs += np.log(np.clip(p_va, eps, 1.0))
                te_logs += np.log(np.clip(p_te, eps, 1.0))
            va_logs /= TTA_K
            te_logs /= TTA_K
            va_softmax = np.exp(va_logs - va_logs.max(1, keepdims=True))
            va_softmax /= va_softmax.sum(1, keepdims=True)
            te_softmax = np.exp(te_logs - te_logs.max(1, keepdims=True))
            te_softmax /= te_softmax.sum(1, keepdims=True)
            oofs[tag][va_idx] = va_softmax.astype(np.float32)
            tests[tag] += te_softmax.astype(np.float32) / N_FOLDS
            bal = balanced_accuracy_score(y[va_idx], oofs[tag][va_idx].argmax(1))
            log(f"  fold {fold} TTA {tag} argmax_bal_acc = {bal:.5f}  inf_t={time.time()-t0:.1f}s")

        # End-of-fold cleanup so memory doesn't accumulate between folds.
        del X_tr, X_va, X_te, X_tr_shuf, te, model
        gc.collect()

    return oofs, tests


def main():
    log(f"config: SMOKE={SMOKE} FOLDS={N_FOLDS} K={TTA_K} SIGMAS={TTA_SIGMAS}")
    train, test, info, test_ids, raw = load_and_engineer()
    oofs, tests = run_cv(train, test, info, raw, test_ids)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    results = {"smoke": SMOKE, "n_folds": N_FOLDS,
               "tta_k": TTA_K, "tta_sigmas": TTA_SIGMAS,
               "variants": {}}
    for tag, oof in oofs.items():
        argmax = balanced_accuracy_score(y, oof.argmax(1))
        bias, tuned = tune_log_bias(oof, y, prior)
        results["variants"][tag] = {
            "argmax_bal_acc": float(argmax),
            "tuned_bal_acc": float(tuned),
            "log_bias": bias.tolist(),
        }
        suffix = "" if SMOKE is False and tag == "baseline" else ""
        oof_path = ART / f"oof_tta_recipe_{tag}{'_smoke' if SMOKE else ''}.npy"
        te_path = ART / f"test_tta_recipe_{tag}{'_smoke' if SMOKE else ''}.npy"
        np.save(oof_path, oof)
        np.save(te_path, tests[tag])
        log(f"{tag}: argmax={argmax:.5f}  tuned={tuned:.5f}  "
            f"bias={[round(b, 3) for b in bias]}")
        log(f"  wrote {oof_path.name} + {te_path.name}")

    # Deltas vs baseline.
    base_tuned = results["variants"]["baseline"]["tuned_bal_acc"]
    for tag, v in results["variants"].items():
        v["delta_vs_baseline"] = v["tuned_bal_acc"] - base_tuned

    res_path = ART / f"tta_recipe_results{'_smoke' if SMOKE else ''}.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
