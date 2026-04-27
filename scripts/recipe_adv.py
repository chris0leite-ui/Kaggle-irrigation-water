"""Adversarial-robustness recipe XGB.

Reuses recipe_full_te's load_and_engineer (FE + 10k OTE source) but writes
its own training loop with σ × IQR Gaussian noise injected on the 11 raw
numeric columns of tr_idx ROWS ONLY, AFTER FE has been computed from
clean values (so derived features stay clean; only raw numerics seen by
the trees are noisy).

Env vars:
  ADV_SIGMA — Gaussian σ as fraction of IQR (default 0.03)
  SMOKE     — 1 to shrink to 20k train / 2-fold / 300 iters

Outputs (suffix _adv{σ_pct}):
  scripts/artifacts/oof_recipe_adv_s{σ_pct}.npy
  scripts/artifacts/test_recipe_adv_s{σ_pct}.npy
  scripts/artifacts/recipe_adv_s{σ_pct}_results.json
  submissions/submission_recipe_adv_s{σ_pct}.csv
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
from recipe_adv_helpers import (RAW_NUMS, compute_iqrs,  # noqa: E402
                                 perturb_train_inplace)
from recipe_full_te import load_and_engineer  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
TARGET = "Irrigation_Need"
IDX2CLS = {0: "Low", 1: "Medium", 2: "High"}
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
ADV_SIGMA = float(os.environ.get("ADV_SIGMA", "0.03"))

TAG = f"adv_s{int(round(ADV_SIGMA * 1000)):03d}"
ART = Path("scripts/artifacts"); SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True); SUB.mkdir(exist_ok=True, parents=True)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"config: ADV_SIGMA={ADV_SIGMA}  TAG={TAG}  SMOKE={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()

    # Reference IQRs from the FULL clean training set (pre-fold).
    iqrs = compute_iqrs(train, RAW_NUMS)
    log(f"IQRs (RAW): {[(c, round(iqrs[c], 3)) for c in RAW_NUMS]}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
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
        eval_metric="mlogloss", enable_categorical=False,
        n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )

    # Per-fold checkpoints (rehydrate-resilient).
    cached = set()
    for f in range(1, N_FOLDS + 1):
        if (ART / f"oof_recipe_{TAG}_fold{f}.npy").exists() and \
           (ART / f"test_recipe_{TAG}_fold{f}.npy").exists():
            cached.add(f)
    if cached:
        log(f"resume: {len(cached)} folds cached: {sorted(cached)}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        if fold in cached:
            vp = np.load(ART / f"oof_recipe_{TAG}_fold{fold}.npy")
            tp = np.load(ART / f"test_recipe_{TAG}_fold{fold}.npy")
            oof[va_idx] = vp; test_pred += tp / N_FOLDS
            bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
            fold_scores.append(bal)
            log(f"  fold {fold} CACHED  argmax_bal_acc = {bal:.5f}")
            continue

        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        # OrderedTE on CLEAN tr (recipe convention).
        log("  fitting OrderedTE on clean tr")
        t0 = time.time()
        rng_te = np.random.default_rng(SEED + fold)
        perm = rng_te.permutation(len(X_tr))
        X_tr_sh = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_sh = te.fit(X_tr_sh, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_sh.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va); X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"]
                     + te.te_col_names())

        # KEY STEP: perturb raw numerics in tr (post-FE, pre-XGB).
        rng = np.random.default_rng(SEED * 1000 + fold)
        n_pert = sum(c in X_tr.columns for c in RAW_NUMS)
        log(f"  injecting σ={ADV_SIGMA}×IQR Gaussian noise on {n_pert} raw numerics")
        perturb_train_inplace(X_tr, iqrs, ADV_SIGMA, rng)

        y_tr = y[tr_idx]
        sw = compute_sample_weight("balanced", y_tr)
        log(f"  training XGB on {len(feat_cols)} features, {len(X_tr):,} rows")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr[feat_cols], y_tr, sample_weight=sw,
                  eval_set=[(X_va[feat_cols], y[va_idx])], verbose=500)
        vp = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        tp = model.predict_proba(X_te[feat_cols]).astype(np.float32)
        oof[va_idx] = vp; test_pred += tp / N_FOLDS

        np.save(ART / f"oof_recipe_{TAG}_fold{fold}.npy", vp)
        np.save(ART / f"test_recipe_{TAG}_fold{fold}.npy", tp)
        bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={model.best_iteration}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_recipe_{TAG}.npy", oof)
    np.save(ART / f"test_recipe_{TAG}.npy", test_pred)

    eps = 1e-9
    test_pred_idx = (np.log(np.clip(test_pred, eps, 1.0)) + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in test_pred_idx]})
    sub.to_csv(SUB / f"submission_recipe_{TAG}.csv", index=False)

    summary = dict(
        adv_sigma=ADV_SIGMA, tag=TAG, smoke=SMOKE,
        seed=SEED, n_folds=N_FOLDS,
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
        n_features=len(feat_cols),
        iqrs=iqrs,
    )
    with open(ART / f"recipe_{TAG}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/recipe_{TAG}_results.json")


if __name__ == "__main__":
    main()
