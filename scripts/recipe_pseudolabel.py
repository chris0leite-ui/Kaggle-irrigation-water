"""Pseudo-label retrain of recipe_full_te — V10 kernel's disabled toggle.

Procedure:
  1. Load recipe_full_te's test predictions (`test_recipe_full_te.npy`).
  2. Compute recipe's tuned-bias argmax class for each test row.
  3. Gate by max-prob ≥ TAU (default 0.98 — conservative to avoid boundary
     error compounding, which killed prior pseudo-label experiments on
     weaker labelers at LB 0.97352).
  4. Form pseudo-train = concat(train, pseudo_test_subset). OOF split is on
     REAL train only — pseudo rows always go to training side, never to val.
     This keeps OOF fold-alignment honest vs recipe_full_te baseline.
  5. Same FE pipeline (load_and_engineer returns the same 443-feature matrix
     for train and test). OTE per fold fits on augmented training subset
     (real_tr ∪ pseudo), so OTE statistics see pseudo-labels as ground truth.
  6. XGB training and log-bias tuning identical to recipe_full_te.

Output artefacts:
    oof_recipe_pseudolabel.npy         (OOF on REAL train rows only)
    test_recipe_pseudolabel.npy        (averaged across 5 folds)
    submission_recipe_pseudolabel.csv
    recipe_pseudolabel_results.json

Why this is likely to lift now even though 2026-04-21 pseudo-labels failed:
  - Prior labeler was hybrid_v3 at LB 0.97352 (~34k errors expected in test's
    270k rows). Labeler errors got encoded as pseudo-labels, which then
    biased OTE statistics and XGB splits on boundary rows.
  - Recipe_full_te at LB 0.97939 has ~5.7k errors expected in test — 6x
    fewer, with TAU=0.98 filtering out the riskiest boundary rows. Expected
    pseudo-label purity above 99.5%.
  - Training-data expansion (504k → ~700k rows at TAU=0.98) gives XGB more
    signal on the rare class (High: 3.3% × 270k × ~0.95 keep-rate ≈ 8.5k
    pseudo-High rows, +40% of real-train's 21k High pool).

SMOKE=1 → 20k real train + 10k pseudo test, 2 folds.
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
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, CLS_MAP, IDX2CLS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

TAU = float(os.environ.get("PSEUDO_TAU", "0.98"))
# Optional: down-weight pseudo rows during XGB training. V10 default = 1.0
# (treated identically to real data). Lower values are a conservative hedge
# against any residual labeler error.
PSEUDO_WEIGHT = float(os.environ.get("PSEUDO_WEIGHT", "1.0"))

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def apply_bias_argmax(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)


def apply_bias_maxprob(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return the max of (bias-adjusted) row probs, after renormalising."""
    lp = np.log(np.clip(probs, 1e-9, 1.0)) + bias
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    e /= e.sum(1, keepdims=True)
    return e.max(1)


def build_pseudo_subset(test_probs: np.ndarray, bias: np.ndarray,
                         tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (keep_mask, pseudo_labels, post_bias_maxprob) for test rows."""
    max_prob_post = apply_bias_maxprob(test_probs, bias)
    labels = apply_bias_argmax(test_probs, bias)
    keep = max_prob_post >= tau
    return keep, labels, max_prob_post


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           pseudo_test_idx: np.ndarray, pseudo_test_labels: np.ndarray,
           a_ote: float = 1.0) -> dict:
    """OOF on REAL train only; pseudo-test rows always in training partition."""
    y_real = train[TARGET].to_numpy()
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
    log(f"xgb_params: {xgb_params}")

    # Pre-extract pseudo subset — same rows across all folds.
    test_pseudo = test.iloc[pseudo_test_idx].copy().reset_index(drop=True)
    log(f"pseudo subset: {len(test_pseudo)} rows  "
        f"label dist = {np.bincount(pseudo_test_labels, minlength=3).tolist()}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_real), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr_real = train.iloc[tr_idx].copy().reset_index(drop=True)
        # Augment training with pseudo rows; preserve the TARGET column as
        # pseudo-labels so OrderedTE can key off it.
        X_tr_pseudo = test_pseudo.copy()
        X_tr_pseudo[TARGET] = pseudo_test_labels
        X_tr = pd.concat([X_tr_real, X_tr_pseudo], ignore_index=True)
        y_tr = X_tr[TARGET].to_numpy()

        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log(f"  fitting OrderedTE on {len(X_tr)} rows "
            f"({len(X_tr_real)} real + {len(X_tr_pseudo)} pseudo)")
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

        # Class-balanced sample weights on the augmented train. Multiply the
        # pseudo-row slice by PSEUDO_WEIGHT on top.
        sw = compute_sample_weight("balanced", y_tr).astype(np.float32)
        sw[len(X_tr_real):] *= PSEUDO_WEIGHT

        log(f"  training XGB on {len(feat_cols)} features, "
            f"{len(X_tr)} train / {len(X_va)} val")
        t0 = time.time()
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y_tr,
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y_real[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = fast_bal_acc(y_real[va_idx].astype(np.int32),
                           oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter={model.best_iteration}  "
            f"wall={time.time()-t0:.1f}s")

    overall = fast_bal_acc(y_real.astype(np.int32), oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    # Step 1: load recipe_full_te's test probs + bias.
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    recipe_bias = np.array(recipe_res["log_bias"])
    recipe_test_probs = np.load(ART / "test_recipe_full_te.npy")
    log(f"recipe tuned OOF = {recipe_res['tuned_log_bias_bal_acc']:.5f}  "
        f"bias={recipe_bias.round(4).tolist()}")

    # Step 2: gate test rows by confidence.
    keep_mask, pseudo_labels, maxprobs = build_pseudo_subset(
        recipe_test_probs, recipe_bias, TAU
    )
    log(f"τ={TAU}  keep_rate={keep_mask.mean():.4f}  "
        f"({keep_mask.sum()}/{len(keep_mask)} rows)")
    log(f"  pseudo label dist = "
        f"{np.bincount(pseudo_labels[keep_mask], minlength=3).tolist()}")
    log(f"  max-prob percentiles (kept): "
        f"p25={np.percentile(maxprobs[keep_mask], 25):.4f}  "
        f"p50={np.percentile(maxprobs[keep_mask], 50):.4f}  "
        f"p99={np.percentile(maxprobs[keep_mask], 99):.4f}")

    # Step 3: build features end-to-end (same as recipe_full_te).
    train, test, info, test_ids = load_and_engineer()

    if SMOKE:
        # Smoke constrains to 20k train / 10k test. Build a pseudo subset
        # from the smoke-sized test using the smoke-test predictions (not
        # the real recipe predictions which are on 270k rows).
        log("SMOKE: shrinking pseudo subset to match smoke-sized test")
        rng = np.random.default_rng(SEED)
        pseudo_test_idx = rng.choice(len(test), size=min(6000, len(test)),
                                     replace=False)
        # Fake smoke labels from class priors (just to exercise the path).
        pseudo_test_labels = rng.choice(3, size=len(pseudo_test_idx),
                                        p=[0.587, 0.380, 0.033])
    else:
        pseudo_test_idx = np.where(keep_mask)[0]
        pseudo_test_labels = pseudo_labels[keep_mask].astype(np.int64)

    # Step 4: 5-fold CV with pseudo rows in training side.
    result = run_cv(train, test, info, pseudo_test_idx, pseudo_test_labels)

    # Step 5: tune log-bias on OOF (still measured on real train only).
    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / "oof_recipe_pseudolabel.npy"
    test_path = ART / "test_recipe_pseudolabel.npy"
    np.save(oof_path, result["oof"])
    np.save(test_path, result["test"])
    log(f"wrote {oof_path} + {test_path}")

    # Sanity: compare with recipe_full_te's tuned OOF.
    baseline_tuned = recipe_res["tuned_log_bias_bal_acc"]
    log(f"Δ tuned OOF vs recipe_full_te = {tuned - baseline_tuned:+.5f}")

    # Build submission only if the tuned OOF actually beats recipe's.
    # (we'll still save artefacts regardless for downstream blending)
    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / "submission_recipe_pseudolabel.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        tau=TAU, pseudo_weight=PSEUDO_WEIGHT,
        pseudo_n=int(len(pseudo_test_idx)),
        pseudo_label_dist=[int(x) for x in np.bincount(
            pseudo_test_labels, minlength=3)],
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        baseline_tuned=baseline_tuned,
        delta_vs_baseline=tuned - baseline_tuned,
    )
    with open(ART / "recipe_pseudolabel_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote scripts/artifacts/recipe_pseudolabel_results.json")


if __name__ == "__main__":
    main()
