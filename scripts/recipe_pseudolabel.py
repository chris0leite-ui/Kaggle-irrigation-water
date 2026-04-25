"""Pseudo-label retrain of recipe_full_te.

Procedure:
  1. Load recipe_full_te's test predictions and tuned bias.
  2. Compute per-row argmax + max-prob after applying the bias.
  3. Keep test rows with max-prob >= TAU (default 0.98) as pseudo-train.
  4. 5-fold CV on REAL train; pseudo rows always go to training side.
  5. OrderedTE per fold fits on (real_tr ∪ pseudo).
  6. XGB training and log-bias tuning identical to recipe_full_te.

Outputs:
  scripts/artifacts/oof_recipe_pseudolabel.npy
  scripts/artifacts/test_recipe_pseudolabel.npy
  scripts/artifacts/recipe_pseudolabel_results.json
  submissions/submission_recipe_pseudolabel.csv
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
TAU = 0.98
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pseudo_subset(test_probs: np.ndarray, bias: np.ndarray,
                        tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (keep_mask, pseudo_labels) using bias-adjusted softmax."""
    lp = np.log(np.clip(test_probs, 1e-9, 1.0)) + bias
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    p = e / e.sum(1, keepdims=True)
    keep = p.max(1) >= tau
    labels = lp.argmax(1)
    return keep, labels


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           pseudo_test_idx: np.ndarray, pseudo_test_labels: np.ndarray) -> dict:
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
        n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )

    test_pseudo = test.iloc[pseudo_test_idx].copy().reset_index(drop=True)
    log(f"pseudo subset: {len(test_pseudo)} rows  "
        f"label dist = {np.bincount(pseudo_test_labels, minlength=3).tolist()}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_real), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr_real = train.iloc[tr_idx].copy().reset_index(drop=True)
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
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y_tr).astype(np.float32)

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
            f"best_iter={model.best_iteration}  wall={time.time()-t0:.1f}s")

    overall = fast_bal_acc(y_real.astype(np.int32), oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    labeler_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    labeler_bias = np.array(labeler_res["log_bias"])
    labeler_test_probs = np.load(ART / "test_recipe_full_te.npy")
    log(f"labeler OOF tuned = {labeler_res['tuned_log_bias_bal_acc']:.5f}  "
        f"bias={labeler_bias.round(4).tolist()}")

    keep_mask, pseudo_labels = build_pseudo_subset(
        labeler_test_probs, labeler_bias, TAU
    )
    log(f"τ={TAU}  keep_rate={keep_mask.mean():.4f}  "
        f"({keep_mask.sum()}/{len(keep_mask)} rows)")
    log(f"  pseudo label dist = "
        f"{np.bincount(pseudo_labels[keep_mask], minlength=3).tolist()}")

    train, test, info, test_ids = load_and_engineer()

    if SMOKE:
        log("SMOKE: synthesising a pseudo subset for the 10k smoke test")
        rng = np.random.default_rng(SEED)
        pseudo_test_idx = rng.choice(len(test), size=min(6000, len(test)),
                                     replace=False)
        pseudo_test_labels = rng.choice(3, size=len(pseudo_test_idx),
                                        p=[0.587, 0.380, 0.033])
    else:
        pseudo_test_idx = np.where(keep_mask)[0]
        pseudo_test_labels = pseudo_labels[keep_mask].astype(np.int64)

    result = run_cv(train, test, info, pseudo_test_idx, pseudo_test_labels)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / "oof_recipe_pseudolabel.npy", result["oof"])
    np.save(ART / "test_recipe_pseudolabel.npy", result["test"])
    log(f"wrote {ART}/oof_recipe_pseudolabel.npy + test_recipe_pseudolabel.npy")

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
        seed=SEED, n_folds=N_FOLDS, tau=TAU,
        pseudo_n=int(len(pseudo_test_idx)),
        pseudo_label_dist=[int(x) for x in np.bincount(
            pseudo_test_labels, minlength=3)],
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
    )
    res_path = ART / "recipe_pseudolabel_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
