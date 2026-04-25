"""N1: One-vs-Rest XGB on the V10 recipe feature set.

Mechanism: 3 independent binary:logistic XGB heads (Low/Medium/High vs rest).
Each head is class-imbalance-aware via `scale_pos_weight = neg/pos`. After
all folds, each head's OOF is per-class isotonic-calibrated to remove the
SPW-induced scale distortion, then the 3 columns are softmax-renormalized
into a 3-class posterior. Decision rule: standard `tune_log_bias` (which
subsumes Bayes-optimal threshold-at-prior; see N1 design notes).

Why this is genuinely new vs the existing recipe: each binary head sees a
2-class CE gradient that does NOT couple to the other classes. On a 3.3% rare
class, multi-class softmax CE distributes its gradient across all 3 outputs
per row, so the rare-class boundary updates only via the shared softmax
denominator. OvR isolates the High-vs-rest gradient — that's the one
architectural trick we have not tested.

5-fold StratifiedKFold(seed=42) aligned with every other OOF on disk.
SMOKE=1 → 20k/2-fold/200-iter for end-to-end validation in ~3 min.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402
from recipe_full_te import (  # noqa: E402
    CLS_MAP, IDX2CLS, TARGET, load_and_engineer,
)
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_head(X_tr, y_tr_bin, X_va, y_va_bin, X_te, spw: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit one binary:logistic XGB head with scale_pos_weight=spw.

    Returns (val_probs, test_probs, best_iter).
    """
    params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5.0, reg_lambda=5.0,
        max_bin=256 if SMOKE else 1024,
        objective="binary:logistic", tree_method="hist",
        eval_metric="logloss", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
        scale_pos_weight=float(spw),
    )
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr_bin, eval_set=[(X_va, y_va_bin)], verbose=500)
    pv = m.predict_proba(X_va)[:, 1].astype(np.float32)
    pt = m.predict_proba(X_te)[:, 1].astype(np.float32)
    return pv, pt, int(m.best_iteration)


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict) -> dict:
    y = train[TARGET].to_numpy().astype(np.int32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    raw_oof = np.zeros((len(train), 3), dtype=np.float32)   # uncalibrated head outputs
    raw_test = np.zeros((len(test), 3), dtype=np.float32)

    fold_iters: list[list[int]] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

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
        log(f"  OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        Xtr = X_tr[feat_cols]
        Xva = X_va[feat_cols]
        Xte = X_te[feat_cols]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        iters = []
        for k in range(3):
            ytrk = (y_tr == k).astype(np.int32)
            yvak = (y_va == k).astype(np.int32)
            n_pos = int(ytrk.sum()); n_neg = len(ytrk) - n_pos
            spw = max(n_neg / max(n_pos, 1), 1e-6)
            log(f"  head class={k} ({['Low','Medium','High'][k]})  "
                f"n_pos={n_pos:,}  n_neg={n_neg:,}  spw={spw:.2f}")
            pv, pt, bi = fit_head(Xtr, ytrk, Xva, yvak, Xte, spw)
            raw_oof[va_idx, k] = pv
            raw_test[:, k] += pt / N_FOLDS
            iters.append(bi)
        fold_iters.append(iters)

        # quick fold diagnostic on softmax(raw)
        eps = 1e-9
        z = np.log(np.clip(raw_oof[va_idx], eps, 1.0))
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z); pf = ez / ez.sum(axis=1, keepdims=True)
        bal = fast_bal_acc(y_va, pf.argmax(1))
        log(f"  fold {fold} raw-softmax argmax bal={bal:.5f}  iters={iters}")

    # Per-class OOF isotonic calibration. Each iso is fit on the leak-free
    # OOF (each row predicted by a fold that did not see it) against the
    # true binary indicator for that class.
    log("fitting per-class isotonic calibration on OOF")
    cal_oof = np.zeros_like(raw_oof)
    cal_test = np.zeros_like(raw_test)
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(raw_oof[:, k], (y == k).astype(np.float32))
        cal_oof[:, k] = ir.predict(raw_oof[:, k])
        cal_test[:, k] = ir.predict(raw_test[:, k])
    # Softmax-renormalize the calibrated columns into a 3-class posterior.
    eps = 1e-9
    z_oof = np.log(np.clip(cal_oof, eps, 1.0))
    z_oof = z_oof - z_oof.max(axis=1, keepdims=True)
    ez_oof = np.exp(z_oof); oof = (ez_oof / ez_oof.sum(axis=1, keepdims=True)).astype(np.float32)
    z_te = np.log(np.clip(cal_test, eps, 1.0))
    z_te = z_te - z_te.max(axis=1, keepdims=True)
    ez_te = np.exp(z_te); test_p = (ez_te / ez_te.sum(axis=1, keepdims=True)).astype(np.float32)

    overall = fast_bal_acc(y, oof.argmax(1))
    log(f"=== OOF (cal+softmax) argmax bal_acc = {overall:.5f}")
    return dict(oof=oof, test=test_p, raw_oof=raw_oof, raw_test=raw_test,
                overall_argmax=float(overall), fold_iters=fold_iters,
                feat_cols=feat_cols)


def main():
    log(f"N1 OvR-XGB on V10 recipe  smoke={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    res = run_cv(train, test, info)

    y = train[TARGET].to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(res["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_xgb_ovr_recipe{suffix}.npy", res["oof"])
    np.save(ART / f"test_xgb_ovr_recipe{suffix}.npy", res["test"])
    np.save(ART / f"oof_xgb_ovr_recipe_raw{suffix}.npy", res["raw_oof"])
    np.save(ART / f"test_xgb_ovr_recipe_raw{suffix}.npy", res["raw_test"])
    log(f"wrote oof/test xgb_ovr_recipe (cal + raw)")

    eps = 1e-9
    test_log = np.log(np.clip(res["test"], eps, 1.0))
    pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in pred_idx]})
    sub_path = SUB / f"submission_xgb_ovr_recipe{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}  "
        f"dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE,
        overall_argmax_bal_acc=res["overall_argmax"],
        tuned_log_bias_bal_acc=tuned, log_bias=bias.tolist(),
        fold_iters=res["fold_iters"],
        n_features=len(res["feat_cols"]),
    )
    with open(ART / f"xgb_ovr_recipe{suffix}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
