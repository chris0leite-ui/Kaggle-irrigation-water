"""Macro-recall-surrogate gradient XGB on top of recipe FE.

Custom xgb.train objective: minimize -E_softmax[R_macro] where R_macro is
macro-averaged per-class recall and softmax is applied at temperature T.

Math:
  surrogate R_macro = (1/K) Σ_k (1/N_k) Σ_{i:y_i=k} p_ik
  gradient:    g_{im} = (p_{ik*} / (K·N_{k*}·T)) · (p_{im} - δ_{m,k*})
  hessian:     h_{im} = (p_{ik*} / (K·N_{k*}·T)) · p_{im}(1 - p_{im}) + eps

Key property: gradient weight is p_{ik*}·(1-p_{ik*}), peaked at p_{ik*}=0.5.
Focuses on BOUNDARY rows that can flip with small updates — directly
attacks the macro-recall metric (binary per-row contribution at the
decision boundary). Opposite of focal (which up-weights totally-wrong rows).

Combine with CE via MR_LAMBDA: L = λ·L_CE + (1-λ)·(-R_macro). Default
λ=0.0 (pure surrogate). Set λ=0.5 if pure surrogate is unstable.

Output paths suffixed _macrorec_T{}_lam{}. SMOKE=1 → 20k train, 2 folds,
small XGB. RUN_FOLD=N → fold-N-only for rehydrate-resilient sequencing.
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
from recipe_full_te import (  # noqa: E402
    CLS_MAP, IDX2CLS, TARGET, load_and_engineer,
)
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2
RUN_FOLD = int(os.environ.get("RUN_FOLD", "0"))
TEMPERATURE = float(os.environ.get("MR_T", "1.0"))
MR_LAMBDA = float(os.environ.get("MR_LAMBDA", "0.0"))
EPS = 1e-6

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)
SUFFIX = f"_macrorec_T{TEMPERATURE:g}_lam{MR_LAMBDA:g}".replace(".", "")
N_CLASSES = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_macrorec_obj(y: np.ndarray, n_classes: int = 3,
                      temperature: float = 1.0,
                      lam_ce: float = 0.0):
    """Return xgb.train obj closure: blend of CE and macro-recall surrogate.

    grad/hess shapes are (n_rows, n_classes) flattened row-major (XGB
    convention for multi-class custom obj when num_class is set).

    Per-class N_k computed once from y; cached for all training calls.
    """
    n_rows = y.shape[0]
    class_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    K = float(n_classes)
    inv_T = 1.0 / max(temperature, 1e-6)
    # Per-row balanced weight: N / (K · N_{y_i}) — same magnitude as
    # compute_sample_weight("balanced") output. Gradients land at O(1)
    # so eps=1e-6 in Hessian doesn't drown the Newton step.
    sample_w = (n_rows / (K * class_counts[y])).astype(np.float64)  # (n_rows,)
    onehot = np.zeros((n_rows, n_classes), dtype=np.float64)
    onehot[np.arange(n_rows), y] = 1.0

    def obj(z_flat: np.ndarray, dmat) -> tuple[np.ndarray, np.ndarray]:
        # z_flat shape: (n_rows * n_classes,) row-major, OR (n_rows, n_classes)
        if z_flat.ndim == 1:
            z = z_flat.reshape(-1, n_classes)
        else:
            z = z_flat
        # Softmax with temperature
        z_T = z * inv_T
        z_T = z_T - z_T.max(axis=1, keepdims=True)
        exp_z = np.exp(z_T)
        p = exp_z / exp_z.sum(axis=1, keepdims=True)
        # Cross-entropy gradient (standard) — used both for grad-CE and as
        # the SAME (p - onehot) shape for macro-recall gradient.
        ce_diff = p - onehot  # (n_rows, n_classes)
        # Macro-recall surrogate gradient: g = (sample_w · p_{ik*} / T) · (p - onehot)
        # sample_w is balanced class weight so per-row gradient is O(1) like CE.
        p_true = p[np.arange(n_rows), y]  # (n_rows,)
        mr_scale = (p_true * inv_T * sample_w)[:, None]  # (n_rows, 1)
        g_mr = mr_scale * ce_diff
        # Diagonal Hessian: same scale × p(1-p), kept positive
        h_mr = mr_scale * (p * (1.0 - p)) + EPS
        # Blend: λ · CE + (1-λ) · MR
        if lam_ce > 0.0:
            # CE with balanced sample weights — same scale as g_mr.
            sw_col = sample_w[:, None]
            g_ce = sw_col * ce_diff
            h_ce = sw_col * p * (1.0 - p) + EPS
            g = lam_ce * g_ce + (1.0 - lam_ce) * g_mr
            h = lam_ce * h_ce + (1.0 - lam_ce) * h_mr
        else:
            g = g_mr
            h = h_mr
        # XGB 2.1+ wants (n_rows, n_classes) for multi-class custom obj.
        return g.astype(np.float32), h.astype(np.float32)

    return obj


def macrorec_eval_metric(y: np.ndarray):
    """Custom eval metric returning -macro_recall (lower = better for XGB)."""
    def feval(z_flat: np.ndarray, dmat) -> tuple[str, float]:
        if z_flat.ndim == 1:
            z = z_flat.reshape(-1, N_CLASSES)
        else:
            z = z_flat
        pred = z.argmax(axis=1)
        bal = balanced_accuracy_score(y, pred)
        return "neg_macrorec", -float(bal)
    return feval


def main() -> None:
    log(f"Macro-recall surrogate XGB  smoke={SMOKE}  T={TEMPERATURE}  "
        f"lam_ce={MR_LAMBDA}  run_fold={RUN_FOLD or 'all'}  suffix={SUFFIX!r}")
    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy().astype(np.int32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores: list[float] = []

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    xgb_params = dict(
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        # NOTE: no built-in objective — custom obj overrides
        num_class=3, tree_method="hist",
        nthread=-1, seed=SEED, verbosity=0,
        disable_default_eval_metric=1,
    )
    n_rounds = 300 if SMOKE else 3000
    es_rounds = 50 if SMOKE else 200

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if RUN_FOLD and fold != RUN_FOLD:
            continue
        log(f"=== fold {fold}/{N_FOLDS} ===")
        ck_oof = ART / f"oof_recipe_full_te{SUFFIX}_fold{fold}.npy"
        ck_te = ART / f"test_recipe_full_te{SUFFIX}_fold{fold}.npy"
        if ck_oof.exists() and ck_te.exists():
            vp = np.load(ck_oof); tp = np.load(ck_te)
            if vp.shape[0] == len(va_idx) and tp.shape[0] == len(test):
                oof[va_idx] = vp; test_pred += tp / N_FOLDS
                bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
                fold_scores.append(bal)
                log(f"  fold {fold} CACHED bal={bal:.5f}")
                continue
            log(f"  shape mismatch; re-running")
            ck_oof.unlink(); ck_te.unlink()

        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        ote = OrderedTE(a=1.0)
        X_tr_shuf = ote.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = ote.transform(X_va); X_te = ote.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + ote.te_col_names()
        # NOTE: no sample_weight — surrogate already per-class normalizes via 1/N_k
        dtr = xgb.DMatrix(X_tr[feat_cols].values, label=y[tr_idx],
                          feature_names=feat_cols)
        dva = xgb.DMatrix(X_va[feat_cols].values, label=y[va_idx],
                          feature_names=feat_cols)
        dte = xgb.DMatrix(X_te[feat_cols].values, feature_names=feat_cols)

        obj = make_macrorec_obj(y[tr_idx], n_classes=N_CLASSES,
                                temperature=TEMPERATURE, lam_ce=MR_LAMBDA)
        feval = macrorec_eval_metric(y[va_idx])

        log(f"  training xgb.train custom obj on {len(feat_cols)} feats, "
            f"{len(X_tr):,} rows  T={TEMPERATURE}  lam_ce={MR_LAMBDA}")
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=n_rounds,
            obj=obj, custom_metric=feval, maximize=False,
            evals=[(dva, "val")],
            early_stopping_rounds=es_rounds,
            verbose_eval=500,
        )
        # Predict raw logits → softmax (with same T as training, then T=1 at deploy?)
        # Use T=1 at deploy: gives sharper posteriors that downstream log-bias / blend
        # gates can use.
        z_va = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1),
                               output_margin=True).reshape(-1, N_CLASSES)
        z_te = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1),
                               output_margin=True).reshape(-1, N_CLASSES)
        # Softmax at T=1 for the blend gate (the trained logits ARE on T-scale, but
        # we want canonical posterior probs as output)
        def softmax(z):
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        vp = softmax(z_va).astype(np.float32)
        tp = softmax(z_te).astype(np.float32)

        np.save(ck_oof, vp); np.save(ck_te, tp)
        oof[va_idx] = vp; test_pred += tp / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} bal={bal:.5f}  best_iter={booster.best_iteration}")

    if RUN_FOLD:
        log(f"RUN_FOLD={RUN_FOLD} — partial run, exiting")
        return

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"OOF argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_recipe_full_te{SUFFIX}.npy", oof)
    np.save(ART / f"test_recipe_full_te{SUFFIX}.npy", test_pred)
    eps = 1e-9
    test_idx = (np.log(np.clip(test_pred, eps, 1.0)) + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_idx]})
    sub_path = SUB / f"submission_recipe_full_te{SUFFIX}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    summary = dict(
        n_folds=N_FOLDS, smoke=SMOKE,
        temperature=TEMPERATURE, mr_lambda=MR_LAMBDA,
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
    )
    with open(ART / f"recipe_full_te{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote results JSON")


if __name__ == "__main__":
    main()
