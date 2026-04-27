"""N1 — Macro-recall surrogate at the META-STACKER level.

Mirrors tier1b_xgb_metastack.py exactly EXCEPT:
  - xgb.train uses the macro-recall surrogate gradient (not CE+softprob)
  - lam_ce blends in a small CE component to keep gradients alive (default 0.3,
    matches recipe_macrorecall.py production setting)

Reuses build_lbbest_stack + load_pool + meta-feature construction from
tier1b_xgb_metastack so this is a drop-in objective swap.

Hypothesis: the +0.62pp H-recall lift we saw at the BASE level may COMPOUND
at the META level because:
  - Components in the bank already carry orthogonal H-class signals
  - Pareto-frontier closure that bound standalone macrorec doesn't necessarily
    bind a meta over a 200-dim component-prob feature space
  - Same fixed-bias blend at α=0.30 onto LB-best 3-stack architecture that
    produced LB 0.98094

Output paths suffixed `_metamacrorec_lam{lam}`. SMOKE not strictly needed
(meta is fast — ~5 min total wall on full data — but supported via SMOKE=1
env that subsets to 100k rows).
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from recipe_macrorecall import make_macrorec_obj, macrorec_eval_metric  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    EXCLUDE, _normed, build_lbbest_stack, iso_cal, load_pool, BIAS,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
SMOKE = os.environ.get("SMOKE") == "1"
LAM_CE = float(os.environ.get("MR_LAMBDA", "0.3"))
TEMPERATURE = float(os.environ.get("MR_T", "1.0"))
SUFFIX = f"_metamacrorec_lam{LAM_CE:g}".replace(".", "")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def main():
    log(f"N1 macro-recall meta-stacker  lam_ce={LAM_CE}  T={TEMPERATURE}  suffix={SUFFIX}")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best stack OOF = {bal(lb_oof, y):.5f}")

    log("loading pool")
    pool = load_pool(y)
    log(f"  {len(pool)} 3-class components loaded")

    log("constructing meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float32)
    log(f"  meta-feature shape: {X_tr.shape}")

    xgb_params = dict(
        num_class=3, tree_method="hist",
        learning_rate=0.05, max_depth=4,
        min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        verbosity=0, seed=SEED, nthread=-1,
        disable_default_eval_metric=1,
    )
    max_rounds = 3000
    es_rounds = 200

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(train), 3), dtype=np.float32)
    test_meta_folds = []
    best_iters = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y), 1):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        obj = make_macrorec_obj(y[tr_idx], n_classes=3,
                                temperature=TEMPERATURE, lam_ce=LAM_CE)
        feval = macrorec_eval_metric(y[va_idx])
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=max_rounds,
            obj=obj, custom_metric=feval, maximize=False,
            evals=[(dva, "val")], early_stopping_rounds=es_rounds,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        # Predict raw margins → softmax at T=1 for canonical posterior
        z_va = booster.predict(dva, iteration_range=(0, bi + 1),
                               output_margin=True).reshape(-1, 3)
        z_te = booster.predict(dte, iteration_range=(0, bi + 1),
                               output_margin=True).reshape(-1, 3)
        def softmax(z):
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        vp = softmax(z_va).astype(np.float32)
        tp = softmax(z_te).astype(np.float32)
        oof_meta[va_idx] = vp
        test_meta_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(argmax_bal)
        log(f"  fold {fold}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    oof_path = ART / f"oof_xgb_metastack{SUFFIX}.npy"
    test_path = ART / f"test_xgb_metastack{SUFFIX}.npy"
    np.save(oof_path, oof_meta)
    np.save(test_path, test_meta)
    log(f"saved {oof_path.name} + test")

    overall_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    overall_tuned = bal(oof_meta, y)
    log(f"\nOOF argmax = {overall_argmax:.5f}  @recipe-bias = {overall_tuned:.5f}")
    log(f"  best_iters = {best_iters}")

    # Iso-calibrate vs y for compatibility with downstream blend gates
    oof_iso, test_iso = iso_cal(oof_meta, test_meta, y)
    iso_path_o = ART / f"oof_xgb_metastack{SUFFIX}_iso.npy"
    iso_path_t = ART / f"test_xgb_metastack{SUFFIX}_iso.npy"
    np.save(iso_path_o, oof_iso)
    np.save(iso_path_t, test_iso)
    iso_argmax = balanced_accuracy_score(y, oof_iso.argmax(1))
    iso_tuned = bal(oof_iso, y)
    log(f"  iso  argmax = {iso_argmax:.5f}  @recipe-bias = {iso_tuned:.5f}")

    # Quick blend gate vs LB-best 3-stack (for monitoring; full 4-gate
    # will run via blend_gate_4gate.py with --candidate metamacrorec_lam03)
    log("\nQuick blend sweep onto LB-best 3-stack (fixed recipe bias):")
    log(f"  {'α':>6}  {'OOF':>8}  {'Δ(LB3)':>9}  {'errs':>6}")
    lb3_o, _ = build_lbbest_stack(y)
    base_bal = bal(lb3_o, y)
    log(f"  {'(LB3)':>6}  {base_bal:.5f}  ----")
    for alpha in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        b = log_blend([lb3_o, oof_iso], np.array([1.0 - alpha, alpha]))
        bb = bal(b, y)
        p = (np.log(np.clip(b, 1e-12, 1)) + BIAS).argmax(1)
        errs = int((p != y).sum())
        log(f"  {alpha:>6.2f}  {bb:.5f}  {bb - base_bal:+.5f}  {errs:>6}")

    summary = dict(
        n_folds=N_FOLDS, smoke=SMOKE,
        lam_ce=LAM_CE, temperature=TEMPERATURE,
        n_components=len(pool),
        meta_feature_shape=list(X_tr.shape),
        fold_scores_argmax=[float(s) for s in fold_scores],
        best_iters=[int(b) for b in best_iters],
        overall_argmax_bal_acc=float(overall_argmax),
        overall_tuned_bal_acc=float(overall_tuned),
        iso_argmax_bal_acc=float(iso_argmax),
        iso_tuned_bal_acc=float(iso_tuned),
    )
    with open(ART / f"xgb_metastack{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nwrote results JSON  total_wall={time.time() - 0:.0f}s")


if __name__ == "__main__":
    main()
