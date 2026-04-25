"""Tier 1b #3: xgb_nonrule 3-seed bag + drop-in to LB-best 3-stack.

Keep fold_seed=42 for OOF alignment (per 2026-04-22 Session B rule that
fold-seed bagging regresses LB). Vary XGB training seed across {42, 7, 123};
average OOF+test probs across seeds (prob-space mean, equivalent to
log-blend with equal weights at high-prob limits).

Replace xgb_nonrule with xgb_nonrule_bag3 in the LB-best 3-stack and
report:
  - bag standalone OOF vs single-seed xgb_nonrule
  - stack OOF with bag replacing single-seed nonrule
  - Jaccard + error counts

If stack OOF lifts ≥ +1e-4, emit a submission candidate.
"""
from __future__ import annotations

import json
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
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
TARGET = "Irrigation_Need"
ID = "id"
RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {ID, TARGET}
FOLD_SEED = 42
XGB_SEEDS = [42, 7, 123]
N_FOLDS = 5
BIAS = np.array([1.4324, 1.4689, 3.4008])


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def train_one_seed(X, X_te, y, xgb_seed):
    log(f"  XGB_SEED={xgb_seed}")
    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05, max_depth=7, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", enable_categorical=True,
        verbosity=0, seed=xgb_seed,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test = np.zeros((X_te.shape[0], 3), dtype=np.float64)
    dte = xgb.DMatrix(X_te, enable_categorical=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=0,
        )
        bi = booster.best_iteration
        vp = booster.predict(dva, iteration_range=(0, bi + 1))
        oof[va_idx] = vp
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test += tp / N_FOLDS
        log(f"    fold {fold+1}/{N_FOLDS} it={bi}  wall={time.time()-t0:.1f}s")
    return oof, test


def main():
    t0 = time.time()
    log("loading data")
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"non-rule features: {len(nonrule_cols)}")

    X = tr[nonrule_cols].copy()
    X_te = te[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_te[c] = te[c].map(mapping).astype("int32").astype("category")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    # Train each seed, but reuse seed=42 from existing artefact
    oofs, tests = [], []
    existing_o = ART / "oof_xgb_nonrule.npy"
    existing_t = ART / "test_xgb_nonrule.npy"
    if existing_o.exists() and existing_t.exists():
        log(f"reusing existing seed=42 artefact")
        oofs.append(np.load(existing_o).astype(np.float64))
        tests.append(np.load(existing_t).astype(np.float64))
        seeds_done = [42]
    else:
        seeds_done = []

    for s in [s for s in XGB_SEEDS if s not in seeds_done]:
        o, t = train_one_seed(X, X_te, y, s)
        oofs.append(o); tests.append(t)
        seeds_done.append(s)

    # Prob-space mean bag
    bag_oof = np.mean(oofs, axis=0)
    bag_test = np.mean(tests, axis=0)
    bag_oof = _normed(bag_oof.astype(np.float32))
    bag_test = _normed(bag_test.astype(np.float32))
    np.save(ART / "oof_xgb_nonrule_bag3.npy", bag_oof)
    np.save(ART / "test_xgb_nonrule_bag3.npy", bag_test)
    log(f"bag saved: oof_xgb_nonrule_bag3.npy  seeds={seeds_done}")

    # Evaluate bag vs single-seed
    log(f"\n=== standalone OOF (on non-rule features) ===")
    log(f"single seed=42 @ recipe bias: {bal(_normed(oofs[0].astype(np.float32)), y):.5f}")
    for i, (o, s) in enumerate(zip(oofs[1:], seeds_done[1:])):
        log(f"single seed={s}  @ recipe bias: {bal(_normed(o.astype(np.float32)), y):.5f}")
    log(f"bag ({len(oofs)} seeds) @ recipe bias: {bal(bag_oof, y):.5f}")

    # Reconstruct LB-best 3-stack with bag-replaced nonrule
    log(f"\n=== LB-best stack with bag-replaced xgb_nonrule leg ===")
    r_o = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    r_t = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1_o = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1_t = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7_o = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7_t = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm_o = _normed(np.load(ART / "oof_realmlp.npy"))
    rm_t = _normed(np.load(ART / "test_realmlp.npy"))

    def build_stack(nr_o, nr_t):
        nr_iso_o, nr_iso_t = iso_cal(nr_o, nr_t, y)
        w3 = np.array([0.25, 0.35, 0.40])
        lb3_o = log_blend([r_o, s1_o, s7_o], w3)
        lb3_t = log_blend([r_t, s1_t, s7_t], w3)
        st1_o = log_blend([lb3_o, rm_o], np.array([0.8, 0.2]))
        st1_t = log_blend([lb3_t, rm_t], np.array([0.8, 0.2]))
        st2_o = log_blend([st1_o, nr_iso_o], np.array([0.925, 0.075]))
        st2_t = log_blend([st1_t, nr_iso_t], np.array([0.925, 0.075]))
        return st2_o, st2_t

    single_o = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    single_t = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    stack_single_o, stack_single_t = build_stack(single_o, single_t)
    stack_bag_o, stack_bag_t = build_stack(bag_oof, bag_test)

    b_single = bal(stack_single_o, y)
    b_bag = bal(stack_bag_o, y)
    errs_single = int((np.log(np.clip(stack_single_o, 1e-12, 1))
                       + BIAS).argmax(1).__ne__(y).sum())
    errs_bag = int((np.log(np.clip(stack_bag_o, 1e-12, 1))
                    + BIAS).argmax(1).__ne__(y).sum())
    delta = b_bag - b_single
    log(f"stack with single seed=42 nonrule: {b_single:.5f}  errs={errs_single}")
    log(f"stack with bag-{len(oofs)} nonrule:  {b_bag:.5f}  errs={errs_bag}  "
        f"Δ={delta:+.5f}")

    # Also sweep α_nonrule at bag level: maybe bag supports higher α
    log(f"\n=== bag α sweep (α_nonrule_iso) ===")
    nr_iso_o, nr_iso_t = iso_cal(bag_oof, bag_test, y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r_o, s1_o, s7_o], w3)
    st1_o = log_blend([lb3_o, rm_o], np.array([0.8, 0.2]))
    alphas = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25]
    sweep = []
    for a in alphas:
        b = log_blend([st1_o, nr_iso_o], np.array([1 - a, a]))
        bv = bal(b, y)
        sweep.append(dict(alpha=a, oof=float(bv)))
        print(f"  α={a:.3f}  OOF={bv:.5f}  Δ={bv - b_single:+.5f}")
    best_sweep = max(sweep, key=lambda r: r["oof"])
    log(f"best bag-α: {best_sweep}")

    out = dict(
        seeds=seeds_done,
        single_seed42_bag_std_oof=[float(bal(_normed(o.astype(np.float32)), y))
                                    for o in oofs],
        bag_std_oof=float(bal(bag_oof, y)),
        stack_single_oof=float(b_single), stack_single_errs=errs_single,
        stack_bag_oof=float(b_bag), stack_bag_errs=errs_bag,
        stack_delta=float(delta),
        bag_alpha_sweep=sweep, best_sweep=best_sweep,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "tier1b_nonrule_bag_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote scripts/artifacts/tier1b_nonrule_bag_results.json")

    if delta >= 1e-4:
        pred = (np.log(np.clip(stack_bag_t, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred]
        path = SUB / "submission_tier1b_stack_nonrule_bag3.csv"
        sub.to_csv(path, index=False)
        log(f"wrote {path}  (Δ={delta:+.5f} ≥ +1e-4)")
    else:
        log(f"Δ={delta:+.5f} below +1e-4 gate; no submission")


if __name__ == "__main__":
    main()
