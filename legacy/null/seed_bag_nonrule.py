"""Seed-bag XGB-nonrule: 5 seeds, averaged OOF + test preds.

The non-rule-features-only XGB (seed=42, LB +0.00056 over greedy) is
the only architecturally-diverse leg in our stack we haven't bagged.
Expected lift: +0.00005 - 0.0002 OOF (variance reduction at the
non-rule level).

Protocol:
  1. Train 5 XGBs with seeds [42, 7, 123, 2024, 9999]. Same 5-fold split
     (seed=42) and same HPs as nonrule_features_only.py — only xgb_params
     seed changes.
  2. Average OOFs + test preds in prob space.
  3. Fixed-greedy-bias log-blend sweep vs greedy AND vs greedy+nonrule-seed42
     (current LB-best) to see if bagging adds anything beyond the single seed.

Artefacts:
  scripts/artifacts/oof_xgb_nonrule_bag.npy
  scripts/artifacts/test_xgb_nonrule_bag.npy
  scripts/artifacts/seed_bag_nonrule_results.json
  submissions/submission_greedy_nonrule_bag_blend.csv (if lift)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42  # split seed — pinned for OOF alignment
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {ID, TARGET}

SEEDS = [42, 7, 123, 2024, 9999]

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tune_log_bias(oof, y, prior):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            bals = []
            for g in grid:
                base[k] = bias[k] + g
                bals.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(bals))
            if bals[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = bals[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def train_one_seed(X, X_test, y, cat_cols, seed):
    xgb_params = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=seed,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"    [seed {seed}] fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc={bal:.5f}  ({time.time()-t0:.1f}s)")
    return oof, test_pred


def main():
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"non-rule features ({len(nonrule_cols)}): {nonrule_cols}")

    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mp = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mp).astype("int32").astype("category")
        X_test[c] = te[c].map(mp).astype("int32").astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # Per-seed training
    per_seed = []
    oof_bag = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
    test_bag = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
    for i, s in enumerate(SEEDS):
        log(f"--- seed {s} ({i+1}/{len(SEEDS)}) ---")
        t0 = time.time()
        oof_s, test_s = train_one_seed(X, X_test, y, cat_cols, seed=s)
        argmax_bal = balanced_accuracy_score(y, oof_s.argmax(axis=1))
        _, tuned_bal = tune_log_bias(oof_s, y, prior)
        log(f"  seed {s}  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}  "
            f"({time.time()-t0:.0f}s)")
        per_seed.append({
            "seed": s, "argmax_bal_acc": float(argmax_bal),
            "tuned_bal_acc": float(tuned_bal),
        })
        oof_bag += oof_s / len(SEEDS)
        test_bag += test_s / len(SEEDS)

    np.save(ART / "oof_xgb_nonrule_bag.npy", oof_bag)
    np.save(ART / "test_xgb_nonrule_bag.npy", test_bag)

    # Bag standalone diagnostics
    bag_argmax = balanced_accuracy_score(y, oof_bag.argmax(axis=1))
    _, bag_tuned = tune_log_bias(oof_bag, y, prior)
    log(f"seed-bag standalone  argmax={bag_argmax:.5f}  tuned={bag_tuned:.5f}")

    # ---- Blend vs greedy with FIXED bias ----
    log("loading greedy + its fitted bias")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = float(greedy_res["greedy_tuned_oof"])

    # Load single-seed nonrule for comparison (current LB-best blend uses this)
    oof_nonrule_s42 = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule_s42 = np.load(ART / "test_xgb_nonrule.npy")

    # Reference: greedy + single-seed nonrule at alpha=0.15 (LB-best)
    oof_lbbest = log_blend2(oof_nonrule_s42, oof_greedy, 0.15)
    test_lbbest = log_blend2(test_nonrule_s42, test_greedy, 0.15)
    lbbest_ba = balanced_accuracy_score(y,
        (np.log(np.clip(oof_lbbest, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    log(f"LB-best ref: greedy + single-seed nonrule @ alpha=0.15  OOF = {lbbest_ba:.5f}")

    log("sweep: greedy + BAG at various alphas (fixed greedy bias)")
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    sweep_vs_greedy = []
    best_vs_greedy = {"alpha": 0.0, "oof": tuned_greedy, "delta_vs_greedy": 0.0,
                      "delta_vs_lbbest": tuned_greedy - lbbest_ba}
    for alpha in grid:
        if alpha == 0.0:
            blend_oof = oof_greedy
        else:
            blend_oof = log_blend2(oof_bag, oof_greedy, alpha)
        lp = np.log(np.clip(blend_oof, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta_gr = ba - tuned_greedy
        delta_lb = ba - lbbest_ba
        marker = ""
        if ba > best_vs_greedy["oof"]:
            best_vs_greedy = {"alpha": alpha, "oof": float(ba),
                              "delta_vs_greedy": float(delta_gr),
                              "delta_vs_lbbest": float(delta_lb)}
            marker = "  <- best"
        sweep_vs_greedy.append({"alpha": alpha, "oof": float(ba),
                                "delta_vs_greedy": float(delta_gr),
                                "delta_vs_lbbest": float(delta_lb)})
        log(f"  alpha_bag={alpha:.2f}  OOF={ba:.5f}  "
            f"Δ_greedy={delta_gr:+.5f}  Δ_lbbest={delta_lb:+.5f}{marker}")

    log(f"best vs greedy+bag blend: alpha={best_vs_greedy['alpha']} "
        f"OOF={best_vs_greedy['oof']:.5f}  "
        f"Δ_lbbest={best_vs_greedy['delta_vs_lbbest']:+.5f}")

    # Decision: only emit submission if this blend OOF beats the LB-best
    # reference (greedy+single-nonrule@0.15 = 0.97421)
    results = {
        "seeds": SEEDS,
        "per_seed": per_seed,
        "bag_standalone_argmax": float(bag_argmax),
        "bag_standalone_tuned": float(bag_tuned),
        "greedy_tuned_oof": tuned_greedy,
        "greedy_bias": bias_greedy.tolist(),
        "lbbest_reference_oof": float(lbbest_ba),
        "sweep_vs_greedy": sweep_vs_greedy,
        "best_vs_greedy": best_vs_greedy,
    }

    if best_vs_greedy["delta_vs_lbbest"] < 1e-5:
        log("NO LIFT vs LB-best — null result")
        results["action"] = "no_submission"
    elif best_vs_greedy["delta_vs_lbbest"] < 3e-4:
        log(f"Δ_lbbest = {best_vs_greedy['delta_vs_lbbest']:+.5f} below +0.0003 "
            "threshold — borderline")
        a = best_vs_greedy["alpha"]
        bl = log_blend2(test_bag, test_greedy, a) if a > 0 else test_greedy
        lp = np.log(np.clip(bl, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_nonrule_bag_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote borderline {sub}")
        results["action"] = "borderline_no_submit"
        results["submission_path"] = str(sub)
    else:
        a = best_vs_greedy["alpha"]
        bl = log_blend2(test_bag, test_greedy, a) if a > 0 else test_greedy
        lp = np.log(np.clip(bl, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_nonrule_bag_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote {sub}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub)

    with open(ART / "seed_bag_nonrule_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/seed_bag_nonrule_results.json")


if __name__ == "__main__":
    main()
