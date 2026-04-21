"""Brainstorm #7: non-rule-features-only 3-class predictor + greedy blend.

Hypothesis: the host's label-generator NN perturbs the rule's output
using features NOT in the DGP (`Humidity, Prev_Irrig, EC, Soil_pH,
Organic_C, Sunlight, Field_Area, Region, Crop_Type, Soil_Type, Season,
Irrigation_Type, Water_Source, Soil_Type`). A model restricted to
these 13 features captures exactly that perturbation signal, then
blends in as architecturally orthogonal to LGBM-dist / XGB-dist which
are dominated by the 6 rule features.

Test protocol (learned from binhigh failure):
  1. Train 3-class XGB on non-rule features only, 5-fold stratified
     (same seed=42 as all other OOFs).
  2. Fixed-greedy-bias sweep over blend weight alpha (log-space mix).
     No per-alpha bias retune. If fixed-bias OOF doesn't lift, the
     component is redundant — abort.
  3. LB-probe best alpha only if fixed-bias OOF lifts >= +0.0005.

Artefacts:
  scripts/artifacts/oof_xgb_nonrule.npy
  scripts/artifacts/test_xgb_nonrule.npy
  scripts/artifacts/nonrule_results.json
  submissions/submission_greedy_nonrule_blend.csv (if lift)
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


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

# Rule features used by the 0.96097-ceiling DGP rule + Crop_Growth_Stage for Kc.
RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {ID, TARGET}

ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    logs = la + lb
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    # Partition columns.
    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"total non-id/target cols: {len(all_cols)}; non-rule: {len(nonrule_cols)}")
    log(f"non-rule features: {nonrule_cols}")

    # Build feature matrix.
    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]

    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {X.shape[1]} ({len(num_cols)} num + {len(cat_cols)} cat)")

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
        seed=SEED,
    )

    log("training 5-fold 3-class XGB on non-rule features only")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

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
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"non-rule standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")
    np.save(ART / "oof_xgb_nonrule.npy", oof)
    np.save(ART / "test_xgb_nonrule.npy", test_pred)

    # Blend into greedy with FIXED bias.
    log("loading greedy blend + its fitted bias")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads(Path(ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = greedy_res["greedy_tuned_oof"]
    log(f"greedy baseline tuned OOF = {tuned_greedy:.5f}  "
        f"bias = {bias_greedy.round(4).tolist()}")

    results = {
        "nonrule_standalone_argmax": float(argmax_bal),
        "nonrule_standalone_tuned":  float(tuned_bal),
        "greedy_tuned_oof": tuned_greedy,
        "greedy_bias": bias_greedy.tolist(),
        "sweep_log_blend": [],
    }

    log("sweep: log-blend alpha (nonrule weight), fixed greedy bias")
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for alpha in grid:
        blend = log_blend2(oof, oof_greedy, alpha)  # alpha on nonrule
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta = ba - tuned_greedy
        results["sweep_log_blend"].append({"alpha": alpha, "oof": float(ba),
                                           "delta_vs_greedy": float(delta)})
        log(f"  alpha_nonrule={alpha:.2f}  OOF (fixed bias) = {ba:.5f}  Δ = {delta:+.5f}")

    best = max(results["sweep_log_blend"], key=lambda d: d["oof"])
    best_alpha = best["alpha"]
    best_oof = best["oof"]
    best_delta = best["delta_vs_greedy"]
    log(f"best alpha={best_alpha}  OOF={best_oof:.5f}  Δ={best_delta:+.5f}")
    results["best"] = best

    if best_alpha == 0.0 or best_delta < 1e-5:
        log("no OOF lift from non-rule blend — no submission")
        results["action"] = "no_submission"
    elif best_delta < 5e-4:
        log(f"OOF lift {best_delta:+.5f} is below the 0.0005 LB-probe threshold; "
            "emit submission but flag as borderline.")
        blend = log_blend2(test_pred, test_greedy, best_alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / "submission_greedy_nonrule_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}  (borderline, do not auto-submit)")
        results["action"] = "borderline_no_submit"
    else:
        blend = log_blend2(test_pred, test_greedy, best_alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / "submission_greedy_nonrule_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub_path)

        blend_oof = log_blend2(oof, oof_greedy, best_alpha)
        cm = confusion_matrix(
            y, (np.log(np.clip(blend_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        log(f"OOF confusion matrix:\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    with open(ART / "nonrule_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/nonrule_results.json")


if __name__ == "__main__":
    main()
