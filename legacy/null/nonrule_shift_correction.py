"""Brainstorm #8: two-stage rule-base + non-rule correction.

Train a 5-class ordinal-shift predictor (`shift = y - rule_pred + 2
in {0..4}`) on non-rule features only. The rule's deterministic
prediction forms the base; the model learns the signed ordinal
correction the NN generator applied. Convert shift probs to 3-class
y probs via the rule_pred -> y offset map, then log-blend into greedy
with fixed bias (same methodology as #7).

Expected mechanism: predicting the shift directly concentrates the
model's capacity on the NN-perturbation signal instead of diluting
it across the overwhelming rule-correct majority.

Artefacts:
  scripts/artifacts/oof_xgb_shift5.npy   # 5-class shift probs (N, 5)
  scripts/artifacts/test_xgb_shift5.npy
  scripts/artifacts/oof_xgb_shift_to_y.npy  # converted 3-class probs
  scripts/artifacts/test_xgb_shift_to_y.npy
  scripts/artifacts/shift_results.json
  submissions/submission_greedy_shift_blend.csv (if lift)
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
RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {ID, TARGET}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_rule_pred(df: pd.DataFrame) -> np.ndarray:
    sm = df["Soil_Moisture"].astype(float).values
    rf = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage = df["Crop_Growth_Stage"].astype(str).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    kc = np.where(np.isin(stage, ACTIVE_STAGES), 2, 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    return np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)


def shift5_to_y3(p_shift: np.ndarray, rule_pred: np.ndarray) -> np.ndarray:
    """Convert 5-class shift probs -> 3-class y probs via rule offset map."""
    n = p_shift.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    # shift in {-2, -1, 0, +1, +2} (columns 0..4)
    # y = clip(rule_pred + shift, 0, 2)
    for rp in (0, 1, 2):
        mask = rule_pred == rp
        if not mask.any():
            continue
        ps = p_shift[mask]
        if rp == 0:
            out[mask, 0] = ps[:, 0] + ps[:, 1] + ps[:, 2]
            out[mask, 1] = ps[:, 3]
            out[mask, 2] = ps[:, 4]
        elif rp == 1:
            out[mask, 0] = ps[:, 0] + ps[:, 1]
            out[mask, 1] = ps[:, 2]
            out[mask, 2] = ps[:, 3] + ps[:, 4]
        else:
            out[mask, 0] = ps[:, 0]
            out[mask, 1] = ps[:, 1]
            out[mask, 2] = ps[:, 2] + ps[:, 3] + ps[:, 4]
    out /= out.sum(1, keepdims=True)
    return out


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

    rule_tr = compute_rule_pred(tr)
    rule_te = compute_rule_pred(te)
    log(f"rule_pred train dist: "
        f"Low {(rule_tr==0).mean():.4f}  Medium {(rule_tr==1).mean():.4f}  "
        f"High {(rule_tr==2).mean():.4f}")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    shift = (y - rule_tr + 2).astype(np.int32)  # in {0..4}
    prior = np.bincount(y) / len(y)

    uniq, counts = np.unique(shift, return_counts=True)
    log(f"shift dist (target): "
        f"{dict(zip(uniq.tolist(), (counts/len(shift)).round(5).tolist()))}")

    # Feature split.
    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"non-rule features: {nonrule_cols}")

    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")

    log("training 5-fold 5-class shift XGB on non-rule features")
    # NB: stratify on y (not shift) to keep fold alignment with other OOFs.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof5 = np.zeros((len(tr), 5), dtype=np.float64)
    test5 = np.zeros((len(te), 5), dtype=np.float64)

    xgb_params = dict(
        objective="multi:softprob",
        num_class=5,
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
    dte = xgb.DMatrix(X_test, enable_categorical=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=shift[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=shift[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof5[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test5 += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        # Evaluate via conversion to y-probs.
        oof_y = shift5_to_y3(oof5[va_idx], rule_tr[va_idx])
        fold_bal = balanced_accuracy_score(y[va_idx], oof_y.argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"y-argmax bal={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    log("converting shift probs -> y probs via rule_pred offset")
    oof_y = shift5_to_y3(oof5, rule_tr)
    test_y = shift5_to_y3(test5, rule_te)

    argmax_bal = balanced_accuracy_score(y, oof_y.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof_y, y, prior)
    log(f"shift->y  argmax={argmax_bal:.5f}  tuned_standalone={tuned_bal:.5f}")
    np.save(ART / "oof_xgb_shift5.npy", oof5)
    np.save(ART / "test_xgb_shift5.npy", test5)
    np.save(ART / "oof_xgb_shift_to_y.npy", oof_y)
    np.save(ART / "test_xgb_shift_to_y.npy", test_y)

    log("loading greedy blend + its fitted bias")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads(Path(ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = greedy_res["greedy_tuned_oof"]
    log(f"greedy baseline tuned OOF = {tuned_greedy:.5f}")

    results = {
        "standalone_argmax": float(argmax_bal),
        "standalone_tuned":  float(tuned_bal),
        "greedy_tuned_oof":  tuned_greedy,
        "greedy_bias":       bias_greedy.tolist(),
        "shift_class_dist":  dict(zip(uniq.tolist(), (counts/len(shift)).tolist())),
        "sweep_log_blend":   [],
    }

    log("sweep: log-blend alpha (shift-derived weight), fixed greedy bias")
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for alpha in grid:
        blend = log_blend2(oof_y, oof_greedy, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta = ba - tuned_greedy
        results["sweep_log_blend"].append({"alpha": alpha, "oof": float(ba),
                                           "delta_vs_greedy": float(delta)})
        log(f"  alpha_shift={alpha:.2f}  OOF (fixed bias) = {ba:.5f}  Δ = {delta:+.5f}")

    # Also test stacking on top of greedy+nonrule (current LB best).
    log("stacking test: greedy + nonrule (alpha=0.15) + shift (sweep)")
    oof_nr = np.load(ART / "oof_xgb_nonrule.npy")
    test_nr = np.load(ART / "test_xgb_nonrule.npy")
    # First recreate greedy+nonrule at alpha=0.15.
    base_oof = log_blend2(oof_nr, oof_greedy, 0.15)
    base_test = log_blend2(test_nr, test_greedy, 0.15)
    base_ba = balanced_accuracy_score(
        y, (np.log(np.clip(base_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    )
    log(f"base (greedy + nonrule 0.15) tuned OOF = {base_ba:.5f}")

    results["base_nonrule_blend_oof"] = float(base_ba)
    results["sweep_on_top_of_nonrule_blend"] = []
    for alpha in grid:
        blend = log_blend2(oof_y, base_oof, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta = ba - base_ba
        results["sweep_on_top_of_nonrule_blend"].append(
            {"alpha": alpha, "oof": float(ba), "delta_vs_nonrule_blend": float(delta)}
        )
        log(f"  +shift alpha={alpha:.2f}  OOF = {ba:.5f}  Δ = {delta:+.5f}")

    best_onto_greedy = max(results["sweep_log_blend"],
                           key=lambda d: d["oof"])
    best_onto_nr = max(results["sweep_on_top_of_nonrule_blend"],
                       key=lambda d: d["oof"])
    log(f"\nbest onto greedy:     α={best_onto_greedy['alpha']:.2f}  "
        f"OOF={best_onto_greedy['oof']:.5f}  Δ={best_onto_greedy['delta_vs_greedy']:+.5f}")
    log(f"best onto nonrule:    α={best_onto_nr['alpha']:.2f}  "
        f"OOF={best_onto_nr['oof']:.5f}  Δ={best_onto_nr['delta_vs_nonrule_blend']:+.5f}")

    # Emit submission only if stacking on top of nonrule beats it by >= 0.0003.
    if best_onto_nr["delta_vs_nonrule_blend"] >= 3e-4:
        alpha = best_onto_nr["alpha"]
        blend = log_blend2(test_y, base_test, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / "submission_greedy_nonrule_shift_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}")
        results["action"] = "stack_on_nonrule_ready"
        results["submission_path"] = str(sub_path)
        results["best_stacked_alpha"] = alpha

        blend_oof = log_blend2(oof_y, base_oof, alpha)
        cm = confusion_matrix(
            y, (np.log(np.clip(blend_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        log(f"OOF confusion matrix (stacked):\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
    elif best_onto_greedy["delta_vs_greedy"] >= 3e-4:
        alpha = best_onto_greedy["alpha"]
        blend = log_blend2(test_y, test_greedy, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / "submission_greedy_shift_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}")
        results["action"] = "standalone_blend_only"
    else:
        log("no OOF lift stacked or standalone — no submission")
        results["action"] = "no_submission"

    with open(ART / "shift_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/shift_results.json")


if __name__ == "__main__":
    main()
