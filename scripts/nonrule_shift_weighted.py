"""Brainstorm #8 rescue: weighted shift training.

The vanilla shift model collapsed because 98.36% of rows have shift=0
and early stopping saturated on the majority class in <100 rounds.
This variant upweights shift != 0 rows by 100x so the training loss
gives flip rows equal gradient, forcing the model to learn shift
discrimination rather than parroting rule_pred.

Setup: 5-class multi:softprob on 13 non-rule features, same 5-fold
split on y (seed=42), sample_weight = 100 if shift != 2 else 1.
Fixed-greedy-bias sweep onto (a) greedy alone, (b) greedy + XGB-
nonrule@0.15 base.

LB-probe only if fixed-bias OOF lifts >= +0.0003 vs current base.
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
FLIP_WEIGHT = 100.0  # weight for shift != 0 rows

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
    n = p_shift.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
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


def log_blend_n(probs_list, weights):
    total = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        total = total + w * np.log(np.clip(p, 1e-9, 1.0))
    total -= total.max(1, keepdims=True)
    e = np.exp(total)
    return e / e.sum(1, keepdims=True)


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    rule_tr = compute_rule_pred(tr)
    rule_te = compute_rule_pred(te)
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    shift = (y - rule_tr + 2).astype(np.int32)  # 0-indexed shift in {0..4}
    prior = np.bincount(y) / len(y)
    is_flip = shift != 2
    sample_w = np.where(is_flip, FLIP_WEIGHT, 1.0).astype(np.float32)
    log(f"flip rows: {is_flip.sum()} ({is_flip.mean():.4f}); "
        f"flip weight = {FLIP_WEIGHT}")
    uniq, counts = np.unique(shift, return_counts=True)
    log(f"shift dist: {dict(zip(uniq.tolist(), (counts/len(shift)).round(5).tolist()))}")

    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]

    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")

    log(f"training 5-fold weighted 5-class shift XGB (flip w={FLIP_WEIGHT})")
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
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=shift[tr_idx],
                          weight=sample_w[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=shift[va_idx],
                          weight=sample_w[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof5[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test5 += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        oof_y = shift5_to_y3(oof5[va_idx], rule_tr[va_idx])
        fold_bal = balanced_accuracy_score(y[va_idx], oof_y.argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"y-argmax bal={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    oof_y = shift5_to_y3(oof5, rule_tr)
    test_y = shift5_to_y3(test5, rule_te)
    argmax_bal = balanced_accuracy_score(y, oof_y.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof_y, y, prior)
    log(f"shift-weighted->y  argmax={argmax_bal:.5f}  tuned_standalone={tuned_bal:.5f}")
    np.save(ART / "oof_xgb_shiftw_to_y.npy", oof_y)
    np.save(ART / "test_xgb_shiftw_to_y.npy", test_y)
    np.save(ART / "oof_xgb_shiftw5.npy", oof5)
    np.save(ART / "test_xgb_shiftw5.npy", test5)

    log("loading greedy + XGB-nonrule OOFs for blend sweeps")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_xgb = np.load(ART / "oof_xgb_nonrule.npy")
    test_xgb = np.load(ART / "test_xgb_nonrule.npy")
    greedy_res = json.loads(Path(ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = greedy_res["greedy_tuned_oof"]

    base = log_blend_n([oof_xgb, oof_greedy], [0.15, 0.85])
    base_test = log_blend_n([test_xgb, test_greedy], [0.15, 0.85])
    lp = np.log(np.clip(base, 1e-9, 1.0))
    base_ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
    log(f"greedy alone OOF = {tuned_greedy:.5f}")
    log(f"baseline (greedy + XGB-nonrule 0.15) OOF = {base_ba:.5f}")

    results = {
        "flip_weight": FLIP_WEIGHT,
        "shiftw_standalone_argmax": float(argmax_bal),
        "shiftw_standalone_tuned": float(tuned_bal),
        "greedy_tuned_oof": tuned_greedy,
        "base_xgb_nonrule_blend_oof": float(base_ba),
        "greedy_bias": bias_greedy.tolist(),
        "sweeps": {},
    }

    log("sweep (a): weighted-shift onto greedy alone (fixed bias)")
    sa = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b = log_blend_n([oof_y, oof_greedy], [alpha, 1 - alpha])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        sa.append({"alpha": alpha, "oof": float(ba),
                   "delta_vs_greedy": float(ba - tuned_greedy)})
        log(f"  alpha_shiftw={alpha:.2f}  OOF = {ba:.5f}  "
            f"Δ greedy = {ba - tuned_greedy:+.5f}")
    results["sweeps"]["onto_greedy"] = sa

    log("sweep (b): weighted-shift stacked onto (greedy + XGB-nonrule@0.15)")
    sb = []
    for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        b = log_blend_n([oof_y, base], [beta, 1 - beta])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        sb.append({"beta": beta, "oof": float(ba),
                   "delta_vs_base": float(ba - base_ba)})
        log(f"  beta_shiftw={beta:.2f}  OOF = {ba:.5f}  Δ base = {ba - base_ba:+.5f}")
    results["sweeps"]["onto_base"] = sb

    best_a = max(sa, key=lambda d: d["oof"])
    best_b = max(sb, key=lambda d: d["oof"])
    log(f"best onto greedy:  α={best_a['alpha']:.2f}  OOF={best_a['oof']:.5f}  "
        f"Δ greedy={best_a['delta_vs_greedy']:+.5f}")
    log(f"best onto base:    β={best_b['beta']:.2f}  OOF={best_b['oof']:.5f}  "
        f"Δ base={best_b['delta_vs_base']:+.5f}")

    if best_b["delta_vs_base"] >= 3e-4:
        blend_test = log_blend_n([test_y, base_test], [best_b["beta"], 1 - best_b["beta"]])
        lp = np.log(np.clip(blend_test, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / "submission_greedy_nonrule_shiftw_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}")
        results["action"] = "stack_on_base_ready"
        results["submission_path"] = str(sub_path)
        results["best_beta"] = best_b["beta"]
    elif best_a["delta_vs_greedy"] >= 3e-4:
        blend_test = log_blend_n([test_y, test_greedy], [best_a["alpha"], 1 - best_a["alpha"]])
        lp = np.log(np.clip(blend_test, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / "submission_greedy_shiftw_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}")
        results["action"] = "onto_greedy_only"
    else:
        log("no OOF lift clears +0.0003 threshold — no submission")
        results["action"] = "no_submission"

    with open(ART / "nonrule_shiftw_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/nonrule_shiftw_results.json")


if __name__ == "__main__":
    main()
