"""Nonrule + rule_pred (and dgp_score) as explicit features.

Tests whether augmenting the 13 non-rule features with the rule's
outputs (rule_pred categorical, dgp_score numeric) adds orthogonal
bits. Rationale: the rule-aware nonrule model can learn joint
corrections like 'if rule says Low AND Humidity is high AND
Previous_Irrigation_mm > X → predict Medium', which pure non-rule
can't express and greedy (rule-feature-dominated) won't because it
underweights the non-rule features.

Risk: if the model simply learns to parrot rule_pred, predictions
will be redundant with greedy and blending will fail like binhigh.
The fixed-bias sweep gives an honest answer.

Features:
  13 non-rule: Soil_Type, Soil_pH, Organic_Carbon,
    Electrical_Conductivity, Humidity, Sunlight_Hours, Crop_Type,
    Season, Irrigation_Type, Water_Source, Field_Area_hectare,
    Previous_Irrigation_mm, Region
  +2 rule-derived: rule_pred (categorical 0/1/2), dgp_score (int 0-9)

Protocol: same 5-fold stratified split (seed=42), fixed-greedy-bias
sweep. LB-probe only if fixed-bias OOF lifts >= +0.0003 vs base
(greedy + XGB-nonrule @ alpha=0.15, OOF 0.97421).
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


def compute_rule(df: pd.DataFrame):
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
    rule_pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    return rule_pred, score


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

    rule_tr, score_tr = compute_rule(tr)
    rule_te, score_te = compute_rule(te)

    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"non-rule cols ({len(nonrule_cols)}): {nonrule_cols}")

    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")

    # Add rule_pred (categorical) + dgp_score (numeric).
    X["rule_pred"] = rule_tr.astype("int32")
    X["rule_pred"] = X["rule_pred"].astype("category")
    X_test["rule_pred"] = rule_te.astype("int32")
    X_test["rule_pred"] = X_test["rule_pred"].astype("category")
    X["dgp_score"] = score_tr.astype(np.float32)
    X_test["dgp_score"] = score_te.astype(np.float32)

    feat_cols = nonrule_cols + ["rule_pred", "dgp_score"]
    log(f"total features: {len(feat_cols)} (+ rule_pred cat + dgp_score num)")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    log("training 5-fold 3-class XGB (non-rule + rule_pred + dgp_score)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_pred = np.zeros((len(te), 3), dtype=np.float64)

    xgb_params = dict(
        objective="multi:softprob",
        num_class=3,
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
            f"argmax={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"nonrule+rule standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")
    np.save(ART / "oof_xgb_nonrule_rulepred.npy", oof)
    np.save(ART / "test_xgb_nonrule_rulepred.npy", test_pred)

    # Load greedy + XGB-nonrule.
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_xgb_nr = np.load(ART / "oof_xgb_nonrule.npy")
    test_xgb_nr = np.load(ART / "test_xgb_nonrule.npy")
    greedy_res = json.loads(Path(ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = greedy_res["greedy_tuned_oof"]

    base = log_blend_n([oof_xgb_nr, oof_greedy], [0.15, 0.85])
    base_test = log_blend_n([test_xgb_nr, test_greedy], [0.15, 0.85])
    base_ba = balanced_accuracy_score(
        y, (np.log(np.clip(base, 1e-9, 1.0)) + bias_greedy).argmax(1)
    )
    log(f"greedy OOF = {tuned_greedy:.5f}")
    log(f"base (greedy + XGB-nonrule 0.15) OOF = {base_ba:.5f}")

    results = {
        "standalone_argmax": float(argmax_bal),
        "standalone_tuned":  float(tuned_bal),
        "greedy_tuned_oof":  tuned_greedy,
        "base_blend_oof":    float(base_ba),
        "greedy_bias":       bias_greedy.tolist(),
        "sweeps": {},
    }

    # (a) Alone onto greedy.
    log("sweep (a): nonrule+rule onto greedy (fixed bias)")
    sa = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b = log_blend_n([oof, oof_greedy], [alpha, 1 - alpha])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(1)
        )
        sa.append({"alpha": alpha, "oof": float(ba),
                   "delta_vs_greedy": float(ba - tuned_greedy)})
        log(f"  alpha={alpha:.2f}  OOF = {ba:.5f}  Δ greedy = {ba - tuned_greedy:+.5f}")
    results["sweeps"]["alone_onto_greedy"] = sa

    # (b) Stacked onto base.
    log("sweep (b): nonrule+rule stacked onto base (fixed bias)")
    sb = []
    for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        b = log_blend_n([oof, base], [beta, 1 - beta])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(1)
        )
        sb.append({"beta": beta, "oof": float(ba),
                   "delta_vs_base": float(ba - base_ba)})
        log(f"  beta={beta:.2f}  OOF = {ba:.5f}  Δ base = {ba - base_ba:+.5f}")
    results["sweeps"]["onto_base"] = sb

    # (c) 3-way: XGB-nonrule + nonrule+rule + greedy.
    log("sweep (c): XGB-nonrule + nonrule+rule + greedy (fixed bias)")
    sc = []
    best_c = {"oof": -1.0}
    for alpha in [0.05, 0.10, 0.15]:
        for beta in [0.05, 0.10, 0.15, 0.20, 0.25]:
            w_g = 1 - alpha - beta
            if w_g < 0.4:
                continue
            b = log_blend_n([oof_xgb_nr, oof, oof_greedy], [alpha, beta, w_g])
            ba = balanced_accuracy_score(
                y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(1)
            )
            entry = {"alpha_xgb": alpha, "beta_new": beta, "w_greedy": w_g,
                     "oof": float(ba),
                     "delta_vs_base": float(ba - base_ba)}
            sc.append(entry)
            log(f"  a_xgb={alpha:.2f} b_new={beta:.2f} w_gr={w_g:.2f}  "
                f"OOF = {ba:.5f}  Δ base = {ba - base_ba:+.5f}")
            if ba > best_c["oof"]:
                best_c = entry
    results["sweeps"]["three_way"] = sc
    results["best_2d"] = best_c

    # Agreement check: how close is this to XGB-nonrule alone?
    jaccard_errors = None
    err_new = (np.log(np.clip(oof, 1e-9, 1.0)) + bias_greedy).argmax(1) != y
    err_nr = (np.log(np.clip(oof_xgb_nr, 1e-9, 1.0)) + bias_greedy).argmax(1) != y
    inter = (err_new & err_nr).sum()
    union = (err_new | err_nr).sum()
    jaccard_errors = float(inter / max(union, 1))
    log(f"error Jaccard (new vs xgb-nonrule) = {jaccard_errors:.4f}  "
        f"intersect={inter} union={union}")
    results["error_jaccard_vs_xgb_nonrule"] = jaccard_errors

    best_a = max(sa, key=lambda d: d["oof"])
    best_b = max(sb, key=lambda d: d["oof"])
    log(f"\nbest onto greedy:  α={best_a['alpha']:.2f}  OOF={best_a['oof']:.5f}  "
        f"Δ greedy={best_a['delta_vs_greedy']:+.5f}")
    log(f"best onto base:    β={best_b['beta']:.2f}  OOF={best_b['oof']:.5f}  "
        f"Δ base={best_b['delta_vs_base']:+.5f}")
    log(f"best 3-way:        a={best_c.get('alpha_xgb',0):.2f} "
        f"b={best_c.get('beta_new',0):.2f} g={best_c.get('w_greedy',0):.2f}  "
        f"OOF={best_c['oof']:.5f}  Δ base={best_c.get('delta_vs_base',0):+.5f}")

    cands = []
    if best_b["delta_vs_base"] >= 3e-4:
        cands.append(("onto_base", best_b,
                      log_blend_n([test_pred, base_test],
                                  [best_b["beta"], 1 - best_b["beta"]])))
    if best_c.get("delta_vs_base", 0) >= 3e-4:
        cands.append(("three_way", best_c, log_blend_n(
            [test_xgb_nr, test_pred, test_greedy],
            [best_c["alpha_xgb"], best_c["beta_new"], best_c["w_greedy"]]
        )))
    if best_a["delta_vs_greedy"] >= 3e-4 and best_a["oof"] > base_ba + 3e-4:
        cands.append(("alone_onto_greedy", best_a,
                      log_blend_n([test_pred, test_greedy],
                                  [best_a["alpha"], 1 - best_a["alpha"]])))

    if cands:
        kind, info, test_blend = max(cands, key=lambda t: t[1]["oof"])
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(1)
        sub_path = OUT / f"submission_greedy_nonrule_rulepred_{kind}.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}")
        results["action"] = f"ready_to_submit_{kind}"
        results["submission_path"] = str(sub_path)
    else:
        log("no OOF lift clears +0.0003 threshold — no submission")
        results["action"] = "no_submission"

    with open(ART / "nonrule_rulepred_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/nonrule_rulepred_results.json")


if __name__ == "__main__":
    main()
