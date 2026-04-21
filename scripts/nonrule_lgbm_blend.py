"""LGBM variant of the non-rule-features-only model + 3-way blend.

Tests whether LGBM's leaf-wise growth captures residual non-rule-feature
signal that XGB's level-wise trees missed. Same 13 non-rule features,
same 5-fold split (seed=42). Fixed-greedy-bias sweeps:
  (a) standalone LGBM-nonrule alpha onto greedy (sanity check)
  (b) 2D sweep: XGB-nonrule alpha + LGBM-nonrule beta onto greedy
  (c) 1D sweep: LGBM-nonrule beta onto greedy + XGB-nonrule-0.15 base

Methodology: fixed greedy bias, LB-probe only if (b) or (c) lifts
>= +0.0003 over the current best (greedy+XGB-nonrule at 0.15).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
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


def log_blend_n(probs_list, weights):
    """Log-space weighted blend, row-softmax."""
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

    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"non-rule features ({len(nonrule_cols)}): {nonrule_cols}")

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

    log("training 5-fold LGBM 3-class on non-rule features")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    params = dict(
        objective="multiclass",
        num_class=len(CLASSES),
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        verbose=-1,
        seed=SEED,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                          categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx],
                          categorical_feature=cat_cols, free_raw_data=False,
                          reference=dtr)
        booster = lgb.train(
            params, dtr, num_boost_round=4000,
            valid_sets=[dva], valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)],
        )
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(X.iloc[va_idx], num_iteration=bi)
        test_pred += booster.predict(X_test, num_iteration=bi) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"argmax bal={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"LGBM nonrule standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")
    np.save(ART / "oof_lgbm_nonrule.npy", oof)
    np.save(ART / "test_lgbm_nonrule.npy", test_pred)

    log("loading greedy + existing XGB-nonrule OOFs")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_xgb = np.load(ART / "oof_xgb_nonrule.npy")
    test_xgb = np.load(ART / "test_xgb_nonrule.npy")
    greedy_res = json.loads(Path(ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = greedy_res["greedy_tuned_oof"]

    # Current best baseline: greedy + XGB-nonrule @ 0.15
    base = log_blend_n([oof_xgb, oof_greedy], [0.15, 0.85])
    base_test = log_blend_n([test_xgb, test_greedy], [0.15, 0.85])
    lp = np.log(np.clip(base, 1e-9, 1.0))
    base_ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
    log(f"baseline (greedy + XGB-nonrule 0.15) OOF = {base_ba:.5f}")
    log(f"greedy alone OOF = {tuned_greedy:.5f}")

    results = {
        "lgbm_nonrule_standalone_argmax": float(argmax_bal),
        "lgbm_nonrule_standalone_tuned":  float(tuned_bal),
        "greedy_tuned_oof": tuned_greedy,
        "base_xgb_nonrule_blend_oof": float(base_ba),
        "greedy_bias": bias_greedy.tolist(),
        "sweeps": {},
    }

    # (a) LGBM-nonrule alone onto greedy, fixed bias.
    log("sweep (a): LGBM-nonrule alone into greedy (fixed bias)")
    sweep_a = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b = log_blend_n([oof, oof_greedy], [alpha, 1 - alpha])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        sweep_a.append({"alpha": alpha, "oof": float(ba),
                        "delta_vs_greedy": float(ba - tuned_greedy)})
        log(f"  alpha_lgbm={alpha:.2f}  OOF = {ba:.5f}  "
            f"Δ greedy = {ba - tuned_greedy:+.5f}")
    results["sweeps"]["lgbm_alone_onto_greedy"] = sweep_a

    # (b) 2D sweep: XGB alpha + LGBM beta + greedy.
    log("sweep (b): XGB_nr alpha + LGBM_nr beta onto greedy (fixed bias)")
    sweep_b = []
    best_b = {"oof": -1.0}
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        for beta in [0.05, 0.10, 0.15, 0.20]:
            w_g = 1 - alpha - beta
            if w_g < 0.3:
                continue
            b = log_blend_n([oof_xgb, oof, oof_greedy], [alpha, beta, w_g])
            ba = balanced_accuracy_score(
                y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
            )
            entry = {"alpha_xgb": alpha, "beta_lgbm": beta, "w_greedy": w_g,
                     "oof": float(ba),
                     "delta_vs_base": float(ba - base_ba)}
            sweep_b.append(entry)
            log(f"  a_xgb={alpha:.2f} b_lgbm={beta:.2f} w_gr={w_g:.2f}  "
                f"OOF = {ba:.5f}  Δ base = {ba - base_ba:+.5f}")
            if ba > best_b["oof"]:
                best_b = entry
    results["sweeps"]["xgb_plus_lgbm_onto_greedy"] = sweep_b
    results["best_2d"] = best_b

    # (c) 1D: LGBM beta onto (greedy + XGB@0.15) base.
    log("sweep (c): LGBM_nr beta stacked onto (greedy + XGB@0.15)")
    sweep_c = []
    for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        b = log_blend_n([oof, base], [beta, 1 - beta])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        sweep_c.append({"beta": beta, "oof": float(ba),
                        "delta_vs_base": float(ba - base_ba)})
        log(f"  beta_lgbm={beta:.2f}  OOF = {ba:.5f}  Δ base = {ba - base_ba:+.5f}")
    results["sweeps"]["lgbm_onto_base_xgb_blend"] = sweep_c

    best_c = max(sweep_c, key=lambda d: d["oof"])
    log(f"\nbest 2D (xgb+lgbm+greedy): a={best_b['alpha_xgb']:.2f} "
        f"b={best_b['beta_lgbm']:.2f} g={best_b['w_greedy']:.2f} "
        f"OOF={best_b['oof']:.5f}  Δ base={best_b['delta_vs_base']:+.5f}")
    log(f"best 1D (lgbm onto base): b={best_c['beta']:.2f} OOF={best_c['oof']:.5f}  "
        f"Δ base={best_c['delta_vs_base']:+.5f}")

    # Submission gate: >= +0.0003 vs base.
    cands = []
    if best_b["delta_vs_base"] >= 3e-4:
        cands.append(("2d", best_b, log_blend_n(
            [test_xgb, test_pred, test_greedy],
            [best_b["alpha_xgb"], best_b["beta_lgbm"], best_b["w_greedy"]]
        )))
    if best_c["delta_vs_base"] >= 3e-4:
        cands.append(("1d", best_c, log_blend_n(
            [test_pred, base_test], [best_c["beta"], 1 - best_c["beta"]]
        )))
    if cands:
        kind, info, test_blend = max(cands, key=lambda t: t[1]["oof"])
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / f"submission_greedy_nonrule_lgbm_{kind}.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}  (kind={kind})")
        results["action"] = f"ready_to_submit_{kind}"
        results["submission_path"] = str(sub_path)
    else:
        log("no OOF lift clears +0.0003 threshold — no submission")
        results["action"] = "no_submission"

    with open(ART / "nonrule_lgbm_blend_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/nonrule_lgbm_blend_results.json")


if __name__ == "__main__":
    main()
