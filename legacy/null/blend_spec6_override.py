"""Override greedy+nonrule (LB-best) with spec-6 predictions at score=6 rows.

Cell-2 from error analysis: rule=Medium, true=Medium, pred=High,
n=4163 at score=6. Greedy+nonrule over-predicts High on ~11% of
score=6 rows.

Spec-6 (XGB on 38k score=6 rows only, score-6 argmax raw_acc 0.963)
makes only 54 M-to-H errors on its domain (vs base's 4163) but
misses 1379 of 1549 True-High rows (H-to-M).

Design: SOFT override at score=6 rows. For each score=6 row, blend
p_new = (1-alpha) * p_greedy_nonrule + alpha * p_spec_6 in log
space. Sweep alpha, fixed greedy bias. Non-score=6 rows unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def compute_score(df):
    sm = df["Soil_Moisture"].astype(float).values
    rf = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage = df["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage, ACTIVE_STAGES), 2, 0).astype(np.int8)
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)


def main():
    log("loading base OOFs")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    oof_spec6 = np.load(ART / "oof_xgb_spec_6.npy")
    test_spec6 = np.load(ART / "test_xgb_spec_6.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])

    log("reconstructing LB-best base: greedy + nonrule @ alpha=0.15")
    oof_base = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_base = log_blend2(test_nonrule, test_greedy, 0.15)

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    score_tr = compute_score(tr)
    score_te = compute_score(te)

    mask_tr_6 = score_tr == 6
    mask_te_6 = score_te == 6
    log(f"score=6 rows: train {mask_tr_6.sum()}  test {mask_te_6.sum()}")

    base_ba = balanced_accuracy_score(y,
        (np.log(np.clip(oof_base, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    log(f"LB-best ref OOF: {base_ba:.5f}")

    # Score-6 only diagnostic: what is the base currently doing at score=6?
    preds_base = (np.log(np.clip(oof_base, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    from sklearn.metrics import confusion_matrix as cm_fn
    cm_base6 = cm_fn(y[mask_tr_6], preds_base[mask_tr_6], labels=[0, 1, 2])
    log(f"base OOF CM at score=6 (rows=true, cols=pred):\n"
        f"{pd.DataFrame(cm_base6, index=CLASSES, columns=CLASSES)}")

    # Spec's CM at score=6 (argmax only, no bias)
    spec_preds6 = oof_spec6[mask_tr_6].argmax(axis=1)
    cm_spec6 = cm_fn(y[mask_tr_6], spec_preds6, labels=[0, 1, 2])
    log(f"spec-6 OOF CM at score=6 (argmax):\n"
        f"{pd.DataFrame(cm_spec6, index=CLASSES, columns=CLASSES)}")

    # Hard override: replace base with spec at score=6 rows
    oof_hard = oof_base.copy()
    oof_hard[mask_tr_6] = oof_spec6[mask_tr_6]
    test_hard = test_base.copy()
    test_hard[mask_te_6] = test_spec6[mask_te_6]
    hard_ba = balanced_accuracy_score(y,
        (np.log(np.clip(oof_hard, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    log(f"HARD override @ score=6:  OOF = {hard_ba:.5f}  Δ = {hard_ba - base_ba:+.5f}")

    # Soft override sweep
    log("SOFT override sweep: alpha on spec_6 at score=6 rows only")
    grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
    sweep = []
    best = {"alpha": 0.0, "oof": base_ba, "delta": 0.0}
    for alpha in grid:
        oof_soft = oof_base.copy()
        if alpha == 1.0:
            oof_soft[mask_tr_6] = oof_spec6[mask_tr_6]
        elif alpha > 0:
            blended = log_blend2(oof_spec6[mask_tr_6], oof_base[mask_tr_6], alpha)
            oof_soft[mask_tr_6] = blended
        # else alpha=0: unchanged
        ba = balanced_accuracy_score(y,
            (np.log(np.clip(oof_soft, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
        delta = ba - base_ba
        marker = ""
        if ba > best["oof"]:
            best = {"alpha": alpha, "oof": float(ba), "delta": float(delta)}
            marker = "  <- best"
        sweep.append({"alpha": alpha, "oof": float(ba), "delta_vs_lbbest": float(delta)})
        log(f"  alpha_s6={alpha:.2f}  OOF = {ba:.5f}  Δ = {delta:+.5f}{marker}")

    results = {
        "base_oof_bal_acc": float(base_ba),
        "hard_override_bal_acc": float(hard_ba),
        "hard_override_delta": float(hard_ba - base_ba),
        "sweep": sweep,
        "best": best,
    }

    if best["delta"] < 1e-5:
        log("no lift — null")
        results["action"] = "no_submission"
    elif best["delta"] < 3e-4:
        log(f"Δ={best['delta']:+.5f} below +0.0003 — borderline")
        a = best["alpha"]
        test_soft = test_base.copy()
        if a == 1.0:
            test_soft[mask_te_6] = test_spec6[mask_te_6]
        elif a > 0:
            test_soft[mask_te_6] = log_blend2(test_spec6[mask_te_6],
                                               test_base[mask_te_6], a)
        preds = (np.log(np.clip(test_soft, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_lbbest_spec6_override.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote borderline {sub}")
        results["action"] = "borderline_no_submit"
        results["submission_path"] = str(sub)
    else:
        a = best["alpha"]
        test_soft = test_base.copy()
        if a == 1.0:
            test_soft[mask_te_6] = test_spec6[mask_te_6]
        elif a > 0:
            test_soft[mask_te_6] = log_blend2(test_spec6[mask_te_6],
                                               test_base[mask_te_6], a)
        preds = (np.log(np.clip(test_soft, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_lbbest_spec6_override.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote {sub}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub)

    with open(ART / "blend_spec6_override_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/blend_spec6_override_results.json")


if __name__ == "__main__":
    main()
