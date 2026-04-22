"""Fixed-bias probe: override greedy predictions on dgp_score==3 with
xgb_spec_3 specialist predictions. Tests the {3} specialist (Open
hypothesis-board bet #3) without retuning log-bias on top.

Same fixed-bias rule as scripts/blend_mlp_probe.py: apply the greedy
baseline's already-tuned log-bias (no retune). If the overlay beats
the greedy baseline under this constraint, the specialist is adding
orthogonal signal; if it needs re-tuning to look good, it's a
selection-overfit lever and should be dropped.

Inputs:
  scripts/artifacts/oof_greedy_blend.npy
  scripts/artifacts/test_greedy_blend.npy
  scripts/artifacts/oof_xgb_spec_3.npy
  scripts/artifacts/test_xgb_spec_3.npy

Outputs:
  scripts/artifacts/spec_3_hybrid_probe_results.json
  (no submission -- run LB probe manually only if OOF is clearly positive)
"""
from __future__ import annotations

import json
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
SPEC_SCORE = 3
ART = Path("scripts/artifacts")


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
    return bias, best


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(ACTIVE_STAGES).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).values


def main() -> None:
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = compute_dgp_score(tr)
    te_scores = compute_dgp_score(te)
    tr_s3 = tr_scores == SPEC_SCORE
    te_s3 = te_scores == SPEC_SCORE

    # --- inputs ---
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_spec3 = np.load(ART / "oof_xgb_spec_3.npy")
    test_spec3 = np.load(ART / "test_xgb_spec_3.npy")

    print(f"rows with dgp_score==3:  train {tr_s3.sum()}  test {te_s3.sum()}")

    # --- reference: greedy tuned (reproduce from scratch) ---
    g_bias, g_bal = tune_log_bias(oof_greedy, y, prior)
    print(f"\ngreedy baseline (reproduce): tuned bal_acc = {g_bal:.5f}  "
          f"bias = {dict(zip(CLASSES, g_bias.round(4)))}")
    print(f"  expected reference: 0.97375 (LB 0.97296)")

    # --- spec-3 standalone on its domain ---
    spec3_y = y[tr_s3]
    spec3_oof = oof_spec3[tr_s3]
    spec3_bal = balanced_accuracy_score(spec3_y, spec3_oof.argmax(axis=1))
    spec3_raw = (spec3_oof.argmax(axis=1) == spec3_y).mean()
    rule_y_on_s3 = np.zeros(len(spec3_y), dtype=np.int32)  # rule says Low for score <= 3
    rule3_bal = balanced_accuracy_score(spec3_y, rule_y_on_s3)
    rule3_raw = (rule_y_on_s3 == spec3_y).mean()
    print(f"\n--- on score==3 domain only ({len(spec3_y)} rows) ---")
    print(f"  rule (all Low)   : raw_acc={rule3_raw:.5f}  bal_acc={rule3_bal:.5f}")
    print(f"  spec-3 argmax    : raw_acc={spec3_raw:.5f}  bal_acc={spec3_bal:.5f}")
    print(f"  greedy argmax    : raw_acc={(oof_greedy[tr_s3].argmax(axis=1)==spec3_y).mean():.5f}  "
          f"bal_acc={balanced_accuracy_score(spec3_y, oof_greedy[tr_s3].argmax(axis=1)):.5f}")

    # --- hybrid: override greedy with spec_3 on score==3 rows ---
    oof_hybrid = oof_greedy.copy()
    oof_hybrid[tr_s3] = oof_spec3[tr_s3]
    test_hybrid = test_greedy.copy()
    test_hybrid[te_s3] = test_spec3[te_s3]

    # --- apply FIXED greedy bias (no retune) ---
    log_hybrid = np.log(np.clip(oof_hybrid, 1e-9, 1.0))
    fixed_pred = (log_hybrid + g_bias).argmax(axis=1)
    fixed_bal = balanced_accuracy_score(y, fixed_pred)

    # --- also report retuned bias for context only (not the decision rule) ---
    retune_bias, retune_bal = tune_log_bias(oof_hybrid, y, prior)

    print(f"\n--- hybrid: greedy with spec_3 override on score==3 ---")
    print(f"  FIXED greedy bias bal_acc : {fixed_bal:.5f}  Δ vs greedy = {fixed_bal - g_bal:+.5f}")
    print(f"  (retune-only reference)   : {retune_bal:.5f}  Δ vs greedy = {retune_bal - g_bal:+.5f}")
    print(f"  greedy baseline           : {g_bal:.5f}  (LB 0.97296)")

    cm_fixed = confusion_matrix(y, fixed_pred)
    print(f"  confusion matrix (FIXED bias):\n"
          f"{pd.DataFrame(cm_fixed, index=CLASSES, columns=CLASSES)}")

    # --- also report soft-blend sweep with fixed bias: (1-a)*greedy + a*spec3 on score==3 rows ---
    print(f"\n--- soft-blend on score==3 rows (alpha * spec3 + (1-alpha) * greedy) w/ FIXED bias ---")
    log_greedy_arr = np.log(np.clip(oof_greedy, 1e-9, 1.0))
    log_spec3_arr = np.log(np.clip(oof_spec3, 1e-9, 1.0))
    sweep = []
    for a in np.arange(0.0, 1.01, 0.1):
        log_blend = log_greedy_arr.copy()
        log_blend[tr_s3] = (1 - a) * log_greedy_arr[tr_s3] + a * log_spec3_arr[tr_s3]
        pred = (log_blend + g_bias).argmax(axis=1)
        sc = balanced_accuracy_score(y, pred)
        sweep.append((float(a), float(sc)))
        print(f"  alpha={a:.2f}  bal_acc={sc:.5f}  Δ={sc - g_bal:+.5f}")

    out = {
        "greedy_tuned_ref": float(g_bal),
        "greedy_bias_fixed": g_bias.tolist(),
        "spec3_domain_rule_bal_acc": float(rule3_bal),
        "spec3_domain_spec_bal_acc": float(spec3_bal),
        "spec3_domain_greedy_bal_acc": float(balanced_accuracy_score(
            spec3_y, oof_greedy[tr_s3].argmax(axis=1))),
        "hybrid_fixed_bias_bal_acc": float(fixed_bal),
        "hybrid_retuned_bal_acc": float(retune_bal),
        "delta_hybrid_fixed_vs_greedy": float(fixed_bal - g_bal),
        "delta_hybrid_retune_vs_greedy": float(retune_bal - g_bal),
        "soft_blend_sweep_on_s3": sweep,
        "n_rows_overridden_train": int(tr_s3.sum()),
        "n_rows_overridden_test": int(te_s3.sum()),
    }
    with open(ART / "spec_3_hybrid_probe_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {ART / 'spec_3_hybrid_probe_results.json'}")


if __name__ == "__main__":
    main()
