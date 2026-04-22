"""Blend per-cell LR with the rule-only predictor.

Per-cell LR standalone OOF tuned = 0.73082 (weak, below even the
rule at 0.96097). But weak standalone is not the signal we care
about — orthogonality to the dominant predictor is. The rule is
the cleanest possible reference: it uses ONLY the 6 rule features
and zero continuous non-rule features. If within-cell continuous
signal exists at all, a rule ⊗ LR blend should beat the rule.

Rule: dgp_score >= 7 -> High, 4..6 -> Medium, <=3 -> Low.
Rule on 630k synthetic: raw_acc 0.98364, bal_acc 0.96097.

This script also computes the per-row error overlap: of the
~10,304 rule-wrong rows, how many does LR get right?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ACTIVE_STAGES = ("Flowering", "Vegetative")
ART = Path("scripts/artifacts")


def compute_dgp_score(df):
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(ACTIVE_STAGES).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).values


def rule_predict(score):
    pred = np.full(len(score), CLS2IDX["Medium"], dtype=np.int32)
    pred[score <= 3] = CLS2IDX["Low"]
    pred[score >= 7] = CLS2IDX["High"]
    return pred


def rule_probs(score):
    n = len(score)
    p = np.full((n, 3), 0.01, dtype=np.float64)
    pred = rule_predict(score)
    p[np.arange(n), pred] = 0.98
    return p / p.sum(axis=1, keepdims=True)


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(
                    balanced_accuracy_score(y, (log_oof + base).argmax(axis=1))
                )
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main():
    tr = pd.read_csv("data/train.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    score = compute_dgp_score(tr)
    rule_pred = rule_predict(score)

    rule_bal = balanced_accuracy_score(y, rule_pred)
    rule_acc = (rule_pred == y).mean()
    print(f"rule-only bal_acc: {rule_bal:.5f} (raw acc {rule_acc:.5f})")
    n_rule_wrong = int((rule_pred != y).sum())
    print(f"rule-wrong rows: {n_rule_wrong} / {len(y)} "
          f"({100*n_rule_wrong/len(y):.2f}%)")

    oof_lr = np.load(ART / "oof_per_cell_lr.npy")
    lr_pred = oof_lr.argmax(axis=1)
    # LR's raw recovery of rule-wrong rows
    rule_wrong_mask = rule_pred != y
    lr_correct_on_wrong = ((lr_pred == y) & rule_wrong_mask).sum()
    lr_wrong_on_right = ((lr_pred != y) & (~rule_wrong_mask)).sum()
    print(f"LR recovers {lr_correct_on_wrong}/{n_rule_wrong} "
          f"rule-wrong rows ({100*lr_correct_on_wrong/n_rule_wrong:.2f}%)")
    print(f"LR introduces {lr_wrong_on_right} new errors on rule-right rows")

    # Blend in log space
    log_rule = np.log(np.clip(rule_probs(score), 1e-9, 1.0))
    log_lr = np.log(np.clip(oof_lr, 1e-9, 1.0))
    results = {}

    print("\n=== rule ⊗ LR log-blend sweep (tuned log-bias) ===")
    for a in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
        blend = np.exp((1 - a) * log_rule + a * log_lr)
        blend = blend / blend.sum(axis=1, keepdims=True)
        bias, tuned_bal = tune_log_bias(blend, y, prior)
        results[f"a={a:.2f}"] = float(tuned_bal)
        print(f"  a_LR={a:.2f}  tuned={tuned_bal:.5f}  bias={bias.round(3).tolist()}")

    # Per-cell-LR-confidence-gated: take LR prediction only where LR disagrees
    # with rule AND LR confidence > tau; otherwise rule.
    print("\n=== hard-gate: use LR when LR_pred != rule AND max(LR_prob) > tau ===")
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        lr_max = oof_lr.max(axis=1)
        disagree = (lr_pred != rule_pred)
        override = disagree & (lr_max > tau)
        final = rule_pred.copy()
        final[override] = lr_pred[override]
        bal = balanced_accuracy_score(y, final)
        print(f"  tau={tau}  n_override={int(override.sum()):>6}  "
              f"bal_acc={bal:.5f}")

    with open(ART / "per_cell_lr_blend_rule_results.json", "w") as f:
        json.dump({
            "rule_bal": float(rule_bal),
            "rule_acc": float(rule_acc),
            "n_rule_wrong": n_rule_wrong,
            "lr_recovers_rule_wrong": int(lr_correct_on_wrong),
            "lr_new_errors_on_rule_right": int(lr_wrong_on_right),
            "blend_sweep": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
