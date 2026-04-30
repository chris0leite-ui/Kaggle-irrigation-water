"""Non-override OOF-honest blend candidate.

Three LB-validated, non-override standalone components, equal-weight
geomean (no OOF-tuned weights → no selection bias on weights), one
final pass of per-class log-bias (the standard calibration step used
across this comp).

Components (all non-override, all standalone-LB-validated):
  v1:  sklearn_rf_meta_natural_v1_lb98129    LB 0.98129
  raw: rawashishsin_2600_standalone           LB 0.98109
  t1b: tier1b_greedy_meta                     LB 0.98094

Outputs:
  submissions/submission_nonoverride_eqblend_v1raw_t1b.csv
  scripts/artifacts/nonoverride_eqblend_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def normed(a, eps=1e-9):
    a = np.clip(a, eps, 1.0)
    return a / a.sum(axis=1, keepdims=True)


def biased_arg(p, b, eps=1e-9):
    return (np.log(np.clip(p, eps, 1.0)) + b).argmax(1)


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    comps = [
        ("v1",  "oof_sklearn_rf_meta_natural_v1_lb98129.npy",
                "test_sklearn_rf_meta_natural_v1_lb98129.npy", 0.98129),
        ("raw", "oof_rawashishsin_2600.npy",
                "test_rawashishsin_2600.npy", 0.98109),
        ("t1b", "oof_tier1b_greedy_meta.npy",
                "test_tier1b_greedy_meta.npy", 0.98094),
    ]
    pool = {}
    for label, oof_p, test_p, lb in comps:
        oof = normed(np.load(ART / oof_p).astype(np.float64))
        tst = normed(np.load(ART / test_p).astype(np.float64))
        pool[label] = dict(oof=oof, test=tst, lb=lb)

    # Standalone diagnostics: each component with its own bias-tune
    standalone = {}
    for label, d in pool.items():
        bias, tuned = tune_log_bias(d["oof"], y, prior)
        oof_arg = biased_arg(d["oof"], bias)
        bal = balanced_accuracy_score(y, oof_arg)
        standalone[label] = dict(bias=bias.tolist(), oof_bal=float(bal),
                                 lb=d["lb"], gap=float(bal - d["lb"]))
        print(f"{label:>4}  OOF {bal:.5f}  LB {d['lb']:.5f}  gap {bal-d['lb']:+.5f}")

    # Equal-weight log-blend (geomean) — NO OOF-tuned weights
    weights = np.array([1/3, 1/3, 1/3])
    oof_blend = log_blend([pool[l]["oof"] for l in ["v1", "raw", "t1b"]], weights)
    test_blend = log_blend([pool[l]["test"] for l in ["v1", "raw", "t1b"]], weights)

    # ONE pass of bias-tune on the blend (standard step, not weight selection)
    bias_blend, tuned_blend = tune_log_bias(oof_blend, y, prior)
    oof_arg = biased_arg(oof_blend, bias_blend)
    test_arg = biased_arg(test_blend, bias_blend)
    bal_blend = balanced_accuracy_score(y, oof_arg)
    print(f"\nEqual-weight geomean blend (v1+raw+t1b):")
    print(f"  OOF bal_acc: {bal_blend:.5f}")
    print(f"  bias: L={bias_blend[0]:+.3f} M={bias_blend[1]:+.3f} H={bias_blend[2]:+.3f}")

    # Per-class recall
    pcr = np.zeros(3)
    for k in range(3):
        m = y == k
        pcr[k] = (oof_arg[m] == k).sum() / max(m.sum(), 1)
    print(f"  PCR: L={pcr[0]:.4f}  M={pcr[1]:.4f}  H={pcr[2]:.4f}")

    # Reference comparisons
    print(f"\nReference points:")
    print(f"  v1 standalone OOF→LB: 0.98063 → 0.98129  (gap -0.00066)")
    print(f"  t1b standalone OOF→LB: 0.98084 → 0.98094 (gap -0.00010, most calibrated)")
    print(f"  PRIMARY idea4b LB:   0.98150  (override family)")

    # Sanity: arithmetic-mean variant for comparison (no selection between them — both reported)
    oof_arith = (pool["v1"]["oof"] + pool["raw"]["oof"] + pool["t1b"]["oof"]) / 3
    test_arith = (pool["v1"]["test"] + pool["raw"]["test"] + pool["t1b"]["test"]) / 3
    bias_a, _ = tune_log_bias(oof_arith, y, prior)
    bal_arith = balanced_accuracy_score(y, biased_arg(oof_arith, bias_a))
    print(f"  (arith-mean variant for ref: OOF {bal_arith:.5f})")

    # Diff vs PRIMARY (idea4b) — count test rows that differ
    primary_csv = SUB / "submission_idea4b_selective_override.csv"
    if primary_csv.exists():
        primary_pred = pd.read_csv(primary_csv)[TARGET].map(CLS2IDX).to_numpy()
        diff_primary = int((test_arg != primary_pred).sum())
        print(f"\n  Test rows differing from PRIMARY (idea4b LB 0.98150): {diff_primary}")
    else:
        diff_primary = None

    # Diff vs HEDGE (RF natural standalone)
    hedge_csv = SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv"
    if hedge_csv.exists():
        hedge_pred = pd.read_csv(hedge_csv)[TARGET].map(CLS2IDX).to_numpy()
        diff_hedge = int((test_arg != hedge_pred).sum())
        print(f"  Test rows differing from HEDGE (RF natural LB 0.98129): {diff_hedge}")
    else:
        diff_hedge = None

    # Save submission (geomean variant — the principled one)
    out_csv = SUB / "submission_nonoverride_eqblend_v1raw_t1b.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in test_arg]}).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    summary = {
        "candidate_csv": str(out_csv),
        "method": "equal-weight (1/3,1/3,1/3) geomean of v1+raw+t1b, bias-tune on blend",
        "components": [
            {"label": l, "oof_bal_standalone": standalone[l]["oof_bal"],
             "lb": standalone[l]["lb"], "gap": standalone[l]["gap"]}
            for l in ["v1", "raw", "t1b"]
        ],
        "blend_oof_bal": float(bal_blend),
        "blend_pcr": pcr.tolist(),
        "blend_bias": bias_blend.tolist(),
        "arith_mean_oof_for_ref": float(bal_arith),
        "diff_vs_primary_idea4b": diff_primary,
        "diff_vs_hedge_rf_natural": diff_hedge,
        "note": ("Equal-weight chosen to avoid OOF-selection-bias on weights. "
                 "Single bias-tune is the standard calibration step, not a "
                 "selectable hyperparameter. NO grid search, NO subset selection."),
    }
    with open(ART / "nonoverride_eqblend_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("Saved: scripts/artifacts/nonoverride_eqblend_results.json")


if __name__ == "__main__":
    main()
