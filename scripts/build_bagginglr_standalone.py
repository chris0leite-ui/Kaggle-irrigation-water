"""Build standalone bagginglr CSV from existing on-disk OOF + test arrays.

Tests the assumption: "is v1's LB 0.98129 driven by RF as L2, or by the
7-component bank?" bagginglr_natural is the SAME bank with LR-bagging as
L2 instead of RF — produced by scripts/n3_l3_bagging_metas.py at OOF
0.98065 (essentially tied with v1 OOF 0.98063). The standalone CSV was
never built — only the L3 (RF+ET+BagLR) mean was emitted.

This script: bias-tune on bagginglr OOF, write standalone CSV, report
diffs vs v1 PRIMARY (LB 0.98129), idea4b (LB 0.98150), and B (LB 0.98140).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

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


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    bl_oof = normed(np.load(ART / "oof_bagginglr_natural.npy").astype(np.float64))
    bl_test = normed(np.load(ART / "test_bagginglr_natural.npy").astype(np.float64))

    # Bias-tune on OOF (single pass — standard calibration step, not selection)
    bias, tuned = tune_log_bias(bl_oof, y, prior)
    oof_arg = biased_arg(bl_oof, bias)
    test_arg = biased_arg(bl_test, bias)
    bal_oof = balanced_accuracy_score(y, oof_arg)
    pcr = per_class_recall(y, oof_arg)

    print(f"bagginglr_natural standalone:")
    print(f"  OOF bal_acc: {bal_oof:.5f}")
    print(f"  bias: L={bias[0]:+.3f} M={bias[1]:+.3f} H={bias[2]:+.3f}")
    print(f"  PCR:  L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}")

    # v1 anchor (for reference)
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float64))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_oof_arg = biased_arg(v1_oof, v1_bias)
    v1_pcr = per_class_recall(y, v1_oof_arg)
    print(f"\nv1 (RF natural) reference:")
    print(f"  OOF bal_acc: {balanced_accuracy_score(y, v1_oof_arg):.5f}  (LB 0.98129)")
    print(f"  PCR:  L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}")
    print(f"  Δ PCR (bagginglr − v1): "
          f"L={pcr[0]-v1_pcr[0]:+.5f}  M={pcr[1]-v1_pcr[1]:+.5f}  H={pcr[2]-v1_pcr[2]:+.5f}")

    # Build CSV
    out_csv = SUB / "submission_bagginglr_natural_standalone.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in test_arg]}).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Diffs vs known LB-anchors
    refs = {
        "v1 PRIMARY (LB 0.98129)":      "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv",
        "idea4b PRIMARY (LB 0.98150)":  "submission_idea4b_selective_override.csv",
        "B (raw+t1b k=2 unan, LB 0.98140)": "submission_2other_raw_tier1b_k2.csv",
    }
    diffs = {}
    for name, fn in refs.items():
        path = SUB / fn
        if not path.exists():
            continue
        ref_pred = pd.read_csv(path)[TARGET].map(CLS2IDX).to_numpy()
        d = int((test_arg != ref_pred).sum())
        diffs[name] = d
        # Class-direction breakdown
        flips = []
        for from_c in range(3):
            for to_c in range(3):
                if from_c == to_c:
                    continue
                n = int(((ref_pred == from_c) & (test_arg == to_c)).sum())
                if n:
                    flips.append(f"{IDX2CLS[from_c]}->{IDX2CLS[to_c]}:{n}")
        print(f"\n  Diff vs {name}: {d}")
        print(f"    flip directions: {' '.join(flips)}")

    # Class counts on test
    counts = np.bincount(test_arg, minlength=3)
    print(f"\nTest class counts: L={counts[0]} M={counts[1]} H={counts[2]}")

    # Save summary
    summary = dict(
        candidate="bagginglr_natural_standalone",
        oof_bal=float(bal_oof),
        oof_pcr=pcr.tolist(),
        bias=bias.tolist(),
        v1_oof_bal=float(balanced_accuracy_score(y, v1_oof_arg)),
        v1_pcr=v1_pcr.tolist(),
        delta_oof_vs_v1=float(bal_oof - balanced_accuracy_score(y, v1_oof_arg)),
        diffs_vs_refs=diffs,
        test_class_counts=counts.tolist(),
        submission_path=str(out_csv),
        purpose=("Test assumption: is v1's LB 0.98129 driven by RF L2 or by the bank? "
                 "Same 7-component natural-cal bank, LR-bagging as L2 instead of RF."),
        expected_lb_ranges=dict(
            bank_carries_lift=">=0.98129 (within noise of v1)",
            l2_matters_marginally="0.98100-0.98128",
            rf_l2_essential="<0.98100",
        ),
    )
    out_json = ART / "bagginglr_standalone_results.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
