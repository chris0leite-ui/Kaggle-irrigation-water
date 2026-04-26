"""Build candidate submissions from combined_meta_eval_v6 at multiple alphas.

Per CLAUDE.md "ALWAYS ASK FIRST" rule, this only EMITS candidates with
diagnostics; user must explicitly approve before any LB submission.

Generates submissions at α ∈ {0.10, 0.20, 0.30, 0.40, 0.50} for the
v6 meta blended into the LB-best 4-stack primary, with per-class recall
analysis and disagreement count vs LB-best primary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLS2IDX, log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    BIAS, ART, DATA, SUB,
    bal_at_bias, build_lbbest_stack, iso_cal, normed,
)


def main() -> None:
    print("[emit] loading data + LB-best 4-stack + v6 meta")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    # LB-best 4-stack PRIMARY = lb_best_3stack × meta_v1_iso α=0.30
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_v1_o = np.load(ART / "oof_xgb_metastack.npy")
    meta_v1_t = np.load(ART / "test_xgb_metastack.npy")
    meta_v1_iso_o, meta_v1_iso_t = iso_cal(meta_v1_o, meta_v1_t, y)
    primary_o = log_blend([lb3_o, meta_v1_iso_o], np.array([0.70, 0.30]))
    primary_t = log_blend([lb3_t, meta_v1_iso_t], np.array([0.70, 0.30]))
    primary_bal = bal_at_bias(primary_o, y)
    print(f"[emit] primary OOF macro = {primary_bal:.5f}  (LB 0.98094 target)")

    # v6 meta
    meta_v6_o = np.load(ART / "oof_xgb_metastack_v6_combined.npy")
    meta_v6_t = np.load(ART / "test_xgb_metastack_v6_combined.npy")
    meta_v6_iso_o, meta_v6_iso_t = iso_cal(meta_v6_o, meta_v6_t, y)
    v6_iso_bal = bal_at_bias(meta_v6_iso_o, y)
    print(f"[emit] v6_iso standalone OOF macro = {v6_iso_bal:.5f}")

    # Per-class recall at primary
    pred_pri = (np.log(np.clip(primary_o, 1e-12, 1)) + BIAS).argmax(1)
    rec_pri = [float((pred_pri[y == c] == c).mean()) for c in range(3)]

    out_rows = []
    for alpha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        blend_o = log_blend([primary_o, meta_v6_iso_o], np.array([1 - alpha, alpha]))
        blend_t = log_blend([primary_t, meta_v6_iso_t], np.array([1 - alpha, alpha]))
        bal = bal_at_bias(blend_o, y)
        delta = bal - primary_bal

        pred_blend = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
        rec_blend = [float((pred_blend[y == c] == c).mean()) for c in range(3)]

        # Test predictions argmax
        test_pred = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)
        test_pri_pred = (np.log(np.clip(primary_t, 1e-12, 1)) + BIAS).argmax(1)
        n_disagree = int((test_pred != test_pri_pred).sum())

        # Class distribution on test
        cls_dist = np.bincount(test_pred, minlength=3).tolist()

        sub_path = SUB / f"submission_combined_v6_a{int(alpha*100):03d}.csv"
        sub_df = pd.DataFrame({
            "id": test["id"],
            "Irrigation_Need": [["Low", "Medium", "High"][i] for i in test_pred],
        })
        sub_df.to_csv(sub_path, index=False)

        row = {
            "alpha": alpha,
            "oof": float(bal),
            "delta": float(delta),
            "rec_L": rec_blend[0],
            "rec_M": rec_blend[1],
            "rec_H": rec_blend[2],
            "delta_recL": rec_blend[0] - rec_pri[0],
            "delta_recM": rec_blend[1] - rec_pri[1],
            "delta_recH": rec_blend[2] - rec_pri[2],
            "n_test_disagree": n_disagree,
            "test_dist": cls_dist,
            "submission": str(sub_path),
        }
        out_rows.append(row)
        print(f"  α={alpha:.2f}  OOF={bal:.5f}  Δ={delta:+.5f}  "
              f"recL={rec_blend[0]:.4f}  recM={rec_blend[1]:.4f}  recH={rec_blend[2]:.4f}  "
              f"disagree={n_disagree}  emit→{sub_path.name}")

    print(f"\nprimary recall: L={rec_pri[0]:.5f}  M={rec_pri[1]:.5f}  H={rec_pri[2]:.5f}")

    # Linear gap projection: prior LR-meta-v1 had OOF +0.00046 → LB -0.00103
    # gap inflation rate ~0.0032 / unit α. Apply to estimate LB.
    print("\nlinear gap projection (from prior LR-meta calibration):")
    print(f"  α=0.10  proj LB = {primary_bal + out_rows[0]['delta'] - 0.0032 * 0.10:.5f}")
    print(f"  α=0.20  proj LB = {primary_bal + out_rows[2]['delta'] - 0.0032 * 0.20:.5f}")
    print(f"  α=0.30  proj LB = {primary_bal + out_rows[4]['delta'] - 0.0032 * 0.30:.5f}")
    print("  NOTE: projection assumes prior-meta-style overfit pattern.")
    print("  v6 uses XGB stacker (same as v1 which transferred POSITIVELY).")
    print("  Best-case LB ≈ primary + OOF Δ (no gap inflation) = ~0.98138")

    with open(ART / "combined_v6_emit_results.json", "w") as f:
        json.dump({
            "primary_oof": float(primary_bal),
            "v6_iso_standalone_oof": float(v6_iso_bal),
            "rec_primary": rec_pri,
            "alphas": out_rows,
        }, f, indent=2)
    print("[emit] DONE — submissions in submissions/  results in artifacts/")


if __name__ == "__main__":
    main()
