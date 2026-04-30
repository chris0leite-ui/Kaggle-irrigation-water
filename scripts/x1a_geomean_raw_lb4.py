"""X1a: prob-level geomean of LB-validated structurally-orthogonal models.

We've never blended at the probability level the two strongest LB-validated
SUBMISSIONS that aren't built from each other:

  rawashishsin v3   LB 0.98109   single XGB depth=3 + sklearn TE(cv=5) on rawashishsin's narrow FE
                                 + ORIG_ROW_WEIGHT=0.5 + no L2 reg
  LB-best 4-stack   LB 0.98094   tier1b 0.7×lb3 + 0.3×xgb_metastack_iso
                                 (lb3 = recipe + pseudo_s1 + pseudo_s7 + RealMLP + xgb_nonrule_iso)

These two are STRUCTURALLY ORTHOGONAL model classes. Override mechanisms
have only mixed them at the argmax level. Prob-level geomean preserves
calibration of both inputs and lets the natural-cal of each contribute.

Mechanism is genuinely untested on this comp.

Steps:
  1. Load rawashishsin OOF + test
  2. Reconstruct LB-best 4-stack OOF + test via tier1b_helpers
  3. Geomean (per-row, per-class log-mean then exp) — sweep weights
     {0.3, 0.4, 0.5, 0.6, 0.7} for rawashishsin
  4. Tune log-bias on each weighted geomean
  5. Pick best by OOF (NOT against fixed bias — let log-bias adapt)
  6. Emit submission
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias, fast_bal_acc, IDX2CLS  # noqa: E402
from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log("loading y...")
    y = load_y()
    n_tr = len(y)
    prior = np.bincount(y, minlength=3) / n_tr
    log(f"  prior = {prior.round(4).tolist()}")

    log("reconstructing LB-best 3-stack...")
    s2_o, s2_t = build_lbbest_stack(y)
    # Add the 4-stack step: log_blend(3stack, xgb_metastack_iso, [0.7, 0.3])
    log("loading xgb_metastack and iso-calibrating...")
    meta_oof = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_test = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_oof, meta_test, y)
    lb4_o = log_blend([s2_o, meta_o_iso], np.array([0.70, 0.30]))
    lb4_t = log_blend([s2_t, meta_t_iso], np.array([0.70, 0.30]))

    # Sanity check: LB-best 4-stack should reproduce ~0.98084 at recipe bias
    bias_recipe = np.array([1.4324, 1.4689, 3.4008])
    lb4_acc = fast_bal_acc(y, (np.log(np.clip(lb4_o, 1e-12, 1)) + bias_recipe).argmax(1))
    log(f"  LB-best 4-stack OOF @ recipe bias: {lb4_acc:.5f} (expected ~0.98084)")

    log("loading rawashishsin v3...")
    raw_o = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_t = normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))

    # Sanity: rawashishsin OOF tuned should be ~0.98010
    bias_raw, raw_acc = tune_log_bias(raw_o, y, prior)
    log(f"  rawashishsin OOF tuned: {raw_acc:.5f} bias={bias_raw.round(4).tolist()}")

    # Sanity: LB-best 4-stack OOF tuned should be ~0.98084 + tiny lift from re-tune
    bias_lb4, lb4_tuned = tune_log_bias(lb4_o, y, prior)
    log(f"  LB-best 4-stack OOF tuned: {lb4_tuned:.5f} bias={bias_lb4.round(4).tolist()}")

    # Geomean weight sweep
    summary = []
    weights = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    log("\n=== geomean weight sweep (raw_w * raw + (1-raw_w) * lb4) ===")
    for w_raw in weights:
        w = np.array([w_raw, 1.0 - w_raw])
        gm_o = log_blend([raw_o, lb4_o], w)
        gm_t = log_blend([raw_t, lb4_t], w)

        bias, tuned = tune_log_bias(gm_o, y, prior)
        log(f"  w_raw={w_raw:.2f}: tuned OOF = {tuned:.5f}  bias={bias.round(4).tolist()}")

        # Test predictions
        test_pred = (np.log(np.clip(gm_t, 1e-12, 1)) + bias).argmax(1)
        cls_count = {c: int((test_pred == i).sum()) for i, c in IDX2CLS.items()}
        log(f"    test class counts: {cls_count}")

        summary.append({
            "w_raw": w_raw,
            "tuned_oof": tuned,
            "bias": bias.tolist(),
            "test_class_counts": cls_count,
        })

    # Pick best by OOF tuned
    best = max(summary, key=lambda x: x["tuned_oof"])
    log(f"\n=== BEST: w_raw={best['w_raw']:.2f}, tuned OOF={best['tuned_oof']:.5f} ===")

    # Emit submission for the best weight
    w_best = np.array([best["w_raw"], 1.0 - best["w_raw"]])
    gm_t_best = log_blend([raw_t, lb4_t], w_best)
    bias_best = np.array(best["bias"])
    test_pred = (np.log(np.clip(gm_t_best, 1e-12, 1)) + bias_best).argmax(1)
    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": [IDX2CLS[i] for i in test_pred],
    })
    out_csv = SUB / f"submission_x1a_geomean_raw{int(best['w_raw']*100):02d}_lb4{int((1-best['w_raw'])*100):02d}.csv"
    sub.to_csv(out_csv, index=False)
    log(f"emitted: {out_csv}")

    # Compare to current LB-best (B = submission_2other_raw_tier1b_k2.csv, LB 0.98140)
    b_sub = pd.read_csv(SUB / "submission_2other_raw_tier1b_k2.csv")
    b_pred = b_sub["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()
    diff = int((test_pred != b_pred).sum())
    log(f"  diff vs B (LB 0.98140): {diff} rows ({100*diff/len(b_pred):.3f}%)")

    # Direction breakdown
    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((b_pred == fr) & (test_pred == to)).sum())
            if n > 0:
                directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    log(f"  directions vs B: {directions}")

    # Save full results
    out_json = ART / "x1a_geomean_results.json"
    out_json.write_text(json.dumps({
        "summary": summary,
        "best": best,
        "candidate_csv": str(out_csv),
        "diff_vs_b": diff,
        "directions_vs_b": directions,
    }, indent=2, default=float))
    log(f"summary written to {out_json}")


if __name__ == "__main__":
    main()
