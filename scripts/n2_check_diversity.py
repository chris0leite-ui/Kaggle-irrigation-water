"""Check N2 ET + kNN structural diversity vs LB-best 4-stack.

For each candidate component:
  - Standalone: argmax bal_acc, tuned bal_acc @ recipe bias
  - vs LB-best 4-stack: errs delta, Jaccard, per-class recall delta
  - Decision: viable as meta-stacker bank input if Jaccard < 0.97 AND
    errors not catastrophically more than anchor (within ~50%).

Doesn't run the full meta-stacker — that comes after N3 lands and we
can re-train v4 with all three new components together.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, DATA, TARGET, bal_at_bias as bal,
    build_lbbest_stack, iso_cal, log,
)


def per_class_recall(p, y):
    pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    return np.array([(pred[y == c] == c).mean() for c in range(3)])


def errors_at_bias(p, y):
    return ((np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1) != y).sum()


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 4-stack")
    lb3_o, lb3_t = build_lbbest_stack(y)
    xgb_oof = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    xgb_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    xgb_iso_o, xgb_iso_t = iso_cal(xgb_oof, xgb_test, y)
    lb4_o = log_blend([lb3_o, xgb_iso_o], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, xgb_iso_t], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_o, y)
    lb4_pcr = per_class_recall(lb4_o, y)
    lb4_err = int(errors_at_bias(lb4_o, y))
    pred_lb = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}  errs = {lb4_err}  "
        f"PCR = [L {lb4_pcr[0]:.4f} M {lb4_pcr[1]:.4f} H {lb4_pcr[2]:.4f}]")

    candidates = [
        ("n2_extratrees", "ET on 35-dist features"),
        ("n2_knn",        f"kNN(k=50, sub=80k) on 35-dist features"),
    ]

    out = {"lb4_baseline": {"oof": float(lb4_bal), "errs": lb4_err,
                            "pcr": lb4_pcr.tolist()},
           "components": []}
    for name, desc in candidates:
        log(f"\n=== {name}: {desc} ===")
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists():
            log(f"  SKIP — {oof_p} missing")
            continue
        c_oof = np.load(oof_p).astype(np.float32)
        c_test = np.load(test_p).astype(np.float32)
        # iso-cal version
        c_iso_o, c_iso_t = iso_cal(c_oof, c_test, y)

        for tag, p_o in [("raw", c_oof), ("iso", c_iso_o)]:
            argmax_bal = (p_o.argmax(1) == y).mean()  # NOT macro-recall; quick
            tuned_bal = bal(p_o, y)
            errs = int(errors_at_bias(p_o, y))
            pcr = per_class_recall(p_o, y)
            pred = (np.log(np.clip(p_o, 1e-12, 1)) + BIAS).argmax(1)
            both_wrong = ((pred != y) & (pred_lb != y)).sum()
            either_wrong = ((pred != y) | (pred_lb != y)).sum()
            jaccard = both_wrong / max(either_wrong, 1)
            log(f"  {tag}: argmax={argmax_bal:.5f} tuned={tuned_bal:.5f} "
                f"errs={errs}  Jaccard_vs_LB4={jaccard:.4f}")
            log(f"       PCR=[L {pcr[0]:.4f} M {pcr[1]:.4f} H {pcr[2]:.4f}]  "
                f"vs anchor [L {lb4_pcr[0]:.4f} M {lb4_pcr[1]:.4f} H {lb4_pcr[2]:.4f}]")

        out["components"].append({
            "name": name, "desc": desc,
            "raw_tuned": float(bal(c_oof, y)),
            "iso_tuned": float(bal(c_iso_o, y)),
            "raw_errs": int(errors_at_bias(c_oof, y)),
            "iso_errs": int(errors_at_bias(c_iso_o, y)),
            "raw_jaccard_vs_lb4": float(((np.log(np.clip(c_oof,1e-12,1))+BIAS).argmax(1) != y).astype(int).dot(
                ((pred_lb != y).astype(int))) / max(
                ((((np.log(np.clip(c_oof,1e-12,1))+BIAS).argmax(1) != y) | (pred_lb != y)).sum()), 1)),
            "iso_jaccard_vs_lb4": float(((np.log(np.clip(c_iso_o,1e-12,1))+BIAS).argmax(1) != y).astype(int).dot(
                ((pred_lb != y).astype(int))) / max(
                ((((np.log(np.clip(c_iso_o,1e-12,1))+BIAS).argmax(1) != y) | (pred_lb != y)).sum()), 1)),
            "raw_pcr": per_class_recall(c_oof, y).tolist(),
            "iso_pcr": per_class_recall(c_iso_o, y).tolist(),
        })

    (ART / "n2_check_diversity_results.json").write_text(json.dumps(out, indent=2))
    log("\nwrote scripts/artifacts/n2_check_diversity_results.json")

    # Quick verdict per component
    log("\n=== verdict ===")
    for c in out["components"]:
        viable_iso = (c["iso_jaccard_vs_lb4"] < 0.97 and c["iso_errs"] < 1.5 * lb4_err)
        viable_raw = (c["raw_jaccard_vs_lb4"] < 0.97 and c["raw_errs"] < 1.5 * lb4_err)
        verdict = []
        if viable_iso:
            verdict.append("ISO viable")
        if viable_raw:
            verdict.append("RAW viable")
        if not (viable_iso or viable_raw):
            verdict = ["BOTH FAIL diversity gate"]
        log(f"  {c['name']}: " + " | ".join(verdict))


if __name__ == "__main__":
    main()
