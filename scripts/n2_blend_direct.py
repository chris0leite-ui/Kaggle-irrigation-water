"""Direct blend gate: ET-iso + kNN-iso × LB-best 4-stack at low α.

Cheap pre-meta-stacker check: if either component lifts the LB-best
4-stack standalone via direct log-blend at fixed bias, we have a real
LB-positive lever. If not, the only remaining hope is the meta-stacker
absorbing them as bank inputs (which is the next step anyway).
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
    ART, BIAS, CLASSES, DATA, SUB, TARGET,
    bal_at_bias as bal, build_lbbest_stack, iso_cal, log,
)


def per_class_recall(p, y):
    pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    return [(pred[y == c] == c).mean() for c in range(3)]


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
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")

    candidates = [("n2_extratrees", "ET on 35-dist"),
                  ("n2_knn", "kNN on 35-dist")]
    out = {"lb4_baseline": float(lb4_bal), "results": []}

    for name, desc in candidates:
        c_oof = np.load(ART / f"oof_{name}.npy").astype(np.float32)
        c_test = np.load(ART / f"test_{name}.npy").astype(np.float32)
        c_iso_o, c_iso_t = iso_cal(c_oof, c_test, y)
        log(f"\n=== {desc} (iso) blend gate vs LB-best 4-stack ===")
        log(f"{'alpha':>6} {'OOF':>9} {'Δ':>9} {'recL':>7} {'recM':>7} {'recH':>7}")

        rows = []
        for a in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
            blend = log_blend([lb4_o, c_iso_o], np.array([1 - a, a]))
            b = bal(blend, y)
            d = b - lb4_bal
            pcr = per_class_recall(blend, y)
            rows.append({"alpha": a, "oof": float(b), "delta": float(d),
                         "pcr": [float(x) for x in pcr]})
            tag = ""
            if d > 1e-4:
                tag = " *"
            log(f"{a:>6.3f} {b:>9.5f} {d:>+9.5f} "
                f"{pcr[0]:>7.4f} {pcr[1]:>7.4f} {pcr[2]:>7.4f}{tag}")
        best = max(rows, key=lambda r: r["delta"])
        out["results"].append({"name": name, "best": best, "sweep": rows})

    (ART / "n2_blend_direct_results.json").write_text(json.dumps(out, indent=2))
    log("\nwrote scripts/artifacts/n2_blend_direct_results.json")

    # Verdict
    log("\n=== verdict ===")
    for r in out["results"]:
        b = r["best"]
        gate_pass = b["delta"] >= 2e-4
        log(f"  {r['name']}: peak α={b['alpha']:.3f}  Δ={b['delta']:+.5f}  "
            f"{'PASS' if gate_pass else 'FAIL'} +2e-4 gate")


if __name__ == "__main__":
    main()
