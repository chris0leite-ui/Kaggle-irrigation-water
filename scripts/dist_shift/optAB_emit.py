"""Emit a deployable submission CSV for Option A or B' if 4-gate passed.

CAND_NAME via env var (CAND=...). Reads the gate JSON; if any anchor
passed all 4 gates AT THE PEAK α, emits:

  submissions/submission_{CAND_NAME}_{anchor}_a{int(peak*100)}.csv

…where the blend is `log_blend([anchor_test, cand_test], [1-α, α])` at
fixed recipe bias (no per-α retune). User MUST approve before LB probe.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, build_lbbest_stack, iso_cal, load_y, normed,
)

CAND_NAME = os.environ.get("CAND", "recipe_full_te_avp")
EPS = 1e-12


def main():
    y = load_y()
    summary_path = ART / f"blend_gate_{CAND_NAME}_results.json"
    if not summary_path.exists():
        print(f"FATAL: gate JSON missing: {summary_path}")
        return
    with open(summary_path) as f:
        summary = json.load(f)

    cand_test = normed(np.load(ART / f"test_{CAND_NAME}.npy").astype(np.float32))

    # Reconstruct anchors test side
    t_r = normed(np.load(ART / "test_recipe_full_te.npy").astype(np.float32))
    _, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    _, meta_ti = iso_cal(meta_o, meta_t, y)
    lb4_t = log_blend([lb3_t, meta_ti], np.array([0.7, 0.3]))
    anchors_test = {"recipe_full_te": t_r, "lb_best_3stack": lb3_t,
                    "lb_best_4stack": lb4_t}

    test_csv = pd.read_csv("data/test.csv")
    cls_inv = {0: "Low", 1: "Medium", 2: "High"}

    emitted = []
    for name, info in summary["anchors"].items():
        if not info.get("gate_4_pass"):
            print(f"  {name}: gate_4_pass=False  peak_Δ={info['peak_delta']:+.5f}  SKIP")
            continue
        alpha = info["peak_alpha"]
        anchor_t = anchors_test[name]
        mix = log_blend([anchor_t, cand_test], np.array([1 - alpha, alpha]))
        pred = (np.log(np.clip(mix, EPS, 1.0)) + BIAS).argmax(1)
        out = pd.DataFrame({"id": test_csv["id"],
                            "Irrigation_Need": [cls_inv[i] for i in pred]})
        path = Path("submissions") / f"submission_{CAND_NAME}_{name}_a{int(alpha*100):03d}.csv"
        out.to_csv(path, index=False)
        emitted.append({"path": str(path), "anchor": name, "alpha": alpha,
                        "peak_delta": info["peak_delta"]})
        dist = out["Irrigation_Need"].value_counts().to_dict()
        print(f"  EMIT {path.name}  peak_Δ={info['peak_delta']:+.5f}  dist={dist}")

    if not emitted:
        print(f"\nNo gate passed for {CAND_NAME}. No submission emitted.")
    else:
        print(f"\n{len(emitted)} submission(s) emitted. ASK USER before LB probe.")


if __name__ == "__main__":
    main()
