"""Dual-α probe for the D experiment 3-meta best config (v1=0, cw=0.4, mlp=0.6).

Per the rule learned from classw a040 closure: when a meta-stacker variant
shows good carryover at small α, ALSO test at larger α (≥0.40) to verify
the carryover is structural and not just v1's calibration dominating.

Test configs at α=0.30 (LB-validated) AND α=0.40:
  - meta_blend = log_blend(v1_iso, classw_iso, mlp_iso, [0, 0.4, 0.6])
  - primary_α = 0.7 × LB3 + α × meta_blend, with bias retuning OFF (fixed BIAS)

If both α land in similar OOF regime AND per-class trade is stable,
the lever is real. If α=0.40 explodes, carryover is illusion.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score
from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
LABELS = ["Low", "Medium", "High"]


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)
    classw_o = normed(np.load(ART / "oof_xgb_metastack_classw.npy"))
    classw_t = normed(np.load(ART / "test_xgb_metastack_classw.npy"))
    classw_iso_o, classw_iso_t = iso_cal(classw_o, classw_t, y)
    mlp_o = normed(np.load(ART / "oof_mlp_metastack.npy"))
    mlp_t = normed(np.load(ART / "test_mlp_metastack.npy"))
    mlp_iso_o, mlp_iso_t = iso_cal(mlp_o, mlp_t, y)

    # PRIMARY baseline
    p_v1 = log_blend([s3_o, v1_iso_o], np.array([0.70, 0.30]))
    pred_v1 = (np.log(np.clip(p_v1, 1e-12, 1)) + BIAS).argmax(1)
    m_v1 = balanced_accuracy_score(y, pred_v1)
    rec_v1 = recall_score(y, pred_v1, average=None)
    print(f"PRIMARY (LB 0.98094): OOF={m_v1:.5f}")

    # Best D config: v1=0, cw=0.4, mlp=0.6
    best_ws = np.array([0.0, 0.4, 0.6])
    meta_o = log_blend([v1_iso_o, classw_iso_o, mlp_iso_o], best_ws)
    meta_t = log_blend([v1_iso_t, classw_iso_t, mlp_iso_t], best_ws)

    print("\nDual-α probe of best D config (v1=0, cw=0.4, mlp=0.6):")
    print(f"  {'α':>5}  {'OOF':>7}  {'Δ':>9}  {'errs':>5}  pcr  guard")
    out = {"baseline_oof": float(m_v1), "best_ws": best_ws.tolist(),
           "rows": []}
    for alpha in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        p_o = log_blend([s3_o, meta_o], np.array([1 - alpha, alpha]))
        pred = (np.log(np.clip(p_o, 1e-12, 1)) + BIAS).argmax(1)
        m = balanced_accuracy_score(y, pred)
        rec = recall_score(y, pred, average=None)
        d = m - m_v1
        drec = (rec - rec_v1).round(6)
        guard = bool((drec >= -5e-4).all())
        emit = guard and d >= 3e-4
        marker = "  ← gate-PASS" if emit else ""
        print(f"  {alpha:>5.2f}  {m:.5f}  {d:+.5f}  {int((pred!=y).sum())}  "
              f"{rec.round(4).tolist()}  {'PASS' if guard else 'FAIL'}{marker}")
        out["rows"].append({"alpha": alpha, "oof": float(m), "d": float(d),
                            "drec": drec.tolist(), "guard": guard, "emit": emit})

    # Carryover analysis: simulate "if LB carryover at α=0.30 holds at α=0.40"
    print("\nCarryover analysis: requires α=0.30 AND α=0.40 to BOTH show similar OOF lift")
    a30 = next(r for r in out["rows"] if r["alpha"] == 0.30)
    a40 = next(r for r in out["rows"] if r["alpha"] == 0.40)
    ratio = a40["d"] / max(a30["d"], 1e-9)
    print(f"  α=0.30  Δ={a30['d']:+.5f}  guard={a30['guard']}")
    print(f"  α=0.40  Δ={a40['d']:+.5f}  guard={a40['guard']}")
    print(f"  α=0.40 / α=0.30 = {ratio:.2f}x")
    if a30["d"] >= 3e-4 and a40["d"] >= 2e-4 and a30["guard"] and a40["guard"]:
        print("  → DUAL-α PASSES: lift is structural, recommend LB probe at α=0.30")
        out["dual_alpha_pass"] = True
    else:
        print("  → DUAL-α FAILS: carryover unstable, do NOT submit (would be classw-style illusion)")
        out["dual_alpha_pass"] = False

    out_path = ART / "d_dual_alpha_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
