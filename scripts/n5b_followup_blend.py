"""N5b follow-up angles 1 + 2: mean-blend two metas + fine alpha sweep.

Angle 1: At fixed alpha=0.30 (LB-validated arch), test whether
  primary' = 0.7 x 3-stack + 0.30 x MEAN(v1_meta_iso, new_meta_iso)
  beats either pure v1 (LB 0.98094) or pure new (OOF 0.98104, expected
  LB regression). Mean preserves the LB-validated architecture.

Angle 2: Fine alpha-sweep around 0.30 for the pure swap candidate
  (replace v1_meta_iso with new_meta_iso in primary). The +0.00020
  OOF at alpha=0.30 vs +0.00058 at alpha=0.50 implies a smooth curve
  - test alpha in {0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50}.

Both run on saved artifacts; <1 min wall.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def macro_at_bias(p, y, b=BIAS):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + b).argmax(1))


def per_class_recall(p, y, b=BIAS):
    return recall_score(y, (np.log(np.clip(p, 1e-12, 1)) + b).argmax(1), average=None)


def report(name: str, oof: np.ndarray, y: np.ndarray, anchor_oof: float,
           anchor_rec: np.ndarray) -> dict:
    m = macro_at_bias(oof, y)
    r = per_class_recall(oof, y)
    d = m - anchor_oof
    drec = (r - anchor_rec).round(6)
    guard = bool((drec >= -5e-4).all())
    print(f"  {name:35s} OOF={m:.5f}  d={d:+.5f}  drec={drec.tolist()}  "
          f"{'PASS' if guard else 'FAIL'}")
    return {"name": name, "oof": float(m), "d": float(d),
            "drec": drec.tolist(), "guard": guard}


def main() -> None:
    print("[1] Loading components...")
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)

    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)

    new_o = normed(np.load(ART / "oof_xgb_metastack_n5b_both.npy"))
    new_t = normed(np.load(ART / "test_xgb_metastack_n5b_both.npy"))
    new_iso_o, new_iso_t = iso_cal(new_o, new_t, y)

    # Build v1 PRIMARY (LB 0.98094) baseline
    p_v1_o = log_blend([s3_o, v1_iso_o], np.array([0.70, 0.30]))
    p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
    base_oof = macro_at_bias(p_v1_o, y)
    base_rec = per_class_recall(p_v1_o, y)
    print(f"    v1 PRIMARY (LB 0.98094) baseline: OOF={base_oof:.5f}  rec={base_rec.round(5)}")

    out = {"baseline": {"name": "v1_PRIMARY", "oof": float(base_oof),
                        "rec": base_rec.tolist()},
           "candidates": []}

    print("\n[2] ANGLE 1: mean-blend v1+new at fixed alpha=0.30")
    # 3 mean variants: arithmetic prob, geometric prob (log-mean), weighted
    arith_o = (v1_iso_o + new_iso_o) / 2; arith_t = (v1_iso_t + new_iso_t) / 2
    arith_o = normed(arith_o); arith_t = normed(arith_t)
    geo_o = log_blend([v1_iso_o, new_iso_o], np.array([0.5, 0.5]))
    geo_t = log_blend([v1_iso_t, new_iso_t], np.array([0.5, 0.5]))

    for label, mo, mt in [("arith_mean", arith_o, arith_t), ("geo_mean", geo_o, geo_t)]:
        p_o = log_blend([s3_o, mo], np.array([0.70, 0.30]))
        out["candidates"].append(report(f"angle1_{label}_a030", p_o, y, base_oof, base_rec))

    print("\n[3] ANGLE 2: fine alpha-sweep around 0.30 (pure swap)")
    for alpha in [0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.500]:
        p_o = log_blend([s3_o, new_iso_o], np.array([1 - alpha, alpha]))
        out["candidates"].append(report(f"angle2_swap_a{int(alpha*1000):03d}", p_o, y, base_oof, base_rec))

    print("\n[4] BONUS: angle 1 + angle 2 cross — mean-blend at non-0.30 alphas")
    for alpha in [0.350, 0.400, 0.450]:
        p_o = log_blend([s3_o, geo_o], np.array([1 - alpha, alpha]))
        out["candidates"].append(report(f"x_geomean_a{int(alpha*1000):03d}", p_o, y, base_oof, base_rec))

    # Find best per-class-safe lift
    safe = [c for c in out["candidates"] if c["guard"]]
    if safe:
        best = max(safe, key=lambda c: c["d"])
        print(f"\n[5] BEST under per-class guardrail:")
        print(f"  {best['name']}  d={best['d']:+.5f}  rec_drift={best['drec']}")
        if best["d"] >= 2e-4:
            # Build & save submission for this best candidate
            print(f"  (eligible for LB probe — emit_submission for review)")
            test = pd.read_csv("data/test.csv")
            # Reconstruct test for this candidate name
            name = best["name"]
            if name.startswith("angle1_arith"):
                mo_t = (v1_iso_t + new_iso_t) / 2; mo_t = normed(mo_t)
                p_t = log_blend([s3_t, mo_t], np.array([0.70, 0.30]))
            elif name.startswith("angle1_geo"):
                p_t = log_blend([s3_t, geo_t], np.array([0.70, 0.30]))
            elif name.startswith("angle2_swap"):
                a = int(name.split("_a")[-1]) / 1000.0
                p_t = log_blend([s3_t, new_iso_t], np.array([1 - a, a]))
            elif name.startswith("x_geomean"):
                a = int(name.split("_a")[-1]) / 1000.0
                p_t = log_blend([s3_t, geo_t], np.array([1 - a, a]))
            pred_t = (np.log(np.clip(p_t, 1e-12, 1)) + BIAS).argmax(1)
            pred_v1_t = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
            n_diff = int((pred_t != pred_v1_t).sum())
            sub = pd.DataFrame({"id": test["id"].values,
                                 "Irrigation_Need": [LABELS[i] for i in pred_t]})
            fname = f"submission_n5b_followup_{name}.csv"
            sub.to_csv(SUB / fname, index=False)
            print(f"  test_diff_vs_primary={n_diff}")
            print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")
        else:
            print(f"  d below +2e-4 LB-transfer gate; no submission emitted")
    else:
        print("\nNO candidate passes per-class guardrail.")

    out_path = ART / "n5b_followup_blend_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
