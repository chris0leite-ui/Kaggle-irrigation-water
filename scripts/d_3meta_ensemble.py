"""D experiment: 3-way meta ensemble (v1 + classw + mlp) in LB-validated PRIMARY arch.

Three iso-calibrated metas:
  - v1 (LB-validated, errs 9044)
  - classw (training-time class balancing, errs 9455)
  - mlp (3-layer NN meta, errs 9645)

All Jaccards 0.78-0.80 — similar orthogonality magnitude, but each
is independently orthogonal to v1.

Blend in LB-validated PRIMARY arch:
  primary' = 0.7 × LB3 + 0.3 × log_blend(v1_iso, classw_iso, mlp_iso, weights)

Sweep weights on the 3-meta simplex, score @ fixed BIAS, gate at:
  Δ ≥ +3e-4 vs PRIMARY OOF AND per-class guardrail PASS

Then dual-α probe per the rule learned from classw a040 closure:
  if best Δ ≥ +3e-4 at α=0.30, ALSO test α=0.40 to verify carryover
  doesn't snap back. Only LB-probe if dual-α holds.
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


def main() -> None:
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)

    # Three meta variants
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)

    classw_o = normed(np.load(ART / "oof_xgb_metastack_classw.npy"))
    classw_t = normed(np.load(ART / "test_xgb_metastack_classw.npy"))
    classw_iso_o, classw_iso_t = iso_cal(classw_o, classw_t, y)

    mlp_o = normed(np.load(ART / "oof_mlp_metastack.npy"))
    mlp_t = normed(np.load(ART / "test_mlp_metastack.npy"))
    mlp_iso_o, mlp_iso_t = iso_cal(mlp_o, mlp_t, y)

    # Baseline PRIMARY (LB 0.98094)
    p_v1_o = log_blend([s3_o, v1_iso_o], np.array([0.70, 0.30]))
    p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
    pred_v1 = (np.log(np.clip(p_v1_o, 1e-12, 1)) + BIAS).argmax(1)
    m_v1 = balanced_accuracy_score(y, pred_v1)
    rec_v1 = recall_score(y, pred_v1, average=None)
    print(f"PRIMARY (LB 0.98094): OOF={m_v1:.5f}  rec={rec_v1.round(5)}")
    print()

    # Sweep 3-way meta weights on simplex (step=0.10)
    print("3-way meta sweep at α=0.30 (LB-validated arch)")
    print(f"  {'w_v1':>6} {'w_cw':>6} {'w_mlp':>6}  {'OOF':>7}  {'Δ':>9}  {'errs':>5}  pcr  guard  emit")
    out = {"baseline_oof": float(m_v1), "rows": []}
    best = None
    for w_v1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for w_cw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            w_mlp = 1.0 - w_v1 - w_cw
            if w_mlp < -1e-9 or w_mlp > 1.0 + 1e-9:
                continue
            ws = np.array([w_v1, w_cw, w_mlp])
            meta_blend_o = log_blend([v1_iso_o, classw_iso_o, mlp_iso_o], ws)
            p_o = log_blend([s3_o, meta_blend_o], np.array([0.70, 0.30]))
            pred = (np.log(np.clip(p_o, 1e-12, 1)) + BIAS).argmax(1)
            m = balanced_accuracy_score(y, pred)
            rec = recall_score(y, pred, average=None)
            d = m - m_v1
            drec = (rec - rec_v1).round(6)
            guard = bool((drec >= -5e-4).all())
            emit = guard and d >= 3e-4
            row = {"w_v1": w_v1, "w_cw": w_cw, "w_mlp": float(w_mlp),
                    "oof": float(m), "d": float(d), "errs": int((pred != y).sum()),
                    "drec": drec.tolist(), "guard": guard, "emit": emit}
            out["rows"].append(row)
            if best is None or (emit and d > best["d"]):
                if emit:
                    best = row

    # Print top-10 by delta among guard-pass
    guard_pass = [r for r in out["rows"] if r["guard"]]
    guard_pass.sort(key=lambda r: -r["d"])
    print("\nTop 10 guard-pass configs by Δ:")
    for r in guard_pass[:10]:
        marker = "  ← EMIT" if r["emit"] else ""
        print(f"  v1={r['w_v1']:.1f} cw={r['w_cw']:.1f} mlp={r['w_mlp']:.1f}  "
              f"OOF={r['oof']:.5f}  Δ={r['d']:+.5f}  errs={r['errs']}  "
              f"drec={r['drec']}{marker}")

    if best:
        print(f"\nBEST emit:")
        print(f"  v1={best['w_v1']} cw={best['w_cw']} mlp={best['w_mlp']:.1f}  "
              f"Δ={best['d']:+.5f}")
        # Build the test blend at this config
        ws = np.array([best['w_v1'], best['w_cw'], best['w_mlp']])
        meta_blend_t = log_blend([v1_iso_t, classw_iso_t, mlp_iso_t], ws)
        p_t = log_blend([s3_t, meta_blend_t], np.array([0.70, 0.30]))
        pred_t = (np.log(np.clip(p_t, 1e-12, 1)) + BIAS).argmax(1)
        pred_v1_t = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
        n_diff = int((pred_t != pred_v1_t).sum())
        test_df = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test_df["id"].values,
                             "Irrigation_Need": [LABELS[i] for i in pred_t]})
        # Encode weights in filename for reproducibility
        tag = f"v1{int(best['w_v1']*10):02d}_cw{int(best['w_cw']*10):02d}_mlp{int(best['w_mlp']*10):02d}"
        fname = f"submission_3meta_{tag}_a030.csv"
        sub.to_csv(SUB / fname, index=False)
        print(f"  test_diff_vs_PRIMARY={n_diff}")
        print(f"  -> SAVED {fname}")
        out["best_emit"] = best
        out["submission"] = fname

    out_path = ART / "d_3meta_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
