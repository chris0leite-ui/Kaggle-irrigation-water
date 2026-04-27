"""4-gate filter sweep across all available meta candidates.

Candidates loaded automatically from scripts/artifacts/oof_xgb_metastack*.npy
and oof_mlp_metastack.npy. For each candidate:
  - Iso-cal it on OOF
  - Build PRIMARY' = 0.7 * LB-best-3-stack + α * candidate_iso for α ∈ {0.20..0.40}
  - Apply 4 gates:
      G1: OOF Δ ≥ +0.0003 vs v1 PRIMARY (LB 0.98094)
      G2: per-class recall guardrail (each class ≥ baseline - 5e-4)
      G3: dual-α stability (both α=0.30 and α=0.40 pass G1+G2)
      G4: |net_rare_class_flip| / churn ≥ 0.5  (NEW rule)

Output: ranked table of survivors. Goal: find candidates that pass ALL 4
WITHOUT LB-probing first (saving slots for true high-confidence candidates).
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


def predict(p, b=BIAS):
    return (np.log(np.clip(p, 1e-12, 1)) + b).argmax(1)


def main():
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)

    # PRIMARY baseline
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)
    p_v1_o = log_blend([s3_o, v1_iso_o], np.array([0.70, 0.30]))
    p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
    pred_v1_o = predict(p_v1_o); pred_v1_t = predict(p_v1_t)
    base_oof = balanced_accuracy_score(y, pred_v1_o)
    base_rec = recall_score(y, pred_v1_o, average=None)
    print(f"PRIMARY baseline OOF={base_oof:.5f}")

    # Discover candidate metas (metastack + mlp + N5b families) on disk.
    # Skip per-fold checkpoints and known-broken / partial.
    cand_files = []
    for p in sorted(ART.glob("oof_xgb_metastack*.npy")):
        n = p.stem.replace("oof_", "")
        if any(s in n for s in ("_fold", "_iso", "_smoke")):
            continue
        if (ART / f"test_{n}.npy").exists():
            cand_files.append(n)
    for p in sorted(ART.glob("oof_mlp_metastack*.npy")):
        n = p.stem.replace("oof_", "")
        if (ART / f"test_{n}.npy").exists():
            cand_files.append(n)
    for p in sorted(ART.glob("oof_meta_l3_xgb_mlp.npy")):
        n = p.stem.replace("oof_", "")
        if (ART / f"test_{n}.npy").exists():
            cand_files.append(n)
    print(f"Found {len(cand_files)} candidate metas:")
    for n in cand_files:
        print(f"  {n}")

    rows = []
    for cand_name in cand_files:
        try:
            cand_o = normed(np.load(ART / f"oof_{cand_name}.npy"))
            cand_t = normed(np.load(ART / f"test_{cand_name}.npy"))
        except Exception as e:
            print(f"  SKIP {cand_name}: {e}"); continue
        if cand_o.shape != v1_o.shape:
            print(f"  SKIP {cand_name}: shape mismatch {cand_o.shape}"); continue
        cand_iso_o, cand_iso_t = iso_cal(cand_o, cand_t, y)

        # Compute α=0.30 and α=0.40 scores
        result = {"cand": cand_name}
        for alpha in [0.30, 0.40]:
            p_o = log_blend([s3_o, cand_iso_o], np.array([1 - alpha, alpha]))
            p_t = log_blend([s3_t, cand_iso_t], np.array([1 - alpha, alpha]))
            pred_o = predict(p_o); pred_t = predict(p_t)
            m = balanced_accuracy_score(y, pred_o)
            rec = recall_score(y, pred_o, average=None)
            d = m - base_oof
            drec = (rec - base_rec).round(6)
            g2 = bool((drec >= -5e-4).all())
            g1 = d >= 3e-4
            # Net-rare-class-flip on TEST side (G4)
            n_to_high = int(((pred_t == 2) & (pred_v1_t != 2)).sum())
            n_from_high = int(((pred_t != 2) & (pred_v1_t == 2)).sum())
            churn = n_to_high + n_from_high
            net = n_to_high - n_from_high
            ratio = abs(net) / max(1, churn)
            g4 = ratio >= 0.5
            result[f"a{int(alpha*100):02d}_oof"] = float(m)
            result[f"a{int(alpha*100):02d}_d"] = float(d)
            result[f"a{int(alpha*100):02d}_g1"] = bool(g1)
            result[f"a{int(alpha*100):02d}_g2"] = bool(g2)
            result[f"a{int(alpha*100):02d}_drec"] = drec.tolist()
            result[f"a{int(alpha*100):02d}_to_high"] = n_to_high
            result[f"a{int(alpha*100):02d}_from_high"] = n_from_high
            result[f"a{int(alpha*100):02d}_net_high"] = net
            result[f"a{int(alpha*100):02d}_churn"] = churn
            result[f"a{int(alpha*100):02d}_ratio"] = float(ratio)
            result[f"a{int(alpha*100):02d}_g4"] = bool(g4)
        # G3 dual-α: both α=0.30 and α=0.40 pass G1 + G2
        result["g3_dual_alpha"] = bool(result["a30_g1"] and result["a30_g2"]
                                       and result["a40_g1"] and result["a40_g2"])
        # All-4-gates (use α=0.30 G1+G2+G4 since it's the LB-validated arch)
        result["all4_pass_a30"] = bool(result["a30_g1"] and result["a30_g2"]
                                        and result["g3_dual_alpha"]
                                        and result["a30_g4"])
        rows.append(result)

    # Print table
    print()
    print(f"{'candidate':45s}  {'a30Δ':>8s}  {'g1':>3s}{'g2':>3s}{'g3':>3s}{'g4':>3s}  "
          f"{'a30 to/from H':>14s}  {'a30 ratio':>10s}  {'all4':>5s}")
    for r in sorted(rows, key=lambda r: -r["a30_d"]):
        marker = " ← PASS" if r["all4_pass_a30"] else ""
        print(f"{r['cand']:45s}  {r['a30_d']:+.5f}  "
              f"{'Y' if r['a30_g1'] else '·':>3s}"
              f"{'Y' if r['a30_g2'] else '·':>3s}"
              f"{'Y' if r['g3_dual_alpha'] else '·':>3s}"
              f"{'Y' if r['a30_g4'] else '·':>3s}  "
              f"{r['a30_to_high']:>3d}/{r['a30_from_high']:>3d}({r['a30_net_high']:+3d})  "
              f"{r['a30_ratio']:>10.3f}{marker}")

    # Save table
    out_path = ART / "four_gate_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({"baseline_oof": float(base_oof), "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Survivors
    survivors = [r for r in rows if r["all4_pass_a30"]]
    print(f"\n{'='*60}")
    print(f"4-GATE SURVIVORS: {len(survivors)}")
    if survivors:
        for r in sorted(survivors, key=lambda r: -r["a30_d"]):
            print(f"  ★ {r['cand']}  Δ={r['a30_d']:+.5f}  ratio={r['a30_ratio']:.3f}")


if __name__ == "__main__":
    main()
