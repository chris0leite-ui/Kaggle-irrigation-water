#!/usr/bin/env python3
"""D1: Direct architecture substitution — replace xgb_metastack_iso with
rawashishsin_iso in the LB-best 4-stack.

LB-best 4-stack (CLAUDE.md):
  stack2 = LB-3-stack × 0.925 + xgb_nonrule_iso × 0.075
  primary = stack2 × 0.70 + xgb_metastack_iso × 0.30   [α=0.30 LB-validated]

Substitution test:
  primary' = stack2 × 0.70 + rawashishsin_iso × 0.30   [same α]

Two variants tested:
  - v2: rawashishsin (n_est=1500)
  - v3: rawashishsin_2600 (n_est=2600 faithful)

Both at fixed recipe bias [1.4324, 1.4689, 3.4008].
4-gate filter, then build LB-probe candidate CSVs.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal, load_y, normed)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
INT2LABEL = {0: "Low", 1: "Medium", 2: "High"}


def per_class_recall(y, pred):
    return np.array([(pred[y == c] == c).mean() for c in range(3)])


def main():
    print("[load] y + LB-best 3-stack")
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)  # this is actually LB-3-stack
                                          # = LB-3-way + RealMLP α=0.20 + nonrule_iso α=0.075

    # Reference: LB-best 4-stack (with xgb_metastack_iso α=0.30)
    mv_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    mv_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
    lb4_o = log_blend([lb3_o, mv_o_iso], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, mv_t_iso], np.array([0.7, 0.3]))
    p_lb4 = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    bal_lb4 = balanced_accuracy_score(y, p_lb4)
    pcr_lb4 = per_class_recall(y, p_lb4)
    errs_lb4 = (p_lb4 != y).sum()
    print(f"\nLB-best 4-stack OOF: {bal_lb4:.6f} (= LB 0.98094)")
    print(f"  PCR: L={pcr_lb4[0]:.5f}  M={pcr_lb4[1]:.5f}  H={pcr_lb4[2]:.5f}")
    print(f"  errs: {errs_lb4}")

    # Sample submission
    sample = pd.read_csv(DATA / "sample_submission.csv")
    primary = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")

    # Test substitution at multiple α (focus on α=0.30 = LB-validated)
    results = {}
    for cand_name, label in [("rawashishsin_2600", "v3 (n_est=2600)"),
                              ("rawashishsin", "v2 (n_est=1500)")]:
        print(f"\n=== {label} ({cand_name}) substitution test ===")
        cand_o = normed(np.load(ART / f"oof_{cand_name}.npy").astype(np.float32))
        cand_t = normed(np.load(ART / f"test_{cand_name}.npy").astype(np.float32))
        cand_o_iso, cand_t_iso = iso_cal(cand_o, cand_t, y)

        # Standalone iso-cal'd at recipe bias (for reference)
        p_std = (np.log(np.clip(cand_o_iso, 1e-12, 1)) + BIAS).argmax(1)
        bal_std = balanced_accuracy_score(y, p_std)
        print(f"  candidate iso-cal'd standalone @ recipe bias: {bal_std:.6f}")

        # Sweep α
        print(f"\n  α-sweep at fixed recipe bias:")
        print(f"  {'α':>6}  {'OOF':>8}  {'Δ vs LB-4':>10}  {'errs':>6}  {'recL':>7} {'recM':>7} {'recH':>7}  {'net_H'}")

        sweep_data = {}
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            w = np.array([1.0 - alpha, alpha])
            sub_o = log_blend([lb3_o, cand_o_iso], w)
            sub_t = log_blend([lb3_t, cand_t_iso], w)
            p = (np.log(np.clip(sub_o, 1e-12, 1)) + BIAS).argmax(1)
            bal = balanced_accuracy_score(y, p)
            pcr = per_class_recall(y, p)
            errs = (p != y).sum()
            # H-direction proxy: vs LB-4 anchor predictions
            blend_is_h = p == 2
            anchor_is_h = p_lb4 == 2
            h_added = (blend_is_h & ~anchor_is_h).sum()
            h_removed = (~blend_is_h & anchor_is_h).sum()
            net_h = int(h_added - h_removed)
            churn = int(h_added + h_removed)
            print(f"  {alpha:>6.3f}  {bal:.6f}  {bal - bal_lb4:+10.5f}  {errs:>6}  {pcr[0]:.5f} {pcr[1]:.5f} {pcr[2]:.5f}  net={net_h:+d}/churn={churn}")
            sweep_data[f"{alpha:.3f}"] = {
                "oof": float(bal),
                "delta_vs_lb4": float(bal - bal_lb4),
                "errs": int(errs),
                "pcr": pcr.tolist(),
                "pcr_delta": (pcr - pcr_lb4).tolist(),
                "net_h": net_h,
                "churn": churn,
            }
            if alpha == 0.30:
                # Save this candidate at α=0.30 (matches LB-best architecture)
                test_pred = (np.log(np.clip(sub_t, 1e-12, 1)) + BIAS).argmax(1)
                sub = sample.copy()
                sub["Irrigation_Need"] = [INT2LABEL[p] for p in test_pred]
                sub_path = SUB / f"submission_subst_{cand_name}_a030.csv"
                sub.to_csv(sub_path, index=False)
                diff = (primary["Irrigation_Need"] != sub["Irrigation_Need"]).sum()
                print(f"\n  [save α=0.30] {sub_path}  diff vs primary: {diff}/{len(sub)} ({100*diff/len(sub):.2f}%)")

        # 4-gate at α=0.30
        d30 = sweep_data["0.300"]
        d40 = sweep_data["0.400"]
        delta_30 = d30["delta_vs_lb4"]
        delta_40 = d40["delta_vs_lb4"]
        g1 = delta_30 >= 3e-4
        pcr_d = np.array(d30["pcr_delta"])
        g2 = (pcr_d >= -5e-4).all()
        g3_ratio = delta_40 / delta_30 if delta_30 > 1e-9 else float("nan")
        g3 = 1.0 <= g3_ratio <= 2.0
        net_h = d30["net_h"]
        churn = d30["churn"]
        g4_ratio = abs(net_h) / max(1, churn)
        g4 = (net_h > 0) and (g4_ratio >= 0.5)

        print(f"\n  4-GATE @ α=0.30 vs LB-best 4-stack:")
        print(f"    G1 (Δ ≥ +3e-4):     {delta_30:+.5f}  {'PASS' if g1 else 'FAIL'}")
        print(f"    G2 (PCR ≥ -5e-4):   {pcr_d.tolist()}  {'PASS' if g2 else 'FAIL'}")
        print(f"    G3 (α0.4/α0.3 ratio): {g3_ratio:.3f}  {'PASS' if g3 else 'FAIL'}")
        print(f"    G4 (net_H>0+ratio≥0.5): net={net_h} ratio={g4_ratio:.3f}  {'PASS' if g4 else 'FAIL'}")
        all_pass = all([g1, g2, g3, g4])
        print(f"    OVERALL: {'PASS — LB-PROBE WARRANTED' if all_pass else 'FAIL — do not LB-probe'}")

        results[cand_name] = {
            "label": label,
            "standalone_iso": float(bal_std),
            "sweep": sweep_data,
            "4gate_a030": {
                "g1": bool(g1), "g2": bool(g2),
                "g3": bool(g3), "g3_ratio": float(g3_ratio),
                "g4": bool(g4), "g4_ratio": float(g4_ratio), "net_h": net_h,
                "all_pass": all_pass,
            },
        }

    # Save results
    out = ART / "d1_rawashishsin_subst_results.json"
    with open(out, "w") as f:
        json.dump({
            "lb_best_4stack_oof": float(bal_lb4),
            "lb_best_4stack_pcr": pcr_lb4.tolist(),
            "lb_best_4stack_errs": int(errs_lb4),
            "results": results,
        }, f, indent=2)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
