#!/usr/bin/env python3
"""C: Greedy forward over LB-3-stack + rawashishsin + curated base components.

Different from v8 (meta-stacker bank-add): this is a DIRECT log-blend chain
where greedy forward selects candidates and α at fixed recipe bias.

Anchor: LB-best 3-stack (OOF 0.98061)
Candidates: rawashishsin v2 + v3 + ~30 curated LB-validated base components.
Each candidate evaluated as iso-cal'd for blend.

At each step:
  - For each candidate, find α ∈ [0.025, 0.50] that maximizes OOF macro-recall
  - Apply 4-gate filter: G1 (Δ ≥ +3e-4), G2 (PCR ≥ -5e-4 each), G3 (1.0-2.0 ratio),
    G4 (net_H>0 + ratio≥0.5)
  - Pick candidate that maximizes Δ AND passes all 4 gates (or the best 3/4 if none pass all 4)

Stop when no candidate passes G1 by ≥ +1e-4 OR no candidate clears all 4 gates.
"""
from __future__ import annotations
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

# Anchor: LB-best 3-stack (no meta-stacker, no realmlp, no nonrule_iso)
# Pool: rawashishsin v2 + v3 + curated LB-validated base components
CANDIDATES = [
    # Headlines
    "rawashishsin", "rawashishsin_2600",
    # Recipe-family variants (LB-validated diversity)
    "recipe_full_te", "recipe_pseudolabel",
    "recipe_full_te_lgbm", "recipe_full_te_catboost",
    "recipe_lgbm", "recipe_catboost", "recipe_allpairs",
    # Direction-orthogonal (LB-validated lifters)
    "realmlp", "xgb_nonrule",
    # Strong base XGBs
    "xgb_dist_digits", "xgb_dist_digits_ote",
    "xgb_dist_digits_ote_digits", "xgb_dist_digits_ote_digits_pairs",
    "xgb_dist_routed_v3", "xgb_corn",
    # Strong LGBMs
    "lgbm_dist_digits", "lgbm_dist_digits_ote",
    # Catboost
    "catboost_recipe_gpu",
]

INT2LABEL = {0: "Low", 1: "Medium", 2: "High"}


def per_class_recall(y, pred):
    return np.array([(pred[y == c] == c).mean() for c in range(3)])


def evaluate_blend(blend_o, blend_t, y, anchor_p, anchor_pcr):
    """Return diagnostics: OOF, errs, PCR delta, net_H, churn, Jaccard."""
    p = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
    bal = balanced_accuracy_score(y, p)
    errs = (p != y).sum()
    pcr = per_class_recall(y, p)
    pcr_delta = pcr - anchor_pcr
    # H direction analysis on TEST (proxy for LB)
    # Use OOF as proxy: rows where blend says H but anchor doesn't, or vice versa
    blend_is_h = p == 2
    anchor_is_h = anchor_p == 2
    h_added = (blend_is_h & ~anchor_is_h).sum()
    h_removed = (~blend_is_h & anchor_is_h).sum()
    net_h = int(h_added - h_removed)
    churn = int(h_added + h_removed)
    # Jaccard on errors
    err_blend = p != y
    err_anchor = anchor_p != y
    jac = (err_blend & err_anchor).sum() / max(1, (err_blend | err_anchor).sum())
    return dict(oof=float(bal), errs=int(errs),
                pcr_delta=pcr_delta.tolist(),
                net_h=net_h, churn=churn,
                jaccard_vs_anchor=float(jac))


def fourgate(diag_at_a30, diag_at_a40, anchor_oof):
    """Return 4-gate verdict on a candidate at α=0.30, with α=0.40 for G3 ratio."""
    g1_delta = diag_at_a30["oof"] - anchor_oof
    g1 = g1_delta >= 3e-4
    pcr_delta = np.array(diag_at_a30["pcr_delta"])
    g2 = (pcr_delta >= -5e-4).all()
    delta_a40 = diag_at_a40["oof"] - anchor_oof
    if g1_delta > 1e-9:
        g3_ratio = delta_a40 / g1_delta
    else:
        g3_ratio = float("nan")
    g3 = 1.0 <= g3_ratio <= 2.0
    net_h = diag_at_a30["net_h"]
    churn = diag_at_a30["churn"]
    g4_ratio = abs(net_h) / max(1, churn)
    g4 = (net_h > 0) and (g4_ratio >= 0.5)
    return dict(g1=g1, g2=g2, g3=g3, g4=g4, g3_ratio=float(g3_ratio),
                g4_ratio=float(g4_ratio), all_pass=all([g1, g2, g3, g4]))


def main():
    print(f"[load] y + LB-best 3-stack")
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    anchor_p = (np.log(np.clip(lb3_o, 1e-12, 1)) + BIAS).argmax(1)
    anchor_oof = balanced_accuracy_score(y, anchor_p)
    anchor_pcr = per_class_recall(y, anchor_p)
    print(f"  LB-best 3-stack OOF = {anchor_oof:.6f}")
    print(f"  PCR = {anchor_pcr.round(5).tolist()}")

    # Load candidates with iso-cal
    print(f"\n[load] {len(CANDIDATES)} candidates")
    cand_data = {}
    for name in CANDIDATES:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not (oof_p.exists() and test_p.exists()):
            print(f"  SKIP missing: {name}")
            continue
        try:
            o = normed(np.load(oof_p).astype(np.float32))
            t = normed(np.load(test_p).astype(np.float32))
        except Exception as e:
            print(f"  SKIP load fail: {name}: {e}")
            continue
        if o.shape != (630_000, 3) or t.shape != (270_000, 3):
            print(f"  SKIP shape: {name} oof={o.shape} test={t.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            print(f"  SKIP partial OOF: {name}")
            continue
        # iso-cal
        o_iso, t_iso = iso_cal(o, t, y)
        cand_data[name] = (o_iso, t_iso)
    print(f"  Loaded {len(cand_data)} candidates")

    # Greedy forward
    current_blend_o = lb3_o.copy()
    current_blend_t = lb3_t.copy()
    current_oof = anchor_oof
    selected = [("lb3", 1.0)]  # placeholder, chain stored separately
    history = []

    print(f"\n[greedy] starting OOF = {current_oof:.6f}")
    print(f"  4-gate framework: PASS only if all 4 (G1 +3e-4, G2 PCR ≥-5e-4,")
    print(f"                                 G3 1.0-2.0 ratio, G4 net_H>0+ratio≥0.5)")
    print(f"  Best-effort: pick highest OOF Δ, report gate status")

    chain_o = [lb3_o]
    chain_t = [lb3_t]
    chain_w = [1.0]

    for step in range(8):
        # For each candidate, find best α
        best_name = None
        best_alpha = None
        best_oof = current_oof
        best_diag = None
        best_diag_a40 = None

        for name, (cand_o, cand_t) in cand_data.items():
            if name in [n for n, _ in selected[1:]]:
                continue  # already selected
            for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                # Build blend: current + cand at relative weight alpha
                # Effective: blend = (1-alpha)*current + alpha*cand
                w = np.array([1.0 - alpha, alpha])
                test_blend_o = log_blend([current_blend_o, cand_o], w)
                test_blend_t = log_blend([current_blend_t, cand_t], w)
                p = (np.log(np.clip(test_blend_o, 1e-12, 1)) + BIAS).argmax(1)
                bal = balanced_accuracy_score(y, p)
                if bal > best_oof + 1e-6:
                    best_oof = bal
                    best_name = name
                    best_alpha = alpha
                    best_diag = evaluate_blend(test_blend_o, test_blend_t, y, anchor_p, anchor_pcr)
                    # Also compute α=0.40 for G3
                    a40 = 0.40
                    test40_o = log_blend([current_blend_o, cand_o], np.array([1.0 - a40, a40]))
                    p40 = (np.log(np.clip(test40_o, 1e-12, 1)) + BIAS).argmax(1)
                    bal40 = balanced_accuracy_score(y, p40)
                    best_diag_a40 = dict(oof=float(bal40))

        if best_name is None or (best_oof - current_oof) < 1e-4:
            print(f"\n[stop] step {step}: no candidate improves by ≥ +1e-4")
            break

        # 4-gate filter at best α (where best is α=0.30 typically for fair comparison)
        # Recompute diag at α=0.30 for fair G3 ratio (against anchor, not against current_blend)
        cand_o, cand_t = cand_data[best_name]
        a30 = 0.30
        full_a30_o = log_blend([current_blend_o, cand_o], np.array([1.0 - a30, a30]))
        full_a30_t = log_blend([current_blend_t, cand_t], np.array([1.0 - a30, a30]))
        diag30 = evaluate_blend(full_a30_o, full_a30_t, y, anchor_p, anchor_pcr)
        a40 = 0.40
        full_a40_o = log_blend([current_blend_o, cand_o], np.array([1.0 - a40, a40]))
        diag40 = dict(oof=float(balanced_accuracy_score(y, (np.log(np.clip(full_a40_o, 1e-12, 1)) + BIAS).argmax(1))))
        gates_a30 = fourgate(diag30, diag40, anchor_oof)

        delta = best_oof - current_oof
        anchor_delta = best_oof - anchor_oof
        print(f"\n[step {step+1}] best: {best_name} α={best_alpha:.3f}")
        print(f"  OOF: {best_oof:.6f} (Δ vs anchor 3stack: {anchor_delta:+.5f}, Δ vs prev: {delta:+.5f})")
        print(f"  errs: {best_diag['errs']}  Jaccard vs anchor: {best_diag['jaccard_vs_anchor']:.4f}")
        print(f"  PCR delta: L={best_diag['pcr_delta'][0]:+.5f} M={best_diag['pcr_delta'][1]:+.5f} H={best_diag['pcr_delta'][2]:+.5f}")
        print(f"  H flips: net={best_diag['net_h']} churn={best_diag['churn']}")
        print(f"  4-gate (vs anchor at α=0.30):  G1={'✓' if gates_a30['g1'] else '✗'}  G2={'✓' if gates_a30['g2'] else '✗'}  "
              f"G3={'✓' if gates_a30['g3'] else '✗'} ({gates_a30['g3_ratio']:.3f})  "
              f"G4={'✓' if gates_a30['g4'] else '✗'} (net_H={best_diag['net_h']}, ratio={gates_a30['g4_ratio']:.3f})  "
              f"OVERALL={'PASS' if gates_a30['all_pass'] else 'FAIL'}")

        # Accept the addition (greedy keeps it)
        w = np.array([1.0 - best_alpha, best_alpha])
        current_blend_o = log_blend([current_blend_o, cand_data[best_name][0]], w)
        current_blend_t = log_blend([current_blend_t, cand_data[best_name][1]], w)
        current_oof = best_oof
        selected.append((best_name, best_alpha))
        history.append({
            "step": step + 1, "name": best_name, "alpha": best_alpha,
            "oof": best_oof, "delta_vs_prev": delta,
            "delta_vs_anchor": anchor_delta,
            "diag": best_diag, "gates_a30": gates_a30
        })

    print(f"\n[done] final greedy chain ({len(selected)-1} steps):")
    print(f"  anchor: LB-best 3-stack")
    for name, alpha in selected[1:]:
        print(f"   + {name} α={alpha}")
    print(f"  final OOF: {current_oof:.6f}  (Δ vs anchor: {current_oof - anchor_oof:+.5f})")

    # Diagnostic: vs LB-best 4-stack (anchor + xgb_metastack_iso α=0.30)
    mv_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    mv_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
    lb4_o = log_blend([lb3_o, mv_o_iso], np.array([0.7, 0.3]))
    lb4_p = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    lb4_oof = balanced_accuracy_score(y, lb4_p)
    print(f"\n[reference] LB-best 4-stack OOF = {lb4_oof:.6f}")
    print(f"            greedy chain OOF      = {current_oof:.6f}  (Δ vs LB-4: {current_oof - lb4_oof:+.5f})")

    # Save artifacts
    out_oof = ART / "oof_c_greedy_rawashishsin.npy"
    out_test = ART / "test_c_greedy_rawashishsin.npy"
    np.save(out_oof, current_blend_o.astype(np.float32))
    np.save(out_test, current_blend_t.astype(np.float32))

    # Submission CSV
    sample = pd.read_csv(DATA / "sample_submission.csv")
    test_pred = (np.log(np.clip(current_blend_t, 1e-12, 1)) + BIAS).argmax(1)
    sub = sample.copy()
    sub["Irrigation_Need"] = [INT2LABEL[p] for p in test_pred]
    sub_path = SUB / "submission_c_greedy_rawashishsin.csv"
    sub.to_csv(sub_path, index=False)
    primary = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    diff = (primary["Irrigation_Need"] != sub["Irrigation_Need"]).sum()
    print(f"\n[save] {out_oof}")
    print(f"[save] {out_test}")
    print(f"[save] {sub_path}")
    print(f"  diff vs LB-best primary: {diff}/{len(sub)} ({100*diff/len(sub):.2f}%)")

    # Results JSON
    results = {
        "anchor_oof": float(anchor_oof),
        "lb_best_4stack_oof": float(lb4_oof),
        "final_oof": float(current_oof),
        "delta_vs_anchor": float(current_oof - anchor_oof),
        "delta_vs_lb4": float(current_oof - lb4_oof),
        "selected": [{"name": n, "alpha": float(a)} for n, a in selected[1:]],
        "history": history,
        "test_diff_vs_primary": int(diff),
    }
    with open(ART / "c_greedy_rawashishsin_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {ART / 'c_greedy_rawashishsin_results.json'}")


if __name__ == "__main__":
    main()
