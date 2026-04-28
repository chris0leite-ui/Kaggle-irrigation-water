#!/usr/bin/env python3
"""D: Audit historical meta-stacker variants for circular leakage.

Compute each variant's standalone OOF @ recipe bias + iso-cal'd OOF.
Cross-reference with their LB results (from CLAUDE.md).
Identify which had the most leakage opportunity by comparing to v8
(known-clean-pool: 0.98105 standalone, 0.98115 iso).

Variants audited:
  v1   (LB-validated foundation, 63-component bank, 2026-04-25)
  v3   (cross-pollinate, LB 0.98060 -0.00034, 2026-04-25)
  v4   (ET+kNN bank-extension, LB 0.97992 -0.00102, 2026-04-25)
  v5   (cross-poll v3 with new components, no LB probe)
  varB (depth=3 seed=7 colsample=0.7 high-rounds, no LB probe)
  varC (depth=5 seed=123 colsample=0.5 low-rounds, no LB probe)
  v6   (combined v6 with focal+distill+realmlp_ens4, LB 0.98012 -0.00082)
  v6lb (LB-probed v6 variant)
  v6_combined  (?)
  v7   (?)
  v7b  (?)
  classw  (class-weighted XGB meta, LB 0.98011 -0.00083)
  bag3  (3-seed XGB bag of v1 inputs, LB tested?)
  3wnn  (3-way mlp variant)
  heavy  (depth=2 heavy-reg)
  b2clean (62-component clean pool, no LB probe)
  perfoldiso_inputs (per-fold iso applied, no LB probe)
  n5b_both  (with n5b angle1 + 2 components)
  narrow  (5-component narrow pool, no LB probe)
  v8   (THIS SESSION, LB 0.98074 -0.00020, clean pool)

Output: scripts/artifacts/d_audit_meta_leakage_results.json
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal, load_y, normed)

ART = Path("scripts/artifacts")

# Variants to audit (canonical names that exist on disk)
VARIANTS = [
    ("v1", "xgb_metastack"),
    ("v3", "xgb_metastack_v3"),
    ("v4", "xgb_metastack_v4"),
    ("v5", "xgb_metastack_v5"),
    ("varB", "xgb_metastack_varB"),
    ("varC", "xgb_metastack_varC"),
    ("v6", "xgb_metastack_v6"),
    ("v6lb", "xgb_metastack_v6lb"),
    ("v6_combined", "xgb_metastack_v6_combined"),
    ("v7", "xgb_metastack_v7"),
    ("v7b", "xgb_metastack_v7b"),
    ("classw", "xgb_metastack_classw"),
    ("bag3", "xgb_metastack_bag3"),
    ("3wnn", "xgb_metastack_3wnn"),
    ("heavy", "xgb_metastack_heavy"),
    ("b2clean", "xgb_metastack_b2clean"),
    ("perfoldiso", "xgb_metastack_perfoldiso_inputs"),
    ("n5b_both", "xgb_metastack_n5b_both"),
    ("narrow", "xgb_metastack_narrow"),
    ("v1_cleanpool", "xgb_metastack_v1_cleanpool"),
    ("v1_groupkfold", "xgb_metastack_v1_groupkfold"),
    ("v1_plus_newfe", "xgb_metastack_v1_plus_newfe"),
    ("j2bag", "xgb_metastack_j2bag"),
    ("v8", "xgb_metastack_v8"),
]

# LB results from CLAUDE.md (where known)
LB_RESULTS = {
    "v1":  {"primary_lb": 0.98094, "primary_oof": 0.98084, "gap": -0.00010, "note": "LB-best primary's meta leg"},
    "v3":  {"primary_lb": 0.98060, "primary_oof": 0.98099, "gap": +0.00039, "note": "Cross-poll LB regression"},
    "v4":  {"primary_lb": 0.97992, "primary_oof": 0.98112, "gap": +0.00120, "note": "ET+kNN bank-ext, big gap"},
    "v6":  {"primary_lb": 0.98012, "primary_oof": 0.98109, "gap": +0.00097, "note": "Combined v6 LB regression"},
    "classw": {"primary_lb": 0.98011, "primary_oof": 0.98030, "gap": +0.00019, "note": "Class-weighted regression"},
    "v8":  {"primary_lb": 0.98074, "primary_oof": 0.98125, "gap": +0.00051, "note": "Clean-pool, BEST carryover -0.49x"},
}


def per_class_recall(y, pred):
    return np.array([(pred[y == c] == c).mean() for c in range(3)])


def main():
    print("[load] y + LB-best 3-stack anchor")
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    anchor_p = (np.log(np.clip(lb3_o, 1e-12, 1)) + BIAS).argmax(1)
    anchor_oof = balanced_accuracy_score(y, anchor_p)
    anchor_pcr = per_class_recall(y, anchor_p)
    print(f"  LB-best 3-stack OOF (anchor): {anchor_oof:.6f}")
    print(f"  PCR: L={anchor_pcr[0]:.5f} M={anchor_pcr[1]:.5f} H={anchor_pcr[2]:.5f}")

    # Reference: v1's meta-stacker (xgb_metastack) iso-cal'd at α=0.30 = LB-best 4-stack
    mv_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    mv_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
    lb4_o = log_blend([lb3_o, mv_o_iso], np.array([0.7, 0.3]))
    lb4_p = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    lb4_oof = balanced_accuracy_score(y, lb4_p)
    print(f"  LB-best 4-stack OOF (reference): {lb4_oof:.6f} (= 0.98094 LB)")
    print()

    print(f"{'variant':<20} {'standalone':<12} {'iso':<12} {'4stack@a30':<12} {'errs':<8} {'recH':<8} {'jaccard'}")
    print("-" * 110)

    rows = []
    for variant_name, file_name in VARIANTS:
        oof_path = ART / f"oof_{file_name}.npy"
        test_path = ART / f"test_{file_name}.npy"
        if not (oof_path.exists() and test_path.exists()):
            continue
        try:
            o = normed(np.load(oof_path).astype(np.float32))
            t = normed(np.load(test_path).astype(np.float32))
        except Exception as e:
            print(f"  SKIP {variant_name}: {e}")
            continue
        if o.shape != (630_000, 3):
            continue

        # Standalone @ recipe bias
        p_raw = (np.log(np.clip(o, 1e-12, 1)) + BIAS).argmax(1)
        bal_raw = balanced_accuracy_score(y, p_raw)
        errs_raw = (p_raw != y).sum()
        pcr_raw = per_class_recall(y, p_raw)

        # iso-cal'd standalone
        try:
            o_iso, t_iso = iso_cal(o, t, y)
            p_iso = (np.log(np.clip(o_iso, 1e-12, 1)) + BIAS).argmax(1)
            bal_iso = balanced_accuracy_score(y, p_iso)
        except Exception:
            o_iso = None
            bal_iso = float("nan")

        # 4-stack equivalent: LB-3-stack × 0.7 + variant_iso × 0.3
        if o_iso is not None:
            lb4_eq_o = log_blend([lb3_o, o_iso], np.array([0.7, 0.3]))
            p_lb4_eq = (np.log(np.clip(lb4_eq_o, 1e-12, 1)) + BIAS).argmax(1)
            bal_lb4_eq = balanced_accuracy_score(y, p_lb4_eq)
            errs_lb4_eq = (p_lb4_eq != y).sum()
            pcr_lb4_eq = per_class_recall(y, p_lb4_eq)
            # Jaccard vs LB-best 4-stack
            err_v = p_lb4_eq != y
            err_lb4 = lb4_p != y
            jac = (err_v & err_lb4).sum() / max(1, (err_v | err_lb4).sum())
        else:
            bal_lb4_eq = float("nan")
            errs_lb4_eq = -1
            pcr_lb4_eq = np.zeros(3)
            jac = float("nan")

        print(f"{variant_name:<20} {bal_raw:<12.5f} {bal_iso:<12.5f} {bal_lb4_eq:<12.5f} {errs_lb4_eq:<8} {pcr_lb4_eq[2]:<8.5f} {jac:.4f}")

        rows.append({
            "variant": variant_name,
            "file": file_name,
            "standalone_oof": float(bal_raw),
            "iso_oof": float(bal_iso),
            "lb4_eq_oof": float(bal_lb4_eq),  # variant iso-cal'd α=0.30 onto LB-3-stack
            "lb4_eq_errs": int(errs_lb4_eq),
            "lb4_eq_pcr": pcr_lb4_eq.tolist(),
            "jaccard_vs_lb4_best": float(jac),
            "lb": LB_RESULTS.get(variant_name),
        })

    print()
    print("=== INTERPRETATION ===")
    print(f"v1 (LB-validated foundation):")
    v1 = next((r for r in rows if r["variant"] == "v1"), None)
    if v1:
        print(f"  standalone {v1['standalone_oof']:.5f}  iso {v1['iso_oof']:.5f}  → LB 4-stack 0.98094")
    v8 = next((r for r in rows if r["variant"] == "v8"), None)
    if v8:
        print(f"v8 (clean-pool, this session): standalone {v8['standalone_oof']:.5f}  iso {v8['iso_oof']:.5f}  → LB 0.98074")
    print()
    print("Variants ranked by OOF (lb4_eq, descending):")
    for r in sorted(rows, key=lambda r: r["lb4_eq_oof"], reverse=True):
        lb_note = ""
        if r["lb"]:
            lb_note = f"  → LB {r['lb']['primary_lb']:.5f}  (gap {r['lb']['gap']:+.5f})"
        print(f"  {r['variant']:<18} lb4_eq={r['lb4_eq_oof']:.5f}{lb_note}")

    print()
    print("=== LEAKAGE DIAGNOSIS ===")
    print("If circular leakage was a major contributor, expect:")
    print("  - Variants with bigger banks → higher OOF (more leakage)")
    print("  - v8 (clean) → significantly lower OOF than v6 (with prior metas in bank)")
    print()
    if v8 and any(r["variant"] == "v6" for r in rows):
        v6 = next(r for r in rows if r["variant"] == "v6")
        delta = v6["lb4_eq_oof"] - v8["lb4_eq_oof"]
        print(f"v6 lb4_eq: {v6['lb4_eq_oof']:.5f}  vs  v8 lb4_eq: {v8['lb4_eq_oof']:.5f}")
        print(f"Δ (v6 - v8): {delta:+.5f}")
        if abs(delta) < 0.0003:
            print(f"→ Leakage magnitude is SMALL ({abs(delta)*100000:.1f}bp).")
            print(f"  Heavy-reg XGB (depth=4 + reg_alpha=5 + reg_lambda=5) is robust to circular features.")
        else:
            print(f"→ Leakage magnitude appears non-trivial ({delta*100000:.1f}bp).")

    out = ART / "d_audit_meta_leakage_results.json"
    with open(out, "w") as f:
        json.dump({
            "anchor_oof": float(anchor_oof),
            "lb_best_4stack_oof": float(lb4_oof),
            "rows": rows,
        }, f, indent=2)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
