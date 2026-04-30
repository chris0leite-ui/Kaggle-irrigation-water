"""Print a clean text summary of all path results for user review."""
import json, sys
from pathlib import Path
ART = Path("scripts/artifacts")

def load_json(p):
    if (ART / p).exists():
        return json.loads((ART / p).read_text())
    return None

print("=" * 60)
print("MULTI-PATH EXECUTION SUMMARY")
print("=" * 60)
print(f"\nLB-best PRIMARY: 0.98129 (v1 RF natural meta)")
print(f"LB-best HEDGE:   0.98109 (rawashishsin v3)")
print()

# Path 4
p4 = load_json("path4_conformal_results.json")
print("Path 4 — Conformal-set selective override")
print("  STATUS: NULL")
print("  Mechanism: per-row conformal prediction sets at multiple α levels.")
print("  Verdict: set-based gating reduces to max_prob threshold (already")
print("           failed in N2 router). Strict null at every α.")
print()

# Path 5
p5 = load_json("path5_l3_rf_minimal_results.json")
if p5:
    print("Path 5 — L3 RF natural on (v1, rawashishsin, dist) minimal features")
    print(f"  Standalone tuned: {p5['l3_tuned']:.5f} (v1 = {p5['v1_tuned']:.5f}, Δ = {p5['delta_tuned']:+.5f})")
    print(f"  Drift max: {max(abs(d) for d in p5['l3_drift']):.2f}  (≤0.40 = PASS)")
    print(f"  PCR delta: {p5['delta_pcr']}")
    print(f"  Test diff vs v1: {p5['n_diff_v1_test']} rows (0.09%)")
    n_pass = p5["gate"]["n_passing"]
    print(f"  4-gate blend sweep: {n_pass}/5 alphas pass all gates")
    print(f"  STATUS: NO LB-PROBE (G2 FAIL on standalone, blends REMOVE-H)")
print()

# Path 2
p2 = load_json("rf_meta_seedbag_results.json")
print("Path 2 — RF natural META seed-bag")
print("  STATUS: SKIPPED — redundant with H1 (CLAUDE.md 2026-04-29)")
print("  Prior: H1 with 3 seeds {42,7,123} of v1's exact arch produced")
print("  13-row diff vs v1 (RF near-deterministic at this config).")
print()

# Path 3
print("Path 3 — LightGBM rawashishsin-parity natural-cal clone")
print("  STATUS: SKIPPED — existing recipe_full_te_lgbm_skte already has")
print("  ORIG_ROW_WEIGHT=0.5 and is in a1+plus bank that LB-regressed.")
print()

# Path 1 / T4
import os
t4_oof = ART / "oof_rawashishsin_pseudo.npy"
t4_results = load_json("rawashishsin_pseudo_results.json")
print("Path 1 — T4 rawashishsin pseudo-label kernel (Kaggle GPU)")
if t4_results:
    print(f"  Standalone tuned: {t4_results.get('tuned_log_bias_bal_acc', '?')}")
    print(f"  Bias: {t4_results.get('log_bias', '?')}")
    print(f"  STATUS: COMPLETE — see emit_lb_candidate.py t4_pseudo")
elif t4_oof.exists():
    print("  STATUS: artifacts present, awaiting analysis")
else:
    print("  STATUS: RUNNING on Kaggle GPU (~70 min wall)")
print()
print("=" * 60)
