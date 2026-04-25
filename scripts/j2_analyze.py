"""J2 post-analysis: read results JSON + project LB transfer.

Compares J2 bag-mean meta-stacker against:
  - prior LR meta-stacker (LB 0.97991, gap +0.00176)
  - prior v4 meta-stacker (LB 0.97992, gap +0.00129)
  - LB-best 4-stack (LB 0.98094, gap −0.00010)

Decision rule:
  - if best gated Δ ≥ +5e-4 AND guardrail PASS → recommend LB probe
  - if 0 < Δ < +5e-4 with PASS → defer (sub-threshold, likely null)
  - otherwise → null, recommend lock+hedge-swap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ART = Path("scripts/artifacts")


def main():
    p = ART / "j2_bootstrap_metastack_results.json"
    if not p.exists():
        print("results JSON not yet present; J2 still running")
        return
    d = json.load(open(p))

    print("=" * 70)
    print("J2 BOOTSTRAP META-STACKER — RESULTS")
    print("=" * 70)
    print(f"  bags        : {d['n_bags']}  fraction={d['fraction']}  bag_size={d['bag_size']}")
    print(f"  pool size   : {d['n_components']}")
    print(f"  elapsed     : {d['elapsed_sec']/60:.1f} min")
    print()
    print(f"  LB-best 3-stack OOF                 : {d['lb3_oof']:.5f}")
    print(f"  LB-best 4-stack OOF (anchor)        : {d['lb4_oof']:.5f}")
    print(f"  bag-mean meta_iso standalone OOF    : {d['bag_mean_oof']:.5f}")
    print(f"  Jaccard(bag-mean, LB-best 4-stack)  : {d['bag_jaccard_vs_lb4']:.4f}")
    print(f"  errors: LB4={d['lb4_errs']}  bag={d['bag_errs']}")
    print()
    print("Per-bag iso OOFs (sanity):")
    for bm in d["bag_meta"]:
        print(f"  bag {bm['bag']:2d}  size={len(bm['names']):3d}  feat_dim={bm['feature_dim']}  "
              f"iso_oof={bm['iso_oof_bal']:.5f}  wall={bm['wall_sec']:.0f}s")

    print()
    print("-" * 70)
    print("Strategy A: log-blend bag_iso onto LB-best 4-stack")
    print("-" * 70)
    print(f"  {'alpha':>6} {'OOF':>9} {'Δ':>9} {'errs':>6}  recL    recM    recH    guardrail")
    pcr_lb = (None, None, None)
    # find guardrail floor from the candidate row recall vs LB4 baseline
    # (we can't reconstruct LB4 PCR without rerunning; so report guardrail as PASS/FAIL flag from json)
    for r in d["strategy_A_sweep"]:
        print(f"  {r['alpha']:>6.3f} {r['oof']:>9.5f} {r['delta']:>+9.5f} {r['errs']:>6}  "
              f"{r['recL']:.4f}  {r['recM']:.4f}  {r['recH']:.4f}")
    print()
    print("Strategy B: replace meta_iso with bag_iso at α=0.30")
    sb = d["strategy_B_repl"]
    print(f"  OOF={sb['oof']:.5f}  Δ={sb['delta']:+.5f}  errs={sb['errs']}  "
          f"L={sb['recL']:.4f} M={sb['recM']:.4f} H={sb['recH']:.4f}  "
          f"guardrail={'PASS' if sb['guardrail_pass'] else 'FAIL'}")

    print()
    print("-" * 70)
    print("CALIBRATION-PROJECTION decision")
    print("-" * 70)
    best = d["best"]
    print(f"  best gated   : strat={best['strategy']} α={best['alpha']:.3f} "
          f"Δ={best['delta']:+.5f} guard={'PASS' if best['guardrail_pass'] else 'FAIL'}")
    print(f"  emitted      : {d['emitted'] or '(none — gate failed or SMOKE)'}")

    # Linear-projection prior gap inflation per α (LR/V4 reference).
    # LR meta-stacker: OOF +0.00083 at α=0.50 → LB −0.00103. Inflation = 0.00186.
    #                  Per-unit-α inflation ≈ 0.00372.
    # V4 meta-stacker: OOF +0.00036 at α=0.35 → LB −0.00102. Inflation = 0.00138.
    #                  Per-unit-α inflation ≈ 0.00394.
    # Take 0.0038 as the prior-on-meta-output gap-inflation rate.
    inflation_per_alpha = 0.0038
    if best["guardrail_pass"]:
        a = best["alpha"]
        oof_delta = best["delta"]
        proj_inflation = inflation_per_alpha * a
        proj_lb_delta = oof_delta - proj_inflation
        proj_lb = 0.98094 + proj_lb_delta
        print()
        print(f"  Calibration projection vs LB-best 0.98094:")
        print(f"    α={a:.3f}, OOF Δ={oof_delta:+.5f}")
        print(f"    prior gap-inflation rate (meta-output)  : ~{inflation_per_alpha:.4f} / α")
        print(f"    projected gap inflation at α            : ~{proj_inflation:.5f}")
        print(f"    projected LB Δ vs primary               : {proj_lb_delta:+.5f}")
        print(f"    projected LB                            : ~{proj_lb:.5f}")
        if oof_delta >= 5e-4 and proj_lb_delta > 0:
            verdict = "LB PROBE WORTH"
        elif oof_delta >= 2e-4 and proj_lb_delta >= -1e-4:
            verdict = "BORDERLINE — LB probe optional"
        else:
            verdict = "NULL — projected LB regression, do NOT probe"
    else:
        verdict = "NULL — guardrail FAIL"

    print()
    print(f"  >>> {verdict} <<<")


if __name__ == "__main__":
    main()
