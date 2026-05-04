# Example — 4-gate leakage filter

A walked example from the irrigation-water comp showing how the
4-gate filter caught a leakage incident before it cost an LB slot.

## The candidate

`R2 hybrid 0.75` — a hybrid blend candidate selected by maximizing
OOF over a 24-point grid. OOF showed Δ +0.00050 vs the LB-best
4-stack at 0.98094.

## Without the filter

We probed it. LB returned 0.98048 — a regression of −0.00046.
Total 24-point grid search burned one slot for −0.00046.
**Postmortem**: grid-search selection bias. With a 24-point grid,
the OOF-maximum is biased upward by ~5bp; the candidate's "real"
OOF was ~+0.00045 instead of ~+0.00050, but the LB cost was the
~0.00050 inflation.

## With the filter

The 4-gate filter would have run:

```
G1 — Standalone OOF clears anchor at recipe-bias?
  Yes (0.98144 vs 0.98094 anchor)

G2 — Blend with anchor lifts at α* > 0?
  Yes (α*=0.55, lift +0.00010)

G3 — Net rare-class-flip ratio ≥ 0.5?
  Yes (0.62)

G4 — Direction asymmetry: more correct flips than incorrect?
  RESHUFFLE (ratio 0.36 — failed asymmetry threshold of 0.5)
```

**G4 fails**. The candidate makes ~3 wrong flips for every 5 right
flips on the rare class. Net positive, but the flip mix is dilute
— more like reshuffling within the same OOF noise band than adding
a real signal.

The 4-gate verdict: **REJECT**. Don't probe.

## And then minimal-input meta sanity check

Even after R2 was rejected, a second candidate (R5) passed all 4
gates. Before probing:

```
Train R5 candidate meta with ONLY 2 components:
  anchor (LB-best 4-stack) + R5 base
  → 2-component OOF: 0.98088
Anchor at recipe-bias: 0.98094

2-component OOF (0.98088) < anchor (0.98094) → STOP
```

The 17-component R5 meta's apparent +0.00043 OOF lift was
*entirely* cross-component memorization. If the marginal candidate
can't beat the anchor in a 2-component meta, the lift on the full
17-component meta is not orthogonal signal.

## The portable rule

Both checks are <10 minutes of CPU. Both prevent LB regressions
that cost ~−0.0005 to −0.0015. The cost asymmetry is overwhelming:
spend 10 minutes per candidate to avoid a −0.0010 LB cost.

```python
# Pseudocode
def passes_pre_lb_probe(candidate, anchor, bank):
    g1 = candidate_oof_clears_anchor(candidate, anchor)
    g2 = blend_lift_at_alpha_star(candidate, anchor) > 0
    g3 = net_rare_class_flip_ratio(candidate, anchor) >= 0.5
    g4 = direction_asymmetry(candidate, anchor) >= 0.5
    if not (g1 and g2 and g3 and g4):
        return False, "gate failure"

    # Minimal-input meta on stacking candidates
    if is_stacking_candidate(candidate):
        meta_2comp = train_meta(components=[anchor, candidate])
        if meta_2comp.oof < anchor.oof:
            return False, "cross-component memorization"

    return True, "PASS"
```

## What this example teaches

1. **OOF Δ is not enough.** The candidate's standalone OOF lift
   said "go probe". The flip-direction structure said "don't".
2. **Grid-search inflates OOF by ~5bp.** Use theory-only or
   LB-validated defaults for final picks.
3. **Stacking on saturated banks creates phantom lift.** The
   minimal-input meta test catches this in 2-3 minutes.
4. **Cost asymmetry favors paranoia.** A +0.0005 phantom OOF lift
   has expected value −0.0010 LB after gate-failure regression.
   Per LB slot, that's a bad trade.
