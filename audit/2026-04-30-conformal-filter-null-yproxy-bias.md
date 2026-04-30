# 2026-04-30 — conformal-filter LB-null + y-proxy-bias lesson

**LB result: `submission_98150_drop_lm.csv` → LB 0.98148 (Δ −0.00002 vs LB-best 0.98150)**

Small regression. Falls in the worst-case range of the predicted ladder
(0.98140 worst → 0.98166 best, median 0.98155, no-lift 0.98150).

## Mechanism

Applied train-side direction-precision diagnostic (per `bdf4a1b`) to the
new LB-best 0.98150 anchor. The diagnostic flagged L→M flips at 7.9%
precision << 39.3% break-even on TRAIN (using strict-independent
5-model y-proxy: `xgb_corn`, `xgb_dist_digits`, `recipe_full_te_catboost`,
`xgb_dist_routed_v3`, `lgbm_te_orig`).

Reverted the 176 L→M flips inherited from B (LB 0.98140) in the new 4b
LB-best (LB 0.98150), keeping all other directions intact.

Predicted: +0.000163 macro / +0.00016 LB (assuming 7.9% precision held).
Actual: −0.00002 LB.

## Diagnosis: strict-independent y-proxy underestimates override precision

The 5 y-proxy models are STRUCTURALLY weaker than the OTHERS in the
override mechanism (`rawashishsin v3` LB 0.98109, `tier1b 4-stack`
LB 0.98094). On L→M boundary rows, the weaker y-proxy models default
to predicting Low (the dominant class on rule-Low cells), making the
override APPEAR wrong (5/5 say Low) when in fact the override is
correctly catching the rare flip-to-Medium signal.

**Implied test L→M precision: ~30-40%** (near 39.3% break-even). Not
below break-even as the y-proxy suggested.

## Portable rule (LEARNINGS.md candidate)

**"Strict-independent y-proxy with weaker models systematically
underestimates override precision when the overriding mechanism uses
STRONGER models."** Direction-precision estimates from such y-proxies
should be treated as LOWER BOUNDS, not point estimates. For boundary-row
overrides in particular, the y-proxy's tendency to default to the
majority class biases precision estimates downward by 20-30 percentage
points. To use direction-precision diagnostics safely:

  1. Use a y-proxy at LEAST as strong as the override mechanism's OTHERS
  2. Or use multiple y-proxy strengths and take the MAX (not min/median)
  3. Or treat y-proxy precision as a LOWER BOUND and require margin
     above break-even of at least the y-proxy's own error rate

The LB 0.98140/0.98150 override mechanism is structurally sound on all
4 directions including L→M (each direction either at or above
break-even on real test). The mechanism's `{raw, tier1b}` k=2 unanimous
filter is already an effective precision gate; layering an additional
direction-prune filter over-cleaned the signal.

## Updated lever status

- SPEC1, SPEC2, SPEC3 (all build on top of L→M drop logic): predicted LB
  ≤ 0.98140. **Skip — share the failure mode.**
- LB 0.98150 / 4b mechanism: structurally complete, all directions are
  at or above break-even. No further direction-pruning expected to help.

## LB ladder (updated)

```
LB 0.98150  submission_idea4b_selective_override.csv  ← LB-best (UNCHANGED)
LB 0.98148  submission_98150_drop_lm.csv              ← THIS PROBE
LB 0.98148  PACK (median of top 100)
LB 0.98140  submission_2other_raw_tier1b_k2.csv (B)
```

LB budget today: 5/10 used (B + TC1 + W3_MHonly + 4b + this), 5 remaining.

## Final-selection recommendation (UNCHANGED)

- **PRIMARY**: `submission_idea4b_selective_override.csv` → LB 0.98150
- **HEDGE**: `submission_sklearn_rf_meta_natural_standalone.csv` →
  LB 0.98129 (orthogonal failure mode — base meta-stacker without
  override layer)
