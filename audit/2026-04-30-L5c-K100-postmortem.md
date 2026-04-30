# 2026-04-30 — L5c K=100 union LB-probe: 47th saturation, −0.00145 regression

## Result

- Submission: `submission_L5c_4bUnionK100.csv`
- Projected LB: 0.98172 (+0.00022 vs 4b)
- **Actual LB: 0.98005 (−0.00145 vs 4b's 0.98150)**
- OOF→LB drift: **−167 bp below projection**, **−225 bp on H→M precision**

## Mechanism (recap)

Union of 4b's 108 flips + 202 high-bank-confidence focal-disagreer
H→M additions. The 202 new flips were selected by:

1. Focal-majority disagreer (4 focal-loss XGB OOFs, all ≠ B's class)
2. raw + tier1b + bank-majority all agree on the new class
3. **Bank-mean probability of new class ≥ 0.80** (the L5b rank cutoff)

OOF empirical H→M precision at this threshold: 93.8% (n=100).
Implied LB H→M precision from the −0.00145 regression: ~71.3%.

## Root cause

**Anchor mismatch between OOF and TEST.**

The OOF probe used `oof_rawashishsin_2600.argmax` as the anchor (the
"flip from this class to focal-majority's class" baseline). The TEST
probe applied to `submission_2other_raw_tier1b_k2` (B at LB 0.98140),
which is a fundamentally different anchor — B is built from raw +
tier1b k=2 unanimous, NOT from rawashishsin alone.

The flips that pass `(rawashishsin != focal_maj) & raw=focal_maj &
tier1b=focal_maj & bank_maj=focal_maj` on OOF are NOT the same set
of rows as flips that pass `(B != focal_maj) & ...` on TEST. OOF
flip rows are systematically EASIER (rawashishsin alone disagrees
with consensus often, including on rows that all 14 banks confidently
get right). TEST flip rows are HARDER (B is itself a near-consensus
mechanism, so B-disagreements are concentrated on genuine boundary
ambiguity).

The 14-bank-mean probability is calibrated on OOF flip rows (easy)
but applied to TEST flip rows (hard). The probability calibration
breaks on the harder set — bank says p_M ≥ 0.80 on test rows where
the true label is actually H ~30% of the time.

## Portable lessons (add to LEARNINGS.md)

### Pattern: OOF-anchor / TEST-anchor mismatch in override mechanisms

When validating an override mechanism on OOF using anchor A_oof but
applying it on TEST using anchor A_test ≠ A_oof:

- The flip-row distributions differ systematically. A_oof
  disagrees with consensus on relatively easy rows; A_test on harder
  ones (because A_test is itself a near-consensus mechanism).
- Bank-confidence calibration learned on OOF flip rows DOES NOT
  transfer to TEST flip rows — even when the "filter" appears
  precision-monotonic on OOF, the calibration is conditional on the
  flip-set distribution.
- Sympmptom: OOF→LB drift is much larger than the LEARNINGS rule's
  documented 3–50 bp inflation for "knob chosen by maximizing OOF."
  Here it was 167 bp on overall LB, 225 bp on the H→M direction.

**Rule**: validate on OOF using the SAME anchor that will be applied
on TEST. If TEST anchor is a submission CSV (B), the OOF analog
must be B's per-row argmax on training data — reconstruct B's OOF
mechanism from its inputs, don't substitute a similar component.

### Pattern: bank-mean probability is unreliable on boundary rows

The 14-bank-mean probability has good calibration GLOBALLY (98.7%
overall accuracy) but **breaks on boundary rows where the bank's
own predictions disagree with a near-consensus anchor**. Boundary
rows are precisely where the bank's individual components disagree;
their probability mass is split, but bank-mean averaging artificially
"sharpens" toward the majority view, OVERESTIMATING confidence on
exactly the rows where confidence should be lowest.

**Rule**: never use bank-mean probability as a confidence ranking
for override-mechanism candidates. Use bag-fold CV-confidence (e.g.,
bagged_v1's per-fold-seed disagreement rate) instead, which is
inherently boundary-aware.

### Saturation count: 47

This is the 47th independent saturation at LB 0.98150. The L1–L5
chain establishes that:

- Loss-function variation on saturated bank: NULL (L1)
- Loss-variant model as disagreer source: NULL with strong precision
  degradation (L2/L3)
- Tightened consensus on disagreer: NULL, plateau at 85% precision
  (L4)
- Bank-confidence-ranked disagreer additions: **catastrophically NULL
  on LB despite strong OOF signal — 47th saturation** (L5/L5b/L5c)

## Recommendation going forward

**DO NOT submit L5c K=50 or K=200** — same anchor-mismatch failure
mode, expected to regress similarly. The L5 family is closed.

Current LB-best remains `submission_idea4b_selective_override.csv`
at 0.98150.
