# Correction note — fresh-check w5_only verdict downgraded

The 8612bfe diagnostic flagged `submission_4b_plus_w5_only.csv` as the only
"borderline-positive" candidate among 7 untested. Cross-checking against
already-submitted candidates revealed the verdict was incomplete.

## What I missed

**`w5_only` is an exact strict subset of `strict90` (already submitted at
LB 0.98143)**:
  - w5_only = 9 M→H flips
  - strict90 = those exact 9 M→H flips + 38 M→L flips
  - 9 of 9 of w5_only's flips are in strict90

8 of those 9 flips are also in W3_MHonly (LB 0.98127, 147-flip M→H set
that LB-regressed -23bp).

## Implied precision from existing LB results

- W3_MHonly: 147 M→H flips → LB -23bp → implied ~4.9% precision on parent pool
  (well below 9.3% M→H break-even). 8 of w5_only's 9 are in this pool.
- strict90: 9 M→H + 38 M→L → LB -7bp.
  - Per commit 742287f: M→L portion drove ~-9bp at 25-30% precision.
  - Stripping M→L → projected LB **0.98148-0.98152** (tied with 4b).

## Why my prior diagnostic was misleading

The "v1 + bank average" axis-agreement of 0.500 looked promising on paper
(suggesting 25-50% precision band). But W3_MHonly's 147-flip parent set
has measured ~4.9% precision via LB. There's no defensible reason to
expect the 9-flip subset to have systematically higher precision than the
parent pool when the selection criterion is the same direction (M→H) on
the same anchor (4b). The "tighter filter" framing is misleading because:
  - Selection criterion = ImpliedPrecision shifts only when filter axes
    are independent OF the parent precision driver
  - Here: same row pool + same direction = same precision distribution
  - Subset cannot be systematically better just because it's smaller

## Revised verdict

**w5_only**: projected LB **0.98148-0.98152** (tied with 4b within noise).
NOT probe-worthy.

## All 7 untested candidates are now saturated

| Candidate | Status | Reason |
|---|---|---|
| safe3_exf_v1_MtoH | borderline-negative | recipe-family unanimous against, ExF P_H=0.151 low |
| safe4_exf_v1_raw_MtoH | empty | 0 flips structurally |
| exf_v1_3axis_MtoH | regression risk | 19 UNSAFE rows undo 4b's flips |
| exf_v1_raw_4axis_MtoH | borderline-negative | other axes unanimous against |
| **w5_only** | **tied within noise** | **subset of LB-tested strict90 minus M→L** |
| w5_3axis | regression | M→L lineage-correlation rule (742287f) |
| w5_strict85 | regression | M→L lineage-correlation rule (742287f) |

## Recommended action stands

Lock final-selection NOW:
- PRIMARY: `submission_idea4b_selective_override.csv` → LB 0.98150
- HEDGE: `submission_rawashishsin_2600_standalone.csv` → LB 0.98109

Reserve all 4 LB slots. The override mechanism family on this OTHERS
pool is structurally saturated.

## New portable rule

**Subset candidates inherit the precision distribution of their parent
pool.** When evaluating a candidate that's a strict subset of an already-
submitted (regressing) candidate, the implied precision from the parent's
LB is the binding constraint, not OOF-axis-agreement metrics on the
subset alone. The "tighter filter wins" intuition fails when the filter
selects from the same row pool as the parent — selection bias on
post-hoc-chosen axes can manufacture apparent quality without changing
the underlying truth distribution. To break this: the subset must be
selected on an axis genuinely INDEPENDENT of the parent's precision
driver (e.g., orthogonal model class, different feature space, etc.) —
which neither w5_only's filter nor any of the other candidates achieve.
