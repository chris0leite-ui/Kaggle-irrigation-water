# 2026-04-30 — 42nd saturation: 3-mechanism CPU sweep on top of 4b

Final-day analytical sweep across three structurally-distinct mechanisms
on top of 4b (LB 0.98150). All NULL on TRAIN-OOF 4-gate. No LB probe.

## (1) idea4d k4-unanimous orthogonal-bank candidate (163 untested flips)

idea4d generated 163 candidate flips from a 4-component lineage-orthogonal
bank (xgb_corn, recipe_macrorec, recipe_basemargin_K2, recipe_residte) at
k=4 unanimous, dropping the raw+tier1b unanimity gate. Projected LB at 65%
precision: 0.98155 (+0.00005).

**TRAIN OOF 4-gate verdict** (`scripts/idea4d_train_oof_precision.py`):
```
direction       n  P(true=bank)  break-even   verdict
M->L           80     0.362         0.614     FAIL
L->M           34     0.235         0.386     FAIL
M->H           46     0.152         0.092     PASS  (n_test=0)
H->M            6     0.333         0.908     FAIL
```
Projected LB at TRAIN-OOF-derived precision: **0.98128** (−0.00022 vs 4b).

Conclusion: orthogonal-bank k4-unanimous WITHOUT raw+tier1b confirmation is
~30pp below break-even on the M→L direction. Closed.

## (2) Cross-seed-stability filter on 4b's 108 test-side flips

For each of 4b's 108 flips, count how many of 3 individual-seed RF natural
predictions (fs7, fs42 n=1000, fs123) agree with the flip class (proxy for
"seed-stable" signal beneath bagged_v1' mean).

```
3/3 seeds agree (seed-stable):  106
2/3 seeds agree (seed-loose):     0
1/3 seeds agree (seed-fragile):   0
0/3 seeds agree (seed-against):   2  (both are L->M)
```

Only 2 candidate-drop flips identified, both in L→M direction. Expected LB
delta from dropping: ±0.0000080 (within noise). Not actionable.

Conclusion: 4b's filter is essentially seed-stable already; the bagged-mean
argmax matches per-seed argmax on 98.1% of flips. Cross-seed stability is
not an exploitable lever above the existing 4b filter.

## (3) Stratified TRAIN-OOF precision profile of analogous 4b H→M decisions

Profile (`scripts/profile_4b_flip_precision.py`) measured natural-cal H→M
analog (tier1b=H, bagged=M, raw=M) precision on TRAIN OOF across strata:

```
Overall (raw=M):                    n=64   P(true=M)=0.641
By bagged_pm range:
  bagged_pm in [0.50, 0.70):        n=23   P(true=M)=0.652
  bagged_pm in [0.70, 0.85):        n=40   P(true=M)=0.650
  bagged_pm in [0.85, 0.95):        n=1    P(true=M)=0.000  (n too small)
By DGP score:
  score=5:                          n=4    P(true=M)=0.750
  score=6:                          n=42   P(true=M)=0.571
  score=7:                          n=12   P(true=M)=0.667
  score=8:                          n=6    P(true=M)=1.000  (n=6, low-power)
By 4-extra-bank agreement on M:
  extra-bank-M = 0/4:               n=10   P(true=M)=0.500
  extra-bank-M = 1/4:               n=35   P(true=M)=0.714
  extra-bank-M = 2/4:               n=9    P(true=M)=0.778
  extra-bank-M = 3/4:               n=6    P(true=M)=0.500
  extra-bank-M = 4/4:               n=4    P(true=M)=0.250
```

ALL strata FAIL the 0.908 H→M break-even on TRAIN OOF. But 4b achieves
~95.6% precision on its 88 H→M flips on TEST (per LB lift back-out).

## Portable rule (LEARNINGS.md candidate)

**Natural-cal H→M TEST precision is ~30pp above its TRAIN-OOF analog
precision; TRAIN-OOF stratification cannot rank candidate flips by LB EV
in this regime.**

The asymmetry is structural — model strength on full-data (test
prediction time) is dramatically above CV-fold strength on borderline
rows. This means tightening or extending the 4b override family using
TRAIN-OOF as the precision oracle systematically biases conservative.

Cousin to the T6-documented "TRAIN-OOF→test transfer-asymmetry" finding;
the natural-cal axis exhibits a particularly steep version of it because
the 7-component RF natural bank is overcomplete on a 6-feature DGP rule,
so OOF noise dominates while full-data fit is near-deterministic.

## Strict counter-flip subset analysis (informational)

The deep-dive's 32-row counter-flip set was characterized by `(4b=H, raw=H,
tier1b=H, bagged=M, bank_maj=M)`. Re-deriving on test gave 808 rows; the
deep-dive's 32 used the bank stability artifact differently (likely
`agreement >= some-threshold`). At the strictest reasonable joint filter
(score=6 ∩ bagged_pm>=0.95 ∩ bank-unanimous): **0 rows**. The deep-dive's
emitted 8-flip CSV `submission_4b_minus_natcal_strict.csv` remains the
narrowest LB-probable variant, but TRAIN OOF analog has only 2 rows there
(50% precision, uninformative).

## 42nd saturation roll-up

This brings the saturation count to 42 independent confirmations at LB
0.98150. The mechanism families now include (additions in this entry):
  - Orthogonal-bank k4-unanimous override without raw+tier1b confirmation
  - Per-seed individual-vote stability filter
  - TRAIN-OOF stratified precision-ranking of override flips

CLAUDE.md NEVER-GIVE-UP rule still applies; the next mechanism must be
structurally distinct from anything that uses the existing 9-OOF bank as
its precision oracle. Genuine novelty likely requires (a) external label
sources (DGP-faithful aux dataset, host NN inversion), (b) a precision
oracle that's not OOF-based (calibration via a held-out clean-label set
not in the 14-bank lineage), or (c) abandoning the 4b anchor entirely
and pursuing an architecture that doesn't depend on bank consensus.

## Files

- `scripts/idea4d_train_oof_precision.py` — 4d k4-unanimous 4-gate
- `scripts/cross_seed_stability_4b.py` — per-seed-stability filter
- `scripts/profile_4b_flip_precision.py` — stratified TRAIN-OOF profile
- `scripts/strict_counter_flip_check.py` — strictest subset analysis
- This audit doc

No submission CSVs emitted. No LB probe.
