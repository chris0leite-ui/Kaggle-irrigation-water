# 07 — Execution results: Options A, B', C (FINAL)

All three drift-leverage options executed 2026-04-28. Three independent
NULLs. Option C closed cleanly via diagnostic; Options A and B'
completed full 5-fold production, then 4-gate analysis vs three
anchors.

## Calibration of the AV signal

`av_predicts_flip_full.py` on the full 630k train (10,304 flips):

```
AUC(P(orig), flip) = 0.5746
Cohen's d (P(orig), flip vs clean) = +0.335
Top-K=5000 precision = 8.12% (5x base rate, RIGHT AT M↔H break-even)
```

Per-score AUC (the ONE diagnostic that drove which options would work):

| score | n      | n_flip | AUC    |
|-------|--------|--------|--------|
| 1     | 115457 | 5      | 0.4955 |
| 2     | 122220 | 365    | 0.5093 |
| **3** | 102157 | **4899** | **0.5227** ← dominant flip cell, no signal |
| 4     | 117837 | 1520   | 0.5806 |
| **5** | 79203  | 274    | **0.7169** ← strong |
| 6     | 38416  | 1549   | 0.6103 |
| **7** | 15026  | 1360   | **0.6258** |
| 8     | 2680   | 330    | 0.168  ← inverted (small n) |

The signal lives at scores 5/6/7. Score=3 (47.5% of all flips) is
essentially random — predicted Option C's null.

## Option C — score=3 specialist with AV input: NULL

```
domain    = score=3 ∩ teacher_argmax=Low (n=101,392)
target    = (y == Medium); prevalence 4.28%
features  = 37 (raw + cats + dist + AV-score + teacher meta)
5-fold OOF AUC = 0.8195   (prior spec_lm_v3 without AV: 0.827 — TIED)
```

Top-K=100 precision peaks at 43.0%, Wilson 90% lower CI = 0.368,
break-even floor 0.393. **No conformal-feasible operating point.**
AV-score did not lift the score=3 specialist as predicted (per-score
AUC at score=3 = 0.52 = noise).

## Option A — AV-score as recipe FE: NULL (recipe-redundant)

```
fold 1  argmax 0.97545  vs recipe 0.97544  Δ +0.00001
fold 2  argmax 0.97578  vs recipe 0.97659  Δ -0.00081
fold 3  argmax 0.97707  vs recipe 0.97721  Δ -0.00014
fold 4  argmax 0.97443  vs recipe 0.97465  Δ -0.00022
fold 5  argmax 0.97512  vs recipe 0.97557  Δ -0.00045
OOF argmax    0.97557   recipe 0.97589   Δ -0.00032
Tuned OOF     0.97953   recipe 0.97967   Δ -0.00014   (within fold noise)
Bias [0.9324, 1.1689, 3.4008]   (Low/Med biases dropped ~0.5/0.3)
```

4-gate vs three anchors:

| anchor              | peak α | peak Δ      | direction   | gates    |
|---------------------|--------|-------------|-------------|----------|
| recipe_full_te      | 0.150  | +0.00006    | REMOVE-High | G1✗ G2✓ G3✓ G4✗ |
| lb_best_3stack      | 0.000  | +0.00000    | n/a (peak=0)| n/a      |
| lb_best_4stack      | 0.000  | +0.00000    | n/a (peak=0)| n/a      |

Jaccards 0.82–0.87 vs anchors (≥ 0.80 redundancy zone).
**G1 fails on all three anchors.** Mechanism: AV-score is a
1-d learned summary of `norain` + `Rainfall_mm` + `dry` + `windy`
— the same features that dominate recipe XGB's depth-4 splits.
Recipe natively recovers the AV signal via its own splits.

## Option B' — orig-weight sample multiplier: NULL (right direction, magnitude bounded)

```
fold 1  argmax 0.97523  vs recipe 0.97544  Δ -0.00021
fold 2  argmax 0.97611  vs recipe 0.97659  Δ -0.00048
fold 3  argmax 0.97749  vs recipe 0.97721  Δ +0.00028  ← positive
fold 4  argmax 0.97505  vs recipe 0.97465  Δ +0.00040  ← positive
fold 5  argmax 0.97533  vs recipe 0.97557  Δ -0.00024
OOF argmax    0.97584   recipe 0.97589   Δ -0.00005   (essentially tied!)
Tuned OOF     0.97955   recipe 0.97967   Δ -0.00012   (within fold noise)
Bias [1.1324, 1.2689, 3.4008]   (closer to recipe than A's was)
```

4-gate vs three anchors:

| anchor              | peak α | peak Δ      | direction | gates           |
|---------------------|--------|-------------|-----------|-----------------|
| recipe_full_te      | 0.075  | +0.00001    | ADD-High  | G1✗ G2✓ G3✓ G4✗ |
| lb_best_3stack      | 0.000  | +0.00000    | n/a       | n/a             |
| **lb_best_4stack**  | **0.025**  | **+0.00001** | **ADD-High** | G1✗ G2✓ G3✓ G4✗ |

**B' produces ADD-High direction at every α on every anchor** — the
correct macro-recall direction, opposite of A's mostly-REMOVE-High
trade vs recipe. But the magnitude tops out at +0.00001 OOF on
LB-best 4-stack — far below the +2e-4 emit gate.

## Summary table

| option       | mechanism                              | standalone Δ | 4-gate result | mechanism status            |
|--------------|----------------------------------------|--------------|---------------|-----------------------------|
| A (avp)      | AV-score as 1-dim recipe FE feature    | tied         | NULL G1×3     | recipe-redundant            |
| B' (origw10) | sw *= (1 + (1−P(synth))) per row       | tied         | NULL G1+G4    | right direction, too small  |
| C (spec3)    | score=3 specialist + AV-score input    | tied (vs spec_lm_v3) | NULL Wilson | predicted (AUC 0.52 at score=3) |

## Combined mechanism diagnosis

The AV classifier learned a 0.697 OOF AUC distinction orig↔synth that
concentrates on the Rainfall threshold, Soil_Moisture threshold, and
decimal-fraction features. **All of these are already first-order
features in the recipe XGB.** Three independent delivery mechanisms
(direct FE, sample weighting, sub-domain specialist input) cannot
extract marginal signal beyond what the recipe's tree splits already
encode.

The diagnostic AUC (0.5746) was modest enough that this outcome was
predictable; the mild positive direction in B' confirms there's a
faint signal, but the magnitude is bounded by feature redundancy.

## Three new portable rules (LEARNINGS.md candidates)

1. **AV-AUC as a feature is recipe-redundant when the AV classifier's
   top gain features overlap heavily with the recipe's top splits.**
   The 1-dim AV summary captures a non-linear COMBINATION of
   features the recipe trees already split on; depth=4 + reg_alpha=5
   + reg_lambda=5 makes the recipe a near-optimal recoverer of the
   same signal natively.

2. **Per-row sample-weight reweighting (here: by 1−P(synth)) produces
   the right macro-recall direction (ADD-High) on a saturated stack
   when the weighting signal correlates with rare-class flip
   incidence.** The trade is structurally clean — but the magnitude
   is bounded by `(diagnostic AUC − 0.5) × class_weight_amplification`,
   which on this problem caps at ~+0.00001 OOF onto LB-best 4-stack
   at any β value.

3. **Score-conditional AUC of a global flip detector predicts which
   sub-domain specialists can use it.** When per-score AUC at the
   target score is < 0.55, do NOT scaffold a conformal/precision-
   based override at that score — it will fail break-even by
   construction.

## Strategic implication

Three independent NULLs across structurally distinct mechanisms
**confirm the LB 0.98094 ceiling holds against the AV-shift signature
lever family.** Combined with the 2026-04-26 NN-on-orig + soft-distill
+ W7 + N5b family closures, the entire "use 10k original as anchor"
mechanism category is now exhaustively closed.

This is the **31st-33rd structural saturation confirmations** at LB
0.98094.

**Final-selection lock unchanged.** Two days to deadline 2026-04-30.
LB best `submission_tier1b_greedy_meta.csv` at 0.98094, hedge
`submission_3way_recipe025_s1035_s7040.csv` at 0.98005.
