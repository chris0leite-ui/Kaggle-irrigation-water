# 2026-04-30 — B: DGP-rule-anchored override audit (42nd saturation, no LB probe)

## Mechanism

Use the closed-form DGP rule (~98.4% global accuracy) as the override
**authority** for rows where 4b is uncertain. Distinct from:
- Bank-majority overrides (4b/Idea-4b) — authority is 14 ML committee
- T1 LLM-judge — authority is haiku
- T6 directional compose — authority is Caruana-greedy ensemble

Fire condition variants tested (TRAIN OOF, 4b OOF analog):
1. `rule != 4b` × score-band (low/mid/high extreme)
2. `rule != 4b` × score-band × `bank_argmax == rule`
3. (1) + (2) + `bank_max < tau` for tau in {0.40 ... 0.80}

## Result — closed on TRAIN OOF, no LB probe spent

**Per-band rule-vs-4b precision** (rule==y | rule != 4b, on TRAIN OOF):

```
[low_safe   s<=2]  n=     3   prec=0.000     bank-confirms n=    1  prec=0.000
[low_band   s==3]  n=  1282   prec=0.368     bank-confirms n=  521  prec=0.528
[mid_band   4..6]  n=  3953   prec=0.620     bank-confirms n= 2648  prec=0.806
[high_band  s==7]  n=   823   prec=0.005     bank-confirms n=    0  prec=  -
[high_safe  s>=8]  n=   161   prec=0.012     bank-confirms n=    0  prec=  -
```

Apply the T6-documented 15-20pp TRAIN-OOF→test asymmetry haircut
(v1 IN bank inflates bank-confirms precision):

- mid_band bank-confirms: 80.6% TRAIN → ~62-65% projected test (well below 92% break-even)
- low_band bank-confirms: 52.8% TRAIN → ~37-42% projected test (deep regression territory)
- high-band, high-safe: rule precision is **catastrophic** (~1%) — these are exactly the rows where the host's NN flips score=7,8 → M, so the rule's H prediction is wrong and 4b correctly disagrees

**Diagnostic insight**: the 14-bank-majority has already absorbed the host
NN-flip pattern at score-extreme bands (s≤1, s≥9: zero bank-confirms
disagreement after the rule). The rule cannot bring marginal authority on
rows where it agrees with the bank, and is **wrong** on rows where it disagrees.

## Connection to score-band rule accuracy

Rule accuracy per score (TRAIN, including all rows not just disagreement):
```
score   n      rule_acc
  0   33767   1.000000
  1  115457   0.999965
  2  122220   0.997005
  3  102157   0.952043   ← border, flips ~5%
  4  117837   0.987105
  5   79203   0.996540
  6   38416   0.959683   ← border, flips ~4%
  7   15026   0.909486   ← below 92% break-even
  8    2680   0.876866   ← below break-even
  9    3237   0.999382
```

The host NN concentrates flips at **boundary scores** (3, 6, 7, 8). On those
rows, the rule is below break-even and the bank/4b correctly disagree.

## B2 follow-up — aux-flip-detector-routed override (43rd saturation)

To rule out the "score-band heuristic was too crude" hypothesis, repeated
the audit using the LEARNED `aux_flipped_from_rule` predictor (precomputed
binary XGB head from `multitask_aux_xgbs.py`) as the flip-likelihood gate
instead of score-band. Distinct from saturated #3 multitask-aux meta-stacker
(which used aux_flip as a stacker FEATURE); B2 uses aux_flip as a hard
ROUTING GATE.

```
aux<0.05  rule!=4b  n=  10  prec=0.900  oof-delta=-0.000003
aux<0.10                n=  37  prec=0.865  oof-delta=-0.000035
aux<0.20                n= 115  prec=0.817  oof-delta=-0.000202
aux<0.30                n= 203  prec=0.773  oof-delta=-0.000439
```

After T6 ~15-20pp TRAIN→test haircut, even the tightest aux<0.05
projects ~70-75% precision (below 92% break-even). All ORIGINAL directions
are H→M; secondary directions (L→M, M→L) appear only at higher tau but
with sub-30% precision.

**Aux head is fundamentally weak** at separating flips from noise:
top-1% aux_flip rows are actually flipped only 37.4% of the time
(versus 1.6% base rate — 23x lift, but absolute precision of 37% means
the gate isn't sharp enough for override authority).

Conclusion: the entire "rule-as-authority for selective override on 4b"
mechanism class is closed under both heuristic gates (score-band, B) and
learned gates (aux_flipped, B2).

## Saturation count: 43

This closes one of the three "untried mechanism categories" listed on the
hypothesis board ("composition of LB-validated submissions via new override
axes"). Specifically: rule-as-authority is closed alongside
LLM-as-authority (T1, 41st saturation) and Caruana-as-authority (T6
directional, 40th saturation).

## What this does NOT close

- **External NN inversion** — host NN architecture unknown; would require
  reproducing the labeling NN to identify systematic flips. Speculative.
- **DGP rule + cleanlab on TRAIN** — could identify the 1.6% flipped rows
  in TRAIN, retrain v1 on cleaned labels. Distinct from this audit (which
  uses the rule for test-side override only, not for training-data
  cleaning).
- **Rule-aware test-time augmentation** — perturb continuous features by
  ±epsilon around rule boundaries (e.g., Soil_Moisture around 25), predict
  K perturbed copies, average. Different from boundary-confined TTA
  (which used model-confidence boundary, not rule-bit boundary).

## LB budget

No LB probe spent on this audit. 8 LB submissions remaining today
(2 already used: B → 0.98140, TC1 → 0.98150).

## Cost

~2 min CPU on 630k TRAIN rows. Cheaper than scaffolding alone for any
LB-validated mechanism family.
