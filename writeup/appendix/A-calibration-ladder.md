# Appendix A — Calibration ladder

The full OOF→LB transfer table for irrigation-water. Used as the
running "is OOF still calibrated to LB?" check.

## Reading the table

- **OOF**: 5-fold StratifiedKFold balanced accuracy, seed 42.
- **LB**: public-LB score from Kaggle CLI.
- **Gap**: LB − OOF. Negative gap means LB above OOF (mechanism-
  specific, see notes).
- A typical fold-std on this comp was ~0.002. Gaps inside ±0.0010
  are consistent with calibration; larger gaps need investigation.

## Full ladder

| Mechanism | OOF | LB | Gap | Phase |
|---|---:|---:|---:|---|
| baseline LGBM tuned | 0.97097 | 0.96972 | −0.00125 | Kickoff |
| LGBM+EXT (concat 10k orig) | 0.97124 | n/a | n/a | Day 1-2 |
| LGBM+DGP tuned | 0.97271 | 0.97271 | 0.00000 | Day 2 |
| 3-way log-blend (greedy) | 0.97375 | 0.97296 | −0.00079 | Day 3 |
| greedy+XGB(non-rule) α=0.15 | 0.97421 | 0.97352 | −0.00069 | Day 3 |
| recipe_full_te | 0.97967 | 0.97939 | −0.00028 | Day 4-7 |
| recipe × pseudo 2-way | 0.98012 | 0.97998 | −0.00014 | Day 8 |
| 3-way multi-seed | 0.98029 | 0.98005 | −0.00024 | Day 8 |
| LB-best 3-stack | 0.98061 | 0.98008 | −0.00053 | Day 9 |
| LB-best 4-stack (tier1b_greedy_meta) | 0.98084 | 0.98094 | +0.00010 | Day 10 |
| Option 1 meta-stacker | n/a | 0.97986 | n/a | Day 11 (LEAKAGE) |
| Soft-distillation student | n/a | regress | −0.00148 | Day 13 (LEAKAGE) |
| LR meta v1 | n/a | regress | −0.00103 | Day 14 (LEAKAGE) |
| LR meta v4 ET+kNN | n/a | regress | −0.00102 | Day 14 (LEAKAGE) |
| P3 perturbed | n/a | regress | −0.00139 | Day 14 (LEAKAGE) |
| R2 hybrid grid-selected | n/a | 0.98048 | −0.00046 | Day 15 (G-S BIAS) |
| mlp_metastack a=0.30 | n/a | 0.98073 | −0.00021 | Day 15 |
| v1 RF natural standalone | 0.98063 | 0.98129 | **+0.00066** | Day 16 |
| rawashishsin v3 standalone | 0.98016 | 0.98109 | **+0.00093** | Day 16 |
| 2-OTHER raw+tier1b k=2 unanimous (B) | 0.98088 | 0.98140 | **+0.00052** | Day 17 |
| Idea 4b triple-consensus (PRIMARY) | ~0.98088 | **0.98150** | **+0.00062** | Day 17 |
| Idea 5 anchor-switch | n/a | 0.98148 | −0.00002 | Day 17 |
| 4b minus 176 L→M flips | n/a | 0.98148 | −0.00002 | Day 17 |
| 4b + W5(M→H) + strict90 | n/a | 0.98143 | −0.00007 | Day 17 |
| T6 directional compose | n/a | 0.98121 | −0.00029 | Day 18 (40th sat) |
| L5c K=100 union | n/a | regress | −0.00145 | Day 18 (47th sat) |
| bagginglr_natural standalone | n/a | 0.98106 | −0.00044 | Day 18 (48th sat) |

## Mechanism-family OOF→LB gaps

| Family | Gap range | Interpretation |
|---|---|---|
| Baseline LGBM | −0.0013 | Slightly OOF-optimistic; one fold-std. Calibrated. |
| Recipe ladder | −0.0008 to −0.0001 | Calibrated. 5-fold OOF trustable to ~1bp. |
| Stacking (well-gated) | −0.0005 to +0.0001 | Tighter; minimal-input meta gate works. |
| Stacking (leaky / grid-selected) | −0.0010 to −0.0014 | OOF-inflated. The 7 leakage incidents live here. |
| Override family | +0.0005 to +0.0009 | LB above OOF. **Not a margin to spend on more stacks.** It's a structural function of which 145 candidate rows the override fires on. |
| Standalone alternative-cal banks (RF natural, rawashishsin) | +0.0007 to +0.0009 | Same structural-positive gap as override — rules-of-thumb decision boundaries differ from OOF folds. |

## Key reads

1. **OOF→LB calibration was trustable for the recipe ladder** to
   within 1bp. We could and did select on OOF in this regime.
2. **Stacking inflated OOF by 5-15bp** when the bank was saturated.
   The 4-gate filter brought this back into spec.
3. **Override mechanisms had a structurally positive gap**. The +6bp
   on Idea 4b is *not* a margin we could keep extracting by adding
   more flips — it's a function of which rows the override targeted.
   Ablation tests on Day 17 (4b minus 176 L→M flips, etc.) confirmed:
   the gap shrinks toward zero as you alter the row set.
4. **Saturation kicked in around 0.98140** and held through 8
   final-day mechanism variants. The 0.98150 → 0.98148 thrashing
   is below the 0.00005 floor probe resolution of the 80/20 split.
