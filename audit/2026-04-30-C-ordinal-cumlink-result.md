# 2026-04-30 — C: ordinal cumulative-link XGB result (44th saturation, no LB probe)

## Mechanism

All 14 bank components use 3-way multinomial softmax. Given the DGP rule's
strict ordering (L < M < H via score thresholds), the labels are ordinal —
and ordinal cumulative-link reformulates as two binary heads:

```
clf1: P(y > L) = P(y in {M, H})    binary, prevalence 41.3%
clf2: P(y > M) = P(y == H)         binary, prevalence  3.3%
```

Reconstruct: `P(L) = 1 - P(>L)`; `P(M) = P(>L) - P(>M)`; `P(H) = P(>M)`.
Monotonicity enforced via `P(>M) <= P(>L)` clip, then floor + renormalize.

Implementation: `scripts/C_ordinal_xgb.py`. 5-fold StratifiedKFold(seed=42)
for v1 alignment, per-fold OTE on recipe FE (556 features after engineering),
XGBoost hist tree-method, 3000 boost rounds with early-stopping(200),
balanced sample weights.

## Production results

```
Fold 1   499s  bi1=910   bi2=1375  raw_argmax_bal=0.95122
Fold 2   441s  bi1=823   bi2=1021  raw_argmax_bal=0.94751
Fold 3   473s  bi1=521   bi2=1693  raw_argmax_bal=0.95091
Fold 4   377s  bi1=686   bi2=909   raw_argmax_bal=0.94875
Fold 5   450s  bi1=593   bi2=1441  raw_argmax_bal=0.94836
Total                              FULL_OOF_raw_argmax = 0.949349
```

## Audit (`scripts/C_ordinal_audit.py`)

**G1 — standalone tuned macro**:
- C tuned: **0.973328** (bias [-0.667, -1.0, 3.0])
- recipe_full_te tuned: 0.979649
- **Δ = −0.006320** (60 bp below recipe FE multinomial baseline)

**G2 — Jaccard (argmax) vs 14 bank components**:
- 13 of 14: Jaccard ≥ 0.9917 (high overlap → redundant)
- xgb_nonrule: **0.6533** (low overlap — but xgb_nonrule deliberately drops
  DGP rule features, so the orthogonality is just rule-feature presence,
  not a new geometry)

**G4 — minimal-input 2-comp meta vs v1 alone**:
```
v1 alone tuned:                    0.980646
log_blend(v1, C; α=0.05) tuned:    0.980639  (Δ = -0.000007)
log_blend(v1, C; α=0.10) tuned:    0.980654  (Δ = +0.000007)  ← best
log_blend(v1, C; α=0.15) tuned:    0.980477  (Δ = -0.000170)
log_blend(v1, C; α=0.20) tuned:    0.980342  (Δ = -0.000305)
log_blend(v1, C; α=0.30) tuned:    0.980078  (Δ = -0.000569)
log_blend(v1, C; α=0.40) tuned:    0.979599  (Δ = -0.001048)
```

Best 2-comp meta lift = +0.000007 (well below fold-noise).

## Diagnosis

C falls cleanly into the saturated "tree learners on recipe FE" Pareto
frontier, despite the structurally distinct ordinal decoder:

1. **Standalone weakness**: 0.973 tuned vs 0.980 baseline — splitting the
   3-way problem into two binaries throws away cross-class joint
   information that multinomial softmax preserves. The two binary heads
   over-rely on the (y == H) head being good, but with H prevalence 3.3%
   the head is noisy even at 1693 boosts.

2. **High Jaccard everywhere except xgb_nonrule**: when C disagrees with
   v1, raw, recipe_full_te, etc., it's mostly on the same hard rows those
   models also struggle with — the same Pareto frontier. The 0.6533
   Jaccard with xgb_nonrule is an artifact of xgb_nonrule's intentional
   feature ablation (drops rule-features), not C providing new geometry.

3. **Sub-noise meta lift**: +7e-6 in best α=0.10 log-blend is fold-variance,
   not signal. Higher α regresses, confirming C dilutes v1's strength.

## Connection to prior saturations

This adds to the "tree variants on recipe FE" cluster of saturations:
- recipe_full_te (LGBM multinomial) — bank component
- recipe_full_te_catboost / catboost_natural — bank components
- recipe_lgbm_native — earlier null
- xgb_nonrule (rule-features dropped) — bank component
- xgb_metastack — bank component
- C ordinal cumulative-link — **THIS, null**

The ordinal decoder is a real mechanism choice, but it doesn't escape the
recipe-FE Pareto frontier any more than the prior multinomial variants did.

## Saturation count: 44

## LB budget

No LB probe spent. Production wall: 2266s = 37.8 min CPU on 5 folds × 2
binary heads.

## What the result tells us about remaining options

- The "tree learners on recipe FE" mechanism class is now closed against
  6+ variants. Decoder reformulations alone don't unlock new signal on
  this feature set.
- Genuinely-new bank components would need to either (a) use a different
  feature representation (e.g., raw + deep hashing, or non-recipe FE), or
  (b) use a non-tree base learner that captures rank structure differently
  (e.g., listwise rank XGB or ordinal NN). Both have been tested in nearby
  forms (path_a_recipe_mlp_local, NN saturations) and saturated.
- This pushes the remaining-untried list further toward training-data
  cleanliness (cleanlab+retrain) or external supervision (sonnet LLM).
