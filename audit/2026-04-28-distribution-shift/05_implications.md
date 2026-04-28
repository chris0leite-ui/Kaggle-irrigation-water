# 05 — Strategic implications

What this body of evidence implies for the comp pipeline.

## A unified explanation of "original-as-anchor" failures

Every prior comp-log experiment that used the 10k original as a
**training source** failed at varying degrees. The distribution-shift
finding (AV AUC 0.697) explains why:

| experiment                                  | result                                  | shift role            |
|---------------------------------------------|-----------------------------------------|-----------------------|
| transfer-check (2026-04-20)                 | 0.96278 vs 0.97097 = 0.00819 gap        | gap = AV-shift effect |
| heavy original aug w=20 (2026-04-21)        | LB regress -0.00026 standalone          | rule-perfect orig pulls model AWAY from synth's flip pattern |
| NN-on-orig 5 archs (2026-04-22)             | all collapsed to rule ceiling 0.96097    | NN learns orig joint, fails on synth's drifted joint |
| TE-from-original (2026-04-22)               | argmax-equivalence theorem (rule ceiling) | TE statistic on orig == rule statistic, no marginal benefit |
| W7 NN-to-original k=1 (2026-04-26)          | min Euclidean distance 1.33, all NN-precision < rule | synth not anchored to orig rows |
| recipeonly soft-distill (2026-04-25)        | gap +0.00201, distill family closed      | teacher built on orig-shifted joint underperforms |

These are not 6 independent failures — they're **6 manifestations of
the single shift observation**. Once you know AV AUC orig↔synth = 0.697,
you should expect any inductive model trained on orig to underperform
on synth.

## Why the rule still works at 98.36% on synth

The rule was reverse-engineered from orig (where it's 100% accurate)
and the synth's NN preserved the rule **by construction** — class
priors didn't shift, the score=3 → score=1 redistribution kept the
rule's overall accuracy near-perfect, and the 1.64% flip budget is
small. **The rule survives the distribution shift because the rule is
a low-dimensional summary** that depends on threshold-crossings, and
the NN preserved the per-class threshold-crossing distribution.

## Why the rule SCORE distribution shifted but the rule ACCURACY didn't

Counter-intuitive at first: synth has score=3 −6.94 pp BUT rule
accuracy on synth (98.36%) is barely below orig (100%). The
resolution: score=3 is the **boundary band that produces the most
flips per row** (4.80% flip rate, 5,041 of 10,304 flips). The synth
removed score-3 mass and replaced it with score-1 mass (flip rate
0.0043%). Accuracy stays high because score-1 rows are easy.

This is also why dropping score-1 rows from training (DROP_SCORES,
2026-04-26) regressed: those rows are deterministic anchors that
calibrate the rest of the decision surface even though they
contribute zero gradient.

## Implications for current candidates

### Final-selection lock — UNCHANGED

The audit-F1 swap stands. The HEDGE candidate
(`submission_3way_recipe025_s1035_s7040.csv`, LB 0.98005) does NOT use
the original at all (it's a 3-way blend over recipe + pseudo_s1 +
pseudo_s7, all trained on synth-train). PRIMARY
(`submission_tier1b_greedy_meta.csv`, LB 0.98094) uses the original
only as `ORIG_mean / ORIG_std` per-cat aggregates and as the source
of `logit_P_*` LR-formula features — both rule-aligned summaries that
ARE consistent with the orig distribution AND with the rule's behavior
on synth.

### Anti-recommendation: any "use orig as training data" lever is dead

Because the AV gap is 0.697 (huge), any inductive model trained on
orig will systematically misfit synth. This includes:

- "Train recipe on orig, predict on synth" (transfer-check pattern).
- "Pretrain on orig, fine-tune on synth" (already nulled at 2026-04-22).
- "Augment training with k× orig copies" (already nulled at w=20 in
  2026-04-21).
- "Use orig as labeler in stage-N pseudo-label" (would inherit the
  shift).

Re-classify all of these as STRUCTURALLY DEAD, not just empirically
nulled.

### Recommendation: orig is a SUMMARY-statistic source, not a training source

The recipe's `ORIG_mean_<cat>_<num>` and `ORIG_std_<cat>_<num>`
features (38 cols) and the LR-formula logits (3 cols, fitted on orig)
ARE consistent with the rule on synth because they're aggregations
that survive the joint-feature shift. These are the right abstraction
level. **Don't add new features that consume per-row orig values.**
Do consider adding:

- More group-by aggregations on orig if they're rule-aligned
  (e.g., `ORIG_quantile_<num>_p25/p75`, untested as far as the
  comp log shows).
- LR formulas trained per-class on orig with different feature
  subsets (we have the pooled logit; per-class breakdown might add
  marginal info).

This is a SPECULATIVE next-step, not a recommendation. Bayesian
prior of clearing the +2e-4 gate: low (≤15%) given the
saturation evidence — but it's the only lever in this neighborhood
that is structurally consistent with the shift findings.

## Implications for the hypothesis board

Add three new portable rules to LEARNINGS.md:

1. **AV AUC orig ↔ synth-train 0.697 is the joint-shift size on this
   problem.** Any future synthetic-tabular comp where the host
   discloses "train+test generated by NN trained on public anchor"
   should run a 5-fold AV between source and anchor BEFORE building
   anchor-based features. If AUC > 0.55, anchor-as-training-source
   levers are dead; treat anchor as summary-statistic source only.

2. **Class-conditional shift is the diagnostic that distinguishes
   "feature drift" from "label noise."** Class priors identical
   (Δpp ≤ 0.1) + per-class feature shift (d ≥ 0.20 on at least one
   axis) means the host NN preserved the labeling rule but drifted
   the features. This is the most common pattern in synthetic-
   tabular Playgrounds.

3. **Per-cell flip rates range over orders of magnitude on a
   rule-cube tabular problem.** On this dataset, cells span 0.000%
   (cell 0, n=33k) to 70.45% (cell 51, n=308). Locate the high-flip
   cells via the 6-bit `(dry, norain, hot, windy, nomulch, kc)`
   factorization; deterministic-cell drop is structurally feasible
   but per-flip-cell override is information-bounded by the
   non-rule feature axes (which globally show no signal — the
   intra-cell signal is per-score-conditional and washes out across
   the marginal).

## Note on this report

This report is **diagnostic, not actionable**. It does not propose an
LB probe. It explains the body of "original-as-anchor" failures,
documents the magnitude of the shift for future synthetic-tabular
comps, and crystallizes the saturation argument (it's not just
"we tested 30 levers and they nulled" — it's "the test/orig
distribution diverges by AUC 0.697, and the HEDGE submission
sidesteps that completely while the PRIMARY uses orig only as
summary-statistic source").

**LB-best primary unchanged at LB 0.98094.** Final-selection lock
unchanged. LB budget unchanged. Two days to deadline 2026-04-30.
