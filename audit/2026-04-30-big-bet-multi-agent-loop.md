# 2026-04-30 — Big-bet multi-agent ideation loop

After the 17-row C7-LM candidate was deemed too tiny ("think bigger"),
re-ran multi-agent ideation with a +0.0003 EV floor and 1-3h CPU budget
per idea. Three big-bet experiments executed. Three new portable rules
discovered. No LB-actionable candidate emerged but the structural ceiling
is now better characterized.

## Pipeline

- 2 parallel ideation subagents (model-architecture lens + training-data lens)
- 14 ideas total, all aiming at +0.0003+ delta
- Self-critique converged on 3 actionable: A7 (adversarial validation), B3
  (flip-specialist on rule-violators), A1 (10k-only host-NN-inversion)

## Result 1 — A7 adversarial validation: AUC = 0.5009

Trained binary classifier with target = `1[is_test]` on 5-fold CV.
Mean AUC across folds = 0.5009 (std 0.0014). Train and test are
**essentially i.i.d.** — the host generator produced both from the same
distribution.

**Portable rule (LEARNINGS.md)**: in synthetic competitions where the
host generates both train and test from the same NN-labeling pipeline,
distribution-shift mechanisms (test-similarity sample weighting,
adversarial-validation feature engineering, "import-test-distribution"
training tricks) have NO Bayesian floor and should be skipped. AUC ≈ 0.5
should be the gating sanity check before any such mechanism is built.

This closes:
  - B1 (10k mix-as-training-data, distribution-shift framing)
  - A7 (adversarial-validation reweighting)
  - Any test-similarity-based sample weighting

## Result 2 — B3 flip-specialist: structurally bounded by Stage-1 AUC

Trained 3-class LGBM on 10,304 rule-violator rows (TRAIN where rule != label).
**Phase 1 OOF on violators**:
```
Overall accuracy: 99.14%
Per-direction recall: L->M 100%, H->M 99.8%, M->H 98.0%, M->L 97.0%
At conf >= 0.90: 99.76% accuracy on 9983 rows
```

Phenomenal performance — but only **conditional on knowing the row is a
violator**. Phase 2 applied specialist to non-violators on TRAIN: 99% FP
rate at any confidence threshold (specialist != rule on non-violators with
near-100% confidence, and 0% of those flagged are actually correct flips).

Phase 3+4 added a 2-stage gate: P(violator | features) classifier with
TRAIN OOF AUC 0.8833. Best-case combined gate at τ_v=0.30 produces 422
TRAIN flips with macro_delta -0.006103 (regression). At τ_v=0.50, 92
flips with delta -0.001. **No (τ_v, τ_s) configuration passes 4-gate**.

Why: AUC 0.88 caps PPV(violator | flag) at ~50% even at the highest
P(violator) percentile. Need PPV > 92% to clear H→M break-even.

**Portable rule (LEARNINGS.md)**: rule-residual flip-detection on synthetic
NN-labeled data is structurally bounded by the binary "is row a violator?"
AUC. With 14 raw features and a 1.6% positive base rate, AUC plateaus
near 0.88-0.91 (consistent with the prior W13 result). PPV at top
percentiles caps at ~50%, well below the 92% needed for H→M overrides.
No flip-detection mechanism using only raw features can clear precision
break-even on the H→M direction for synthetic-NN-flip targets.

This closes:
  - B3 flip-specialist family
  - C2 two-stage rule-then-fixup classifier
  - A3 NN-flip-risk classifier
  - W13 wrongness-predictor extensions
  - Any binary "is row wrong" → 3-class flip-direction composition

## Result 3 — A1 10k-only host-NN-inversion: 10k = rule

Trained LGBM on 10,000 original irrigation_prediction.csv rows, 5-fold CV.
**OOF accuracy: 99.75%. Rule accuracy on 10k: 100.0000%.**

Surprise finding: the 10k original dataset's labels follow the DGP rule
**perfectly** (no flips). On synthetic train, the 10k-trained model:
  - Accuracy on synthetic labels: 98.25% (lower than rule's 98.36%)
  - Agrees with rule: 99.86%
  - On synthetic violators: agrees with rule 99.29%, predicts true (flipped)
    class only 0.7%

**Portable rule (LEARNINGS.md)**: when the original dataset is rule-
consistent and the synthetic dataset has NN-introduced flips, models
trained on the original cannot recover the synthetic flip pattern — they
only re-learn the rule. "Train-on-original to invert host NN" is
structurally null in this regime. The rule is already implicitly captured
by every existing bank component, so adding a 10k-trained model adds no
orthogonal signal.

This closes:
  - A1 10k-only host-NN-inversion
  - A5 cost-sensitive 10k-anchor retraining (10k anchor = rule, redundant)
  - B6 co-training synthetic + 10k (10k contributes only rule signal)
  - Any "leverage 10k clean labels for flip-pattern recovery" mechanism

## Saturation count: 45

Three new mechanisms closed in this session, with associated portable
rules. The structural ceiling at LB 0.98150 remains.

## Files

- `scripts/A7_adv_validation.py` — 5-min train-vs-test sanity check
- `scripts/B3_flip_specialist.py` — phase 1 (specialist on violators only)
- `scripts/B3_phase2_test_application.py` — phase 2 (apply to test, FP rate finding)
- `scripts/B3_phase3_two_stage.py` — phase 3 (2-stage with broken stage 1)
- `scripts/B3_phase4_recalibrated.py` — phase 4 (calibrated 2-stage, all gates fail)
- `scripts/A1_10k_inversion.py` — 10k-only training, 10k = rule finding
- `submissions/submission_B3_flip_specialist_t85.csv` — naive specialist apply (DO NOT LB-PROBE)

## Recommendations forward

The remaining genuine-novelty options either need external resources we
don't have (host NN architecture, additional clean labels beyond 10k) or
are speculative bets at < 5% prior given the 45-saturation closure list:

  - B7 macro-BACC differentiable surrogate as literal training loss
  - A3 multi-task auxiliary heads (label + violator-flag + DGP-score)
  - Bootstrap 50+ stratified-resample seed bag

None has a strong Bayesian argument; each is incremental at best. The
honest read is that the team's local pipeline has reached its measurable
ceiling at LB 0.98150. The +0.00069 gap to leader Cdeotte is most plausibly
captured by something not in our toolkit (different model architecture,
additional data, or extensive hyperparameter search at a scale we haven't
attempted).
