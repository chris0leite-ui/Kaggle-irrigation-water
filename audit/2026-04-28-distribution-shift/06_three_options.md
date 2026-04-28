# Three options to leverage the drift information

After the 2026-04-28 distribution-shift report (`README.md` + 5 task
files), three structurally distinct ways to USE the drift information
remain plausible. All three pass the "structurally novel" filter:
none has been tested in the comp log, and each attacks a different
failure mode of prior orig-anchor levers.

The diagnostic AUC of `1 − P(synth)` on flip prediction is **0.585**
(Cohen's d = −0.349, n = 9823 clean / 177 flip), gating Option A's
prior. See `scripts/dist_shift/av_predicts_flip.py`.

## Option A — AV-score as a 1-dim flip-detection feature for recipe XGB

### Mechanism
Train AV classifier on orig (10k) vs synth-train-subsample (10k) using
target-free features. Predict P(synth | row) on the FULL 630k train
and 270k test. Add `1 − P(synth)` (i.e., P(orig | row)) as a single
new numeric column to the recipe FE matrix. Retrain recipe XGB.

### Why now
- Diagnostic AUC 0.585 (n=10k, n_flip=177) is qualitatively stronger
  than N5b's GMM/IsoForest/kNN density estimators (all ~0.51). The
  AV classifier captures **discriminative joint-shift signature**
  (Rainfall-threshold-crossings, digit fingerprints) that
  density-of-orig-features cannot.
- Mechanism distinct from every prior orig-anchor lever:
  - W7 used k=1 NN distance to orig (~0.51).
  - N5b OOD scores (3 features) used unsupervised density (~0.51).
  - TE-from-orig used per-key target stats (argmax-equivalent to
    rule).
  - **None used a discriminative classifier's per-row shift score.**

### Concrete plan (~80 min CPU total)
1. ~10 min: extend `scripts/dist_shift/av.py` to predict on the FULL
   train + test (currently predicts only on the 20k balanced subset).
   Save `oof_av_p_synth_train.npy` (630k,) and `test_av_p_synth.npy`
   (270k,).
2. ~50 min: parameterize `scripts/recipe_full_te.py` with an
   `EXTRA_AV_PSYNTH=1` env var that loads `1 − P(synth)` and adds it
   as a numeric column. Run 5-fold StratifiedKFold(seed=42) for OOF
   alignment with the saved bank.
3. ~5 min: 4-gate analysis (G1 ≥ +2e-4 OOF, G2 errs ≤ anchor +5%, G3
   PCR each class ≥ anchor − 5e-4, G4 |net_rare_class_flip| / churn
   ≥ 0.5 with ADD-direction).
4. If gates pass: emit submission CSV; ASK user before LB probe.

### Gate-likely outcome
This is structurally a "wide programmatic FE +1 dim" experiment —
same family as the 2026-04-27 wide_fe NULL (1331 candidate features
added → 2.2% pickup rate, blend null at every α). The recipe's 350+
OTE features may already encode the AV signal indirectly via
per-cat-pair conditional probabilities.

### Bayesian prior
**~20% LB lift** (vs the pre-2026-04-25 ~30% prior; downgraded by
the wide_fe FE-redundancy finding). Cost is 80 min and a possible LB
slot. Diagnostic value is high either way: confirms or refutes the
"AV-shift signature is a real flip-prediction feature" hypothesis.

### Side benefit
The `oof_av_p_synth_train.npy` artifact can also serve as input to
options B and C; it's a strict prerequisite anyway.

---

## Option B — Density-ratio importance weighting for orig-augmented training

### Mechanism
For each orig row i, compute `w_i = P(synth | row_i) / P(orig | row_i)
= AV_p / (1 − AV_p)`. Concat orig (10k rows, weighted by `w_i`) with
synth-train (630k rows, weight=1) and retrain recipe XGB. The
weighted concat shifts orig's effective distribution toward synth's,
correcting the joint-shift bias.

### Why now
- 2026-04-21 heavy-aug at uniform `w=20` LB-regressed (-0.00026
  standalone). 2026-04-25 utaazu's `ORIG_ROW_WEIGHT=0.35` also
  AV-driven was flagged "skip on principled grounds" because the
  AV signal hadn't been computed.
- **Now we have it.** The principled per-row weighting was missing.
- Mechanism: an orig row that "looks synth-y" (high P(synth))
  contributes near-1× weight; an orig row that looks orig-y (low
  P(synth)) contributes near-0×. Effectively SUBSAMPLES orig to its
  synth-distribution overlap — exactly the rows that ARE
  representative of the synth's joint feature pattern.

### Concrete plan (~50 min CPU)
1. Reuse `oof_av_p_synth_train.npy` from Option A's preflight (or
   rebuild from scratch).
2. Compute `w_i = clip(p / (1 − p), 0, 10)` per orig row to avoid
   extreme weights (clip at 10 = the prior `w=20` upper limit).
3. Parameterize recipe to accept `ORIG_DENSITY_RATIO=1` and use those
   weights when concatenating orig.
4. 5-fold seed=42 retrain. 4-gate analysis.

### Risks
- Density ratios have high variance; clipping to 10 may not be
  enough.
- The 2026-04-22 NN-on-orig family closure showed that orig's joint
  is *fundamentally* different from synth's — IS estimation may
  amplify that mismatch on rare-class rows where orig has few
  examples (336 High rows in orig).

### Bayesian prior
**~12% LB lift.** Heavy-aug nulled before; this is principled but
the orig-as-training-data lever is structurally bounded. Expected
outcome: NULL with informative diagnostic value (confirms density-
ratio doesn't rescue the orig-anchor lever family even when done
properly).

---

## Option C — Conformal score=3-flip detector using AV-score as a feature

### Mechanism
Train a binary specialist on score=3 ∩ teacher_argmax=Low rows (the
single largest flip cluster: 4,899 of 10,304 flips, 4.80% rate).
Inputs: existing recipe dist features + `1 − P(synth)` from the AV
classifier as the new top-1 feature. Target: `(y == Medium)`.
Conformal calibration to find the threshold τ where Wilson 90% lower
CI on precision ≥ 39.3% (the macro-recall break-even floor for
Low↔Medium overrides).

### Why now
- The 2026-04-26 spec_lm_v3 specialist on the same domain hit AUC
  0.827 but failed the precision break-even floor. AV-score is
  novel input that wasn't available then.
- The 2026-04-26 score=6 deep-dive (5 stages) closed at "feature-
  indistinguishability between teacher-residual missed-H rows and
  cell mean." The score=3 boundary is structurally similar but
  with a different break-even (39.3% L↔M vs 8.1% M↔H).
- Per-score AUC of P(synth) at score=3 is ~0.49 in the 10k
  diagnostic — i.e., **no signal at score=3**. This is bad news for
  Option C and good news for the integrity of the methodology
  (we'd be lying to ourselves to push C without that caveat).

### Concrete plan (~40 min CPU)
1. Pull AV-score features from Option A's preflight.
2. Train binary XGB on score=3 ∩ teacher-argmax-Low rows. Add
   AV-score + recipe distances + 7 non-rule continuous features.
3. Conformal calibration with Mondrian split: τ such that
   precision_lower_CI ≥ 39.3%.
4. Apply to test: compute number of overrides at deployable τ.
5. Macro-recall delta on OOF + 4-gate.

### Bayesian prior
**~5% LB lift.** Per-score AV-AUC at score=3 is ~0.49 (no signal
at the dominant flip band). The break-even precision 39.3% (4×
stricter than score=6's 8.1%) and per-CLAUDE.md's prevalence floor
make the override domain mathematically too small. **Likely closes
as a 31st saturation** but could surface a subtle insight if AV-score
adds even 1-2 percentage points of precision.

### When to skip
If Option A's preflight shows AV-score per-fold AUC < 0.55 on score=3
flips specifically, skip Option C entirely (the 10k-subsample 0.49
estimate is unstable at small n_flip but already directional).

---

## Side-by-side

| option | mechanism                              | cost   | EV     | risk profile         |
|--------|----------------------------------------|--------|--------|----------------------|
| A      | AV-score as recipe FE                  | ~80m   | 20%    | wide_fe-style null   |
| B      | density-ratio importance weighting     | ~50m   | 12%    | density-instability  |
| C      | conformal score=3 specialist + AV-score | ~40m   | 5%     | break-even floor     |

Combined sequential cost (A → B → C) ~3 hours CPU. All three share
the AV-classifier preflight (~10 min) which Option A already needs.

## Recommendation

Run Option A's preflight + standalone diagnostic before deciding on
B and C. If A passes the 4-gate filter, ASK user for an LB probe
before submitting (per CLAUDE.md rule). If A nulls cleanly (Jaccard >
0.85 vs LB-best 4-stack OR errs > anchor ×1.05), B and C are unlikely
to recover the lever family because they all share AV-score as the
new signal source.

If all three null, this becomes the **31st-33rd structural saturation
confirmations**, with three new portable rules (AV-as-feature,
density-ratio reweighting, conformal-with-AV) adding to LEARNINGS.md.

**No LB probe in this proposal.** Final-selection lock unchanged:
PRIMARY 0.98094 + HEDGE 0.98005 (audit-F1 swap). Two days to
deadline 2026-04-30.
