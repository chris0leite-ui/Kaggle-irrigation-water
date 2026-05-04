# Appendix B — Glossary

Compact definitions for jargon used in this write-up. ML basics
assumed; everything beyond that is defined here.

## Cross-validation

**OOF — out-of-fold predictions.** When you split the training data
into K folds and train K models (each leaving one fold out), the
predictions on each fold's held-out rows are concatenated into an
"OOF" array of the same length as training. OOF score is your
honest estimate of generalization without touching test.

**5-fold StratifiedKFold.** K=5 split that preserves class
proportions in each fold. Standard for imbalanced classification.

**fold-std.** Standard deviation of the per-fold metric. A typical
proxy for OOF estimation noise.

**GroupKFold.** Variant where rows sharing a group ID are kept in
the same fold. Used when leakage is possible across rows (e.g., same
patient, same session, same neighbour).

## Metrics

**Balanced accuracy / macro-recall.** Mean of per-class recall. The
irrigation comp's metric. Treats each class equally regardless of
prior; punishes models that ignore rare classes.

**log-bias / threshold tuning.** Adding a per-class constant to the
log-probabilities before argmax. Equivalent to a class-conditional
prior shift. For balanced accuracy, tuning log-bias on OOF picks the
macro-recall-optimal operating point.

## Modelling

**LGBM / XGBoost / CatBoost.** Gradient-boosted decision tree
libraries. Standard tabular toolkit.

**Target encoding (TE).** Replace a categorical value with the mean
target conditional on that value. Must be done out-of-fold to avoid
leakage.

**Multi-seed bagging.** Train the same model with multiple random
seeds; average their probabilities. Reduces variance without
changing bias.

**Specialist split.** Train a separate model on a subset of the data
(e.g., the rare-class rows) and route relevant test rows to it.

## Stacking

**Component / stack input.** A single OOF predictor feeding into a
meta-stacker.

**Meta-stacker.** A second-level model (linear / RF / GBT) trained
on a stack of OOF columns.

**Bank.** A collection of meta-stacker components. "14-bank" =
14 components.

**Natural calibration.** Probabilities that already sum to the
class priors. Useful as a meta-stacker input because it's bias-
neutral.

## Leakage

**OOF inflation.** OOF score artificially higher than true
generalization. Common cause: information from one fold's training
contaminating another fold's OOF prediction.

**Stacking feature leak.** When meta-stacker input columns are
derivatives of each other, the meta-stacker overfits the
inter-column relationship rather than learning new signal.

**Minimal-input meta sanity check.** Train candidate meta with
ONLY 2 components (anchor + new). If 2-comp OOF lands below anchor,
the N-comp lift was cross-component memorization.

**Grid-search selection bias.** Choosing a hyperparameter by
maximizing OOF inflates the OOF estimate. Worse with denser grids.

## Comp-specific

**DGP — data-generating process.** The host's labelling rule. For
this comp, a closed-form integer rule on 6 features through a small
NN that flipped ~1.6% of labels.

**Override mechanism.** Hand-coded selective row flips applied to
an LB-best primary CSV, gated by multi-rule consensus.

**Triple-consensus.** Override fires only when 3 independent
predictors agree. Used in `submission_idea4b`.

**4-gate filter.** Pre-LB-probe checklist (G1 standalone OOF /
G2 blend lift / G3 net-rare-class-flip ratio / G4 direction
asymmetry).

**Saturation event.** A null candidate at the current LB. The agent
logs each saturation in `audit/`. By comp end, 48 saturation events
at LB 0.98150.

**Calibration ladder.** Running table of (mechanism, OOF, LB, gap).
The single most-checked artifact during the comp.

## Process

**Kaggle CLI.** `kaggle` command-line tool. Used for:
`kaggle competitions submissions <slug>` (read LB),
`kaggle competitions submit <slug> -f <csv> -m <msg>` (write LB,
counts against daily budget).

**Submission budget.** 10 per day on Playground, 5 on most Featured.
Plus 2 final submissions selected for private LB scoring at
deadline.

**Public LB / private LB.** Public is scored on a small held-out
slice (typically 20-30% of test); private on the rest. Final
ranking uses private.

**Probe resolution.** The smallest LB delta the public split can
reliably distinguish. For an 80/20 public split with 270k test, the
floor is around 0.00005.

**Plateau.** A stretch of nulls at the same LB. Triggers the
Research-loop in this framework.

**Persona rotation.** Re-prompting the agent with a different role
(senior / junior / analyst / researcher / "10 wild options") to
break stuck-loop framing.
