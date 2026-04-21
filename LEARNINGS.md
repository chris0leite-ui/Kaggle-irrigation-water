# Learnings

Portable patterns from this competition. Keep entries short, concrete,
and framed so a future-you can apply them without rereading the full
`CLAUDE.md`. Prune and promote the best items into the global playbook
at <https://github.com/chris0leite-ui/kaggle-claude-code-setup>.

## Data quirks

- (e.g. sentinel values, selection bias between features, label noise)

## Modelling

- (e.g. which families extrapolated well, which leaned on spurious
  correlations, what CV scheme matched LB)
- **Post-hoc log-bias coord-ascent subsumes per-tree majority
  undersampling for balanced-accuracy optimization.** If a model's
  softmax probabilities are well-calibrated, log-bias tuning picks the
  macro-recall-optimal operating point directly. Wrappers that rebuild
  balanced probabilities during training (BalancedRF, EasyEnsemble,
  RUSBoost) end up at the same operating point through a different
  mechanism. On a 3-class tabular problem with 58.7/37.9/3.3% priors,
  LGBM + log-bias (0.97271) dominated all three balanced-ensemble
  variants (0.965–0.969) and blended with them added ≤+0.0001. Rule:
  don't run balanced wrappers as a diversity source when log-bias is
  already in the pipeline — they're a different parameterisation of
  the same correction.

## Multi-class imbalance — tactical gotchas

- **AdaBoost-family defaults silently break on 3-class.** In sklearn
  1.8 `AdaBoostClassifier`'s `algorithm` param was removed; in
  `imblearn.ensemble.RUSBoostClassifier` the default `algorithm='SAMME'`
  with decision-stump base learner collapses to `bal_acc=0.333` on a
  3-class problem because the first weak learner's error exceeds
  `1-1/K=0.67` and `α≤0` ends boosting. Fix: pass
  `estimator=DecisionTreeClassifier(max_depth=5)` (or deeper).
- **`EasyEnsembleClassifier` defaults are stump-AdaBoost inside a
  bagging loop.** Same stump-collapse failure as above. Swap the
  inner estimator to a deeper AdaBoost over `max_depth=5` trees.
- **`prior-reweight` is degenerate on already-balanced probs.**
  Dividing by class priors and argmax-ing collapses EasyEnsemble /
  RUSBoost outputs to `bal_acc≈0.333`. Use argmax or coord-ascent
  log-bias starting from `-log π`, not fixed prior-reweight.

## DGP / archaeology

- (e.g. seed recovery attempts, closed-form recovery, pooled-feature
  shift analysis)

## Process

- (e.g. what stop-conditions worked, how to budget LB submissions,
  signs that CV–LB divergence is diagnostic)

## Rejected ideas

- (with one-line reason each)
