# Learnings

Portable patterns from this competition. Keep entries short, concrete,
and framed so a future-you can apply them without rereading the full
`CLAUDE.md`. Prune and promote the best items into the global playbook
at <https://github.com/chris0leite-ui/kaggle-claude-code-setup>.

## Data quirks

- (e.g. sentinel values, selection bias between features, label noise)

## Modelling

- **Trees dominate when the DGP is piecewise-constant with integer
  thresholds.** kNN (k=50) on the 6 rule features hit 0.954 tuned,
  below the rule (0.961); LGBM hit 0.973 because it can place splits
  on the exact thresholds and use the remaining features. Default to
  trees on any tabular competition with a synthetic-generator feel.
- **Bias tuning is non-optional for imbalanced balanced-accuracy
  metrics.** +0.010 on LGBM, +0.010 on kNN; coord-ascent over
  per-class log-bias converges in < 10 s.

## DGP / archaeology

- **Look for 100 % rule recovery on "original" datasets first.** If
  the Kaggle Playground set links to a real-world dataset, try to
  fit it deterministically with trees before assuming noise is
  irreducible. RF importance collapsing onto a small feature subset
  + an unconstrained DT reaching 100 % accuracy in low-leaf-count
  is the signature of a host-written integer rule. We found ours
  on day 1 via this path.
- **Split-threshold clustering is a tell.** When every DT split on a
  continuous feature lands on the same round number (25, 300, 30,
  10 in our case), you are looking at the generator's threshold,
  not a learned approximation.
- **Populated lookup tables reveal closed forms.** Discretise the
  continuous features at those thresholds, build a cross-product
  table, and check for any mixed-label cells. Zero mixed cells =
  deterministic rule, solvable by inspection.
- **Synthetic-generator noise is boundary-local and one-step.**
  Rows deep in a class are 100 % rule-correct; boundary-band rows
  flip to neighbour classes at rates that track distance from the
  rule thresholds. Signature of a tabular generative model
  (TabDDPM, VAE, GAN) trained to imitate the rule.

## Domain knowledge

- **Research domain knowledge early as a hypothesis-seeder, but do
  not invest in physics-faithful feature engineering until the DGP
  is confirmed.** The agronomy primer we wrote on day 1 (soil-water
  balance, FAO-56 Kc, Penman–Monteith, Indian cropping seasons)
  *did* point us at the right six features — Soil_Moisture,
  Rainfall, Temperature, Wind, Mulching, Crop_Growth_Stage — so the
  time wasn't wasted. But hand-engineered physics columns in LGBM
  gave Δ = −0.00052 (null), because a synthetic host-written
  integer rule has no physics in it to recover. The domain primer
  was deleted once the DGP was known; if you ever need it again,
  re-derive from the feature names in < 30 min.

## Process

- (e.g. what stop-conditions worked, how to budget LB submissions,
  signs that CV–LB divergence is diagnostic)

## Rejected ideas

- (with one-line reason each)
