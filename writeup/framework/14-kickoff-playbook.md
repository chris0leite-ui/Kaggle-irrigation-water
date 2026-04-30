# 14 — Kickoff playbook

A day-by-day plan for the comp lifecycle. Designed for a 30-day
Playground or 60-day Featured tabular comp; scale phases as needed.

## Day 1 — Setup and settle facts

**Goal**: every settled-once fact in `comp-context.md`. First
submission of a baseline. CV well-calibrated to LB.

1. Clone the repo template ([13-repo-template.md](13-repo-template.md)).
2. Fill `comp-context.md` from the comp page. Don't re-ask these
   facts later.
3. Run `bootstrap.sh`. Verify train/test/sample_submission landed.
4. **EDA on a 50% stratified subsample**: data shape, class priors,
   missingness, train/test categorical drift, top-N feature signals
   (F-stat / chi²).
5. **Write `brief.md`**: verbatim host material (description, eval,
   data, rules). Keep it ≤150 lines.
6. **Baseline LGBM**: tuned via prior-reweight + log-bias for the
   metric. 5-fold StratifiedKFold OOF.
7. **First submission**: ask PI, single-shot. Goal is to *calibrate*
   OOF→LB gap, not to score.
8. **Compute and commit**: OOF→LB gap. If within one fold-std,
   trust OOF deltas going forward.

**Exit**: `comp-context.md` populated, baseline submitted, OOF→LB
gap < one fold-std.

## Day 2-3 — Domain hypothesis seeds + heuristics

**Goal**: cheapest-possible mechanisms enumerated, lift available
bounded.

1. **Domain research turn** (don't skip even if it feels off-topic).
   Read what irrigation/finance/medical/etc. domain knowledge exists
   for this problem. Capture in `DOMAIN.md`. Use as hypothesis
   seeder; don't deeply engineer features yet.
2. **DGP archaeology** (if synthetic): brute-force candidate rules,
   inspect train labels, check for closed-form patterns. The 6
   features that matter are usually visible from F-stat and chi².
3. **Heuristic baselines** before any tree/NN:
   - Single-feature thresholds (H1).
   - Hand-coded rules from domain reading (H3).
   - Closed-form rule if DGP archaeology found one.
4. **Tree baseline**: LGBM, XGBoost. Tune log-bias for metric.

**Exit**: 2-3 heuristic baselines + 2 tree baselines, all with OOF
and (optionally) LB. The "lift available" envelope is bounded.

## Day 4-7 — Recipe ladder

**Goal**: a single 5-fold pipeline (`recipe.py`) that captures the
best preprocessing + features + model + tuning. This is the
workhorse.

1. **Target encoding** on categorical-rich subset.
2. **Multi-seed bagging** (3-5 seeds) for variance reduction.
3. **Specialist split** for rare classes (if class imbalance
   matters under metric).
4. **Tune log-bias / threshold** for the metric.
5. **Calibration check**: each new component, refresh OOF→LB ladder.

**Exit**: `recipe.py` is the LB-best non-stack mechanism, OOF→LB gap
known and stable.

## Day 8-15 — Stacking + orthogonality

**Goal**: a bank of error-orthogonal components feeding a meta-stacker.

1. **Build a component bank**: 5-15 OOF predictors from different
   model families (LGBM/XGB/CatBoost/RF/MLP) under different
   reweighting schemes.
2. **Orthogonality check**: pairwise Jaccard < 0.85 on OOF error
   sets. Skip components that overlap.
3. **Meta-stacker**: linear / RF / LGBM on OOF stack. **4-gate
   filter** before any LB probe.
4. **Minimal-input meta sanity check** on every meta candidate.

**Exit**: a 4-gate-passing meta candidate above recipe.

## Day 16-25 — Orthogonal mechanisms (where the real lift hides)

**Goal**: mechanisms NOT on the standard tabular menu. This is
where Research-loop matters most.

Examples from the irrigation comp:

- **Override decision rules**: hand-coded selective flips on the
  LB-best primary, gated by multi-rule consensus.
- **Bank-confidence-ranked unions**: union the top-K confident
  predictions across the bank.
- **Boundary-routed predictors**: route boundary rows to a
  specialist model.

Required process:

1. **Research turn first** (Kaggle public notebooks, prior-comp
   writeups in same domain).
2. **Heuristic-first** for each new mechanism family.
3. **4-gate filter**, theory-only hyperparameters.
4. **One LB probe per family**, then move on.

**Exit**: 2-5 orthogonal mechanism families tested. The current
LB-best is one of them.

## Day 26 - (deadline - 4) — Saturation + plateau-breaks

**Goal**: every plateau triggers a Research-loop. Don't declare
ceilings.

1. After every 5 saturation events at the same LB → Research-loop
   (mandatory).
2. After every 3 consecutive nulls → persona rotation (Junior /
   Researcher / 10-wild-options).
3. Update `REPORT.md` weekly with the running calibration ladder.

## Final 3 days

**Goal**: lock primary + hedge. No new families.

1. **Day -3**: identify the LB-best primary (highest LB, mechanism
   well-understood). Identify a **hedge candidate** with
   orthogonal failure mode.
2. **Day -2**: confirm both submissions are stable (re-build from
   committed artifacts, diff against earlier emit).
3. **Day -1**: hold submissions. No panic experiments.
4. **Day 0**: PI selects the 2 final submissions. Stop.

**Anti-rule**: don't **ever** lock and stop early. The framework's
NEVER-LOCK-FINALS rule applies until the final 3-day window.

## After the comp

1. **Postmortem**: write `writeup/postmortem/` (this exact format).
2. **Update LEARNINGS.md**: portable patterns extracted from this
   comp, indexed for the next one.
3. **Update the comp-template repo**: any new files worth lifting
   forward.
4. **Update the skill** (`skill/kaggle-comp/`): refresh examples
   from the latest comp.
