# Kaggle / ML Competition Kickoff Playbook

Load at the start of any new competition (new repo, new session). Ask
for these proactively; do not wait to be prompted.

## Core principle: git is the only communication channel

Between sessions, between parallel Claude instances, between you and
the user — **assume nothing survives in chat**. Every insight, result,
and artifact must be committed and pushed. Specifically:

- Source code, notebooks, and scripts go in `scripts/` and `notebooks/`.
- **OOF (`oof_*.npy`) and test (`test_*.npy`) prediction arrays are
  first-class outputs of every training script**, not debug artefacts.
  Commit them via a `.gitignore` allowlist so small arrays survive a
  container rehydrate but raw data does not.
- Per-experiment `*_results.json` in `scripts/artifacts/` for every
  fixed-bias sweep, HP search, or blend result.
- Session logs in `CLAUDE.md`, portable lessons in `LEARNINGS.md`,
  work-report in `REPORT.md`, LB-best + reproduction in `README.md`.
- Submission CSVs under `submissions/submission_*.csv` are committed
  by default so anyone can reproduce the LB probe.

A session that didn't `git push` is a session that didn't happen.

## Day 1 checklist

1. **`brief.md` in the repo**: paste the full host material —
   description, rules, eval page, data description, host forum /
   notebook comments. Invariances and constraints often live here.
2. **LB submission budget**: daily limit, total limit, already-spent.
   Track remaining; rank candidates by expected information gain.
3. **Current LB rank + distances** to clusters the user cares about
   (top-N, median, target).
4. **Deadline + weekly hours** the user expects to invest.
5. **Between-session channels**: forum, CSV inspection, collaborators.
6. **Tooling check**: `which kaggle`, `ls ~/.kaggle/kaggle.json`,
   `env | grep -i kaggle`. On Claude Code web, use the environment-
   variable UI for `KAGGLE_USERNAME` / `KAGGLE_KEY` / `KAGGLE_API_TOKEN`.

## Session arc (order matters — resist skipping)

The arc below is the staircase; each rung tells you what the next
rung should look like. Skipping a rung almost always wastes compute
on the rung after next.

### 1. Setup — <1 hour
- Bootstrap data download (script, not manual).
- Establish folder structure: `data/ submissions/ scripts/
  scripts/artifacts/ notebooks/ plots/ legacy/ src/`.
- `.gitignore` with allowlist exceptions for small `.npy` and final
  `submission_*.csv`.
- Confirm Kaggle CLI auth end-to-end with a no-op `kaggle competitions
  list`.

### 2. Floor baselines — <30 minutes
- Majority class / stratified random to establish the floor.
- For imbalanced multi-class: the floor is `1 / n_classes` (macro
  metrics) or `max(p_c)` (raw accuracy).
- **A result below the floor is a bug, not a model — stop and find it.**

### 3. EDA — 1–2 hours
- Class distribution (count + percentage).
- Feature-by-feature: missingness, distribution, categorical
  vocabulary, train/test drift.
- Signal ranking: F-stat for numerics, chi² for categoricals.
- **Emit a self-contained HTML report** (`plots/eda/report.html`)
  with base64-embedded images. Reference it; don't regenerate.
- Split the EDA workload: use a **50 % stratified subsample** so
  the other 50 % is held out for model OOF alignment.

### 4. Heuristics + domain knowledge — 2–4 hours
- Research the domain: Wikipedia, top papers, textbooks, kaggle
  forum. Write a `DOMAIN.md` capturing physical / statistical
  priors, feature relationships, canonical lookup tables.
- Build a hand-crafted heuristic using domain intuition:
  *single dominant feature + 1–2 thresholds*.
- Build 2–3 linear models (e.g., multinomial logit on minimal
  feature sets) with tuned decision rule.
- **Each heuristic / linear tier tells you something:**
  - Dominant-feature heuristic → how much signal a single variable
    carries (bounds the lower-tier ceiling).
  - Linear models → whether signal is mostly additive.
  - Gap between linear and tree → how much of the lift is
    interactions vs main effects.

### 5. Out-of-the-box tree baselines — 2 hours
- LGBM, XGBoost (level-wise), CatBoost — one vanilla run of each
  with default-ish HPs (one reasonable set, not a sweep).
- Apply the metric-specific decision rule:
  - Balanced accuracy / macro-F1 with imbalanced classes →
    coord-ascent on per-class log-bias.
  - MAE / RMSE → no adjustment.
  - AUC → raw scores.
- The baseline tier locks in where the "default" solution lives.
  Beating it by +0.01 takes the bulk of the competition.

### 6. Advance-the-front — rest of the competition
- Feature engineering (run 3–5 families, keep what sticks).
- Seed bagging, HP tuning, stacking.
- External data, DGP archaeology, distillation, pseudo-labeling.
- Domain-aware model classes (ordinal, counts, tabular NN).
- Iterate. See "Optimization gradient discipline" below.

### 7. Closeout — last 1–2 days
- Two final submissions: one narrow-variance proven (hedge), one
  stretch (best-expected but more variance).
- README updated with LB rank + reproduction recipe.
- LEARNINGS.md promoted to the playbook.

## Optimization gradient discipline (simulated annealing for effort)

The per-experiment improvement bar should **monotonically decrease**
through the session:

| Phase | Bar for "interesting lift" | Typical cost ceiling |
|---|---|---|
| Baselines | +0.05 (vs floor) | 30 min |
| Out-of-box trees | +0.01 | 1 hour |
| FE / HP tune | +0.003 | 2 hours |
| Stacking / archaeology | +0.001 | 4 hours |
| End-of-competition | +0.0003 | 8 hours |

**Anti-patterns to refuse:**

- Doing a 4-hour experiment for +0.0003 lift while a +0.01 lever is
  still un-tried. Reject high-effort, low-expected-lift work until
  the cheap levers are exhausted.
- Conversely: accepting a +0.0003 null as "good enough" mid-session
  and stopping. If you have 3 days left and the cheap bets are gone,
  the right move is either a big-swing architectural experiment or
  saving LB subs for the next session.

Track the gradient explicitly in `CLAUDE.md`: every daily-log entry
records (cost, expected delta, actual delta). When actual << expected
three experiments in a row, cheap levers are exhausted and the bar
shifts down.

## Prune complexity between steps

Failure modes accumulate faster than wins. Between phases, actively
cut:

- Script files that produced null results (move to `legacy/`).
- Features that didn't improve OOF (don't carry them into the next
  FE experiment).
- Submission CSVs older than a week unless they're on the hypothesis
  board.
- Plots and analysis artifacts no one uses.
- Branches of experiments that haven't been merged and have no
  active owner.

If a complexity isn't paying rent in the current phase, it's
actively slowing down the next one.

## Artifacts for blending

Ensemble diversity is worth more than standalone improvements past
~0.97 on tabular competitions. To compound across experiments:

- **Every training script emits `oof_*.npy` (shape `(n_train,
  n_classes)`) and `test_*.npy` (shape `(n_test, n_classes)`).**
- OOF / test arrays use a **fixed fold split** (e.g. `StratifiedKFold
  seed=42, n_splits=5`) documented in `OOFS.md`. Every script that
  deviates must state it loudly.
- Commit high-value OOFs via `.gitignore` allowlist. A committed OOF
  array is a blendable asset for future sessions and parallel
  Claude instances.
- `OOFS.md` maintains a table: file, OOF bal_acc / loss, LB (if
  submitted), provenance (script + commit).

## Blending & ensemble methodology

- **Fixed-baseline-bias sweep is the pre-LB filter.** When adding a
  new component to an already-OOF-tuned stack, sweep its weight with
  the baseline's fitted log-bias reused as-is. If fixed-bias OOF
  doesn't lift, the component is redundant — retuning bias on top
  manufactures fake lift that vanishes on LB.
- **Real LB delta ≈ 1/3 OOF delta** when stacking tuned blends on
  tuned baselines. Budget LB submissions with this discount unless
  the lift is architectural (see next rule).
- **Gap shrinkage is the signature of honest architectural signal.**
  If `OOF − LB` shrinks when adding a component, the lever is
  architectural and will transfer. If it grows, it's selection
  overfit. Track the gap in a calibration-ladder table.
- **Diversity from "ignoring a feature class" beats "using it
  differently".** A model restricted to a disjoint feature subset
  often blends into an ensemble of full-feature models, even if its
  standalone accuracy is near-random. Architectural diversity on the
  same feature set (LGBM vs XGB vs EBM) is usually null.
- **Rank / Borda aggregation is dominated by prob / log space** for
  log-bias-tuned multi-class decision rules. Row-softmaxed ranks
  erase the absolute-probability separation the metric's decision
  rule needs.
- **Jaccard overlap is necessary but not sufficient** for a useful
  blend. Compute `Jaccard(err_A, err_B)` on OOF error rows; skip the
  blend sweep if > ~0.90, but low Jaccard alone doesn't guarantee
  lift — also check that the candidate's unique errors have smaller
  magnitude than the base's correct-prob on the same rows.
- **Specialists need 20–80 % minority class** in their sub-domain.
  Below that threshold the specialist collapses to "predict the
  majority" and bal_acc approaches ~0.5.
- **Don't augment specialist training with clean data** when the
  specialist's job is to deviate from a clean predictor. The clean
  data pulls its boundary toward the mainstream, eroding exactly the
  signal you wanted.
- **Deterministic > learned at OOF parity.** When a rule is ≥99.99 %
  accurate on a sub-domain and a learned model matches OOF exactly,
  prefer the rule — it has zero test-time variance; learned models
  misfire on OOD extremes.
- **Training-distribution engineering ≠ inference routing.** Dropping
  easy rows from training AND routing them at inference helps (implicit
  class-prior rebalancing). Routing alone at inference (no training
  change) often hurts.
- **Shift-target framings collapse when any class owns ≥95 % of
  rows.** Early stopping saturates on the majority; model becomes a
  one-class predictor. Direct-y 3-class keeps per-row discrimination
  alive.
- **Feature-subspace bagging on a small pool underperforms the full
  model.** Rule of thumb: feature pool size ≥ ~3× subset size, else
  subsets share too many features and the ensemble collapses to a
  weaker full-feature version.

## Daily-log format

Keep entries compact and uniform. One entry per experiment or micro-
phase, not per commit.

```
### YYYY-MM-DD — short-title

- Goal: [one-line question this entry answers]
- Changed: [1–3 lines of files / commands]
- Result:
  - [metric @ argmax]
  - [metric @ tuned]
  - [LB if submitted]
- Read-out: [what you learned, 1–3 sentences]
- Next bet: [the single next experiment, or "parked"]
```

## Hypothesis board (in CLAUDE.md, end of session log)

- **Current best**: [file path, OOF, LB, gap]
- **Open bets**: ranked by expected-ROI / effort
- **Ruled out this competition**: with 1–2 line reason
- **Parked**: for possible revival if stuck

## Anti-patterns (refuse these)

- **LB submission without an OOF gate.** Every submission should
  have a documented predicted LB range derived from the calibration
  ladder. Burning a submission on a "let's just see" is expensive.
- **Retuning log-bias on layered blends.** Each layer of OOF tuning
  compounds selection overfit; the LB gap grows ~linearly with the
  number of tuned hyperparameters layered on top of each other.
- **"One more experiment" when current Δ < fold std.** If the last
  three experiments landed inside 1σ fold-std noise and the bar is
  still at +0.001, the lever is exhausted. Pivot, don't grind.
- **Retraining a baseline that was OOF-tuned** as the base for new
  stacking. Build new components against the *raw* component OOFs,
  not tuned ones, so the fixed-bias sweep has a clean baseline.
- **Parallel-session LB races without coordination.** When two
  Claude instances work on the same competition, both can burn LB
  subs. Agree a budget split via `CLAUDE.md` or branch-naming
  convention.

## Session-close checklist

- [ ] All OOF / test `.npy` arrays committed (or logged as null in
      `scripts/artifacts/*_results.json`).
- [ ] `CLAUDE.md` daily-log entry written.
- [ ] `LEARNINGS.md` updated with any new portable pattern.
- [ ] `REPORT.md` reflects current best model tier.
- [ ] `README.md` LB-best line + reproduction recipe current.
- [ ] `NEXT_STEPS.md` hypothesis board updated.
- [ ] `git push` to both feature branch and `main` (if merged).

## Methodology principles

- Understand the problem before optimising. Interpretable models,
  domain research, causal discovery, seed recovery, pooled-feature
  shift analysis, and reading host material are all routes.
- CV–LB divergence is a diagnostic signal, not noise. Track the
  multiplier; it measures how much training-specific structure is
  being exploited.
- DGP archaeology is a distinct phase with its own tools (rule
  enumeration, residual EDA, closed-form fits). Apply when stuck
  above a plateau the ensemble family can't cross.
- Record dead ends alongside wins. A null result that took 2 hours
  is worth ~500 words of explanation so the next session doesn't
  retry.
- Transferable method > reproducible result. Promote wins to the
  playbook only after a second competition confirms them.
- Gaps between model families carry signal (linear vs GAM =
  nonlinearity; GAM vs EBM = interactions; EBM vs deep tree =
  high-order interactions).
- Parallel Claude instances are complementary; use branch-naming and
  `OOFS.md` to coordinate. Review `main` and other feature branches
  at session start.

## TDD / scaffolding

Red-green TDD against `src/` + `tests/` helps during early modelling
(clean interfaces, preprocessor correctness). It becomes friction
during archaeology, where one-off scripts are faster. Plan to archive
tests to `legacy/` once the pipeline stabilises.
