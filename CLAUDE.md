# CLAUDE.md

Guidance for Claude Code when working in this repository.

## ⚠️ FIRST THING TO DO IN EVERY NEW SESSION

**If `data/train.csv` does not exist, run `./bootstrap.sh` before anything else.**

Containers are ephemeral — competition data is re-downloaded on each fresh
session. `bootstrap.sh` installs deps and fetches `train.csv`, `test.csv`,
and `sample_submission.csv` via `kaggle competitions download`. It
auto-uses the `KAGGLE_API_TOKEN` env var (already configured at the
container level) and falls back to an interactive prompt if absent.

Do **not** use `download_data.py` to get the competition data — that
script targets the optional `l3llff/irrigation-water` *dataset* (real-
world data the synthetic set was generated from), not the competition.

## Competition

- **Name**: Predicting Irrigation Need (Playground Series - Season 6, Episode 4)
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e4
- **Slug**: `playground-series-s6e4`
- **Task**: 3-class classification (`Low` / `Medium` / `High`) on tabular data
- **Metric**: balanced accuracy (macro-recall)
- **Deadline**: ~2026-04-30 (10 days to go as of 2026-04-20 — confirm on Timeline page)
- **LB submission budget**: 10 / day, 2 final submissions selected, 0 spent at kickoff
- **Team size limit**: 3
- **Data license**: CC BY 4.0

### LB state at kickoff (2026-04-20)

- **Top score (rank 1)**: 0.98219 — Chris Deotte
- **Rank 100 score**: 0.98114 (huge tied pack at exactly 0.98114 from ~100 through 108+)
- **Gap top ↔ tied pack**: ~0.00105 (~1 part in 1000)
- **Not yet submitted**
- Implication: beating the 0.98114 "default model" pack is a hard floor
  (basically everyone ran a straightforward LGBM/XGB on the raw features).
  Real gains come from out-of-the-pack tricks: threshold tuning for
  balanced accuracy, adding the original irrigation dataset, careful
  feature engineering, or DGP archaeology on the synthetic data.

See `brief.md` for the full host material (description, rules,
evaluation, data description, host forum posts).

## Commands

```bash
pip install -r requirements.txt

# Download competition data (uses new KGAT_ token format via env var)
KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle competitions download \
  -c playground-series-s6e4 -p data/
unzip -o data/playground-series-s6e4.zip -d data/

# Download the original Irrigation Prediction dataset (extra training data)
python download_data.py
```

## Architecture

```
notebooks/     Narrative notebooks. Final submission notebook lives here.
scripts/       Reproducible analysis and submission-builder scripts.
data/          Competition data (gitignored).
submissions/   Built submission CSVs (only submission_*.csv committed).
plots/         Diagnostics, organised by topic subfolder.
legacy/        Archived exploratory code, stale plots, dead ends.
brief.md       Verbatim host material (description, rules, eval, data).
CLAUDE.md      This file — development log and session guidance.
LEARNINGS.md   Portable patterns for future competitions.
REPORT.md      Work report: observations, models, results, rejected ideas.
README.md      TL;DR + reproduction instructions.
```

## Session log

### 2026-04-20 — kickoff

- Goal: bootstrap the repo, capture brief/rules/LB state, set up Kaggle
  credentials, and queue a first experiment that beats the 0.98114 tied pack.
- Changed: scaffold in place (template + kaggle-kickoff skill); `brief.md`
  populated with competition description, evaluation (balanced accuracy),
  rules, column list, and flagged invariances; `CLAUDE.md` now reflects
  LB state and download commands; competition data downloaded to `data/`
  (train 630k × 20, test 270k × 19).
- LB delta: n/a (not yet submitted).
- Data finding: **class distribution is severely skewed** — Low 58.7%,
  Medium 37.9%, High 3.3%. Under balanced accuracy this means the `High`
  class drives the scoreboard; per-class threshold tuning is the highest-
  expected-value first experiment.
- Next bet: LGBM baseline on raw + target-encoded categoricals, OOF probs
  from stratified 5-fold CV, then grid/Brent search over per-class
  thresholds maximizing macro-recall. Submit only after comparing OOF
  balanced accuracy of argmax vs tuned decision rule — if tuned rule
  doesn't beat argmax on OOF, re-examine before burning a sub.

### 2026-04-20 — benchmarks + EDA report

- Goal: land a reproducible EDA (on a held-out subsample) and a dummy +
  LGBM benchmark with decision-rule ablation on OOF.
- Changed: `scripts/eda.py` now stratified-subsamples 50% of train
  (seed=42) and emits `plots/eda/report.html`, a self-contained HTML
  with embedded PNGs + feature-signal ranking tables; `scripts/benchmark.py`
  runs the 5-fold stratified LGBM pipeline and saves OOF + test probs
  to `scripts/artifacts/`; `submissions/baseline_lgbm_{argmax,tuned}.csv`
  generated but not submitted.
- Results (OOF balanced accuracy, seed=42, 5-fold CV):
  - majority / random baselines → 0.3333 (floor)
  - LGBM argmax → 0.96135
  - LGBM prior-reweight argmax → 0.97065 (+0.0093)
  - **LGBM tuned log-bias → 0.97097** (+0.0003 over prior-reweight)
- Best log-bias: Low +0.23, Medium +0.67, High +3.40 — matches the
  balanced-accuracy intuition that `High` needs a large positive bump.
- Confusion-matrix mass lives in Medium↔High; Low is essentially
  solved.
- LB delta: n/a (still no submissions; 10/10 day budget intact).
- Next bet: we're ~0.010 below the 0.98114 tied pack with a
  no-feature-engineering single-seed LGBM. Cheapest gains: (a) 3–5
  seed bag of LGBM, (b) richer feature interactions (esp.
  Soil_Moisture × Rainfall, Crop_Growth_Stage × Mulching_Used), (c)
  try XGBoost or CatBoost and blend.

## Hypothesis board

- **Open**:
  - Incorporating the original Irrigation Prediction dataset (explicitly
    allowed) may help, but may also hurt if its DGP differs from the
    synthetic train distribution. Test as a controlled ablation.
  - The huge tie at 0.98114 suggests a "ceiling" from the public baseline
    everyone is running. Room to move is likely in (a) ensembling across
    seeds/models, (b) threshold tuning, (c) leveraging the ordinal
    structure (`Low < Medium < High`) via ordinal-aware losses despite
    the metric being order-agnostic.
  - Most of the residual error is Medium↔High confusion. Feature
    interactions that separate these two classes (e.g. Soil_Moisture ×
    Rainfall_mm, Crop_Growth_Stage × Mulching_Used) should move bal_acc.
- **Confirmed**:
  - Default `argmax` is suboptimal under balanced accuracy when classes
    are imbalanced → prior-reweight + coord-ascent log-bias moves OOF
    from 0.96135 → 0.97097 (+0.0096). Keep this as the decision rule
    for every subsequent model.
- **Ruled out**: (none yet)
- **Parked**:
  - Seed recovery / DGP archaeology on the synthetic generator — high
    effort, unclear payoff with only 10 days; revisit if stuck above
    0.9815.

## Playbook

The reusable Kaggle playbook lives at
<https://github.com/chris0leite-ui/kaggle-claude-code-setup> (branch
`claude/kaggle-playbook`). Kickoff steps, workflow norms, and
methodology are maintained there — update that repo when a transferable
lesson surfaces.
