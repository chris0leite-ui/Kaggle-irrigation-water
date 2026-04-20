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

### 2026-04-20 — first submission, CV↔LB calibrated

- Goal: spend one submission to answer whether the 0.98114 tied pack
  is running argmax or already tuned, so downstream decisions aren't
  based on guesses about the pack.
- Changed: `submissions/submission_baseline_lgbm_tuned.csv` committed
  to the repo (gitignore exception) and uploaded to Kaggle.
- Result: **LB public = 0.96972** at rank 726 / 2357 (top 31%).
- OOF vs LB: 0.97097 − 0.96972 = **−0.00125**, inside one fold-std
  (~0.002). CV is well-calibrated; future experiment deltas from
  5-fold OOF can be trusted.
- Read-out: the pack is NOT running raw argmax (that would have landed
  them near our 0.96 tier). They have structural advantages — feature
  engineering, original dataset, seed bagging, better hyperparameters,
  or some combination. Our earlier hypothesis "the pack already uses
  the threshold trick" is confirmed.
- LB budget: 1 / 10 spent today; 9 remaining.
- Gap math for remaining budget: stacking best-case expected deltas
  from NEXT_STEPS steps 3–6 → ~+0.007 → ~0.977, still below the pack
  (0.98114). Step 2 (original dataset) is the swing factor: +0.004 of
  lift from it would put us in pack territory; negative / flat means
  we need to look at the public-notebook recipe.
- Next bet: execute step 2 (original-dataset ablation) and step 3
  (domain features into LGBM) in parallel — both are cheap enough to
  fit in one session, and step 2 alone resolves the biggest
  uncertainty in the remaining plan.

### LB state at kickoff (2026-04-20)

- **Top score (rank 1)**: 0.98219 — Chris Deotte
- **Rank 100 score**: 0.98114 (huge tied pack at exactly 0.98114 from ~100 through 108+)
- **Gap top ↔ tied pack**: ~0.00105 (~1 part in 1000)
- **First submission (tuned LGBM)**: 0.96972, rank 726/2357
- Implication: beating the 0.98114 "default model" pack is a hard floor
  (basically everyone ran a straightforward LGBM/XGB on the raw features).
  Real gains come from out-of-the-pack tricks: threshold tuning for
  balanced accuracy, adding the original irrigation dataset, careful
  feature engineering, or DGP archaeology on the synthetic data.

See `brief.md` for the full host material (description, rules,
evaluation, data description, host forum posts).

## Domain knowledge — pruned

The repo previously carried an agronomy primer (`DOMAIN.md`,
`domain/*.md`) with FAO-56 Kc tables, soil-water-balance equations,
Penman–Monteith evapotranspiration, Indian cropping-season context,
etc. **Deleted 2026-04-20** once the DGP was reverse-engineered: the
synthetic label is produced by a closed-form integer rule on 6
features (see `scripts/dgp_formula.py`), not a physical simulation.
Hand-engineered physics-inspired features added **zero lift** in
LGBM (`benchmark_fe.py` Δ = −0.00052) because the trees already
discover the same interactions.

For the next synthetic competition: **research domain knowledge
early** as a hypothesis-seeder — it told us what "irrigation_need"
means and which feature axes to probe, which pointed us at the rule.
But do **not** invest in physics-faithful feature engineering until
the DGP's actual functional form is confirmed.

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

### 2026-04-20 — domain primer, heuristics, linear formulas, blend

- Goal: build a physical frame of reference (non-tree baselines) so we
  understand how much of the LGBM score is "equation" vs "interaction",
  and test whether weaker models bring orthogonal signal.
- Changed: `DOMAIN.md` (soil-water balance equation, feature-to-term
  mapping, Indian cropping-season context, FAO-56 Kc lookup, soil
  field-capacity lookup — **deleted 2026-04-20** once the DGP was
  shown to be a closed-form integer rule, not a physics sim);
  `scripts/heuristic.py` (no-training, threshold-fit-per-fold
  predictor); `scripts/formula_mnlogit.py` (three hand-crafted
  MNLogit formulas F1/F2/F3); `scripts/benchmark_multi.py` (XGBoost
  done, CatBoost killed at fold 1); `scripts/blend_lgbm_mnlogit.py`
  (blend sweep).
- Results (OOF balanced accuracy, 5-fold stratified, seed=42):
  - Heuristic H1 (Soil_Moisture alone): 0.62911
  - Heuristic H2 (raw water balance, equal z-wts): 0.60606
  - Heuristic H3 (H2 + Kc + mulch + soil cap): 0.63041
  - MNLogit F1 tuned: 0.64721
  - MNLogit F2 tuned: 0.78074
  - MNLogit F3 tuned: 0.73294
  - LGBM tuned (prior result): 0.97097
  - XGBoost tuned (per-fold ~0.961–0.964): ~0.962
  - CatBoost fold-1 argmax: 0.96000 (killed; no edge)
  - LGBM + MNLogit blend (sweep w∈[0,0.5]): Δ = +0.00000
- Observations:
  - Soil_Moisture alone (H1) reaches ~2/3 of the distance from random
    to competitive. The single feature carries a huge fraction of the
    signal, matching its F-stat lead (~82k, 4× the next feature).
  - H2 < H1: equal-weight z-scoring dilutes a dominant signal.
    Heuristic-weight choice is a decision, not a free parameter.
  - H3 ≈ H1: Kc + mulch + capacity add ~0.001 — directionally right,
    too crude to beat the "just sort by soil moisture" baseline.
  - MNLogit F2 > F3: dropping main effects in favor of interactions
    under L2 regularization is an inefficient parameterization.
  - LGBM → H3 = +0.34 bal_acc on the *same* physical features — so
    the dominant gain is from nonlinear interactions, not feature
    selection. Any hand-engineered linear combination is a floor, not
    a ceiling.
  - Blend null result confirms MNLogit is simply too weak to add to
    LGBM. Model-diversity gains need a *strong* second model.
- LB delta: still n/a (0/10 day budget consumed).
- Next bet: feature engineering on LGBM (plug F2/H3 engineered cols
  into LGBM training), seed-bag LGBM, LGBM+XGB blend, then test the
  original Irrigation Prediction dataset as an ablation. Ranked list
  with expected deltas lives in REPORT.md §4.

### 2026-04-20 — LGBM + engineered domain features (null result)

- Goal: test whether hand-built water-balance features lift LGBM
  above the 0.97097 baseline — the highest-ROI item in
  `NEXT_STEPS.md` §3.
- Changed: `scripts/benchmark_fe.py` runs the same 5-fold stratified
  LGBM pipeline with 8 extra cols (`ET0_proxy`, `Kc_stage`,
  `ETc_proxy`, `Soil_deficit`, `Is_Rainfed`, `Eff_Rainfall_active`,
  `Crop_x_Stage`, `Season_x_Region`); artefacts persisted to
  `scripts/artifacts/oof_lgbm_fe.npy`, `test_lgbm_fe.npy`,
  `bench_fe_results.json`; submissions
  `submission_lgbm_fe_{argmax,tuned}.csv`.
- Results (OOF balanced accuracy, 27 features, seed=42, 5-fold CV):
  - LGBM+FE argmax → 0.96133 (baseline 0.96135, Δ = −0.00002)
  - LGBM+FE prior-reweight → 0.96981 (baseline 0.97065, Δ = −0.00084)
  - **LGBM+FE tuned log-bias → 0.97045** (baseline 0.97097,
    Δ = **−0.00052**)
  - Fold std (argmax) = 0.00088 → the drop is well within 1σ noise.
  - Best bias: Low +0.2324, Medium +0.5689, High +3.4008 —
    essentially unchanged from baseline (+0.23 / +0.67 / +3.40).
- Observation: LGBM at `num_leaves=127`, `min_data_in_leaf=200` is
  clearly not leaf-limited — trees already find these interactions on
  their own, so prebuilt versions add no new splits. The "prebuilt
  interactions help when splits are near-leaf-limit" hypothesis in
  NEXT_STEPS.md §3 doesn't hold at this leaf count.
- LB delta: still n/a (0/10 day budget consumed).
- Next bet: seed-bag LGBM (3–5 seeds, retune bias on averaged OOF) —
  cheapest remaining win at expected +0.0005–0.001. Then LGBM+XGB
  blend, then original-dataset ablation. NEXT_STEPS.md §3 downgraded
  to "ruled out"; §4 promoted to top.

### 2026-04-20 — original-dataset ablation + transfer check (small +)

- Goal: resolve NEXT_STEPS §2 — does concatenating the 10k-row
  original Irrigation Prediction dataset (`data/archive.zip`) with
  each training fold improve OOF, and how close are the DGPs?
- Changed: `scripts/benchmark_external.py` runs the concat pipeline
  (5-fold stratified on synthetic; each fold fits on synthetic-train
  ∪ all-original, validates on synthetic-val only, so OOF is
  apples-to-apples with the baseline). `scripts/transfer_check.py`
  trains LGBM on 8k original rows and predicts on the full 630k
  synthetic train, as a DGP-overlap diagnostic. Artefacts:
  `scripts/artifacts/{oof,test}_lgbm_ext.npy`, `bench_ext_results.json`,
  `transfer_check_results.json`. Submissions:
  `submission_lgbm_ext_{argmax,tuned}.csv`.
- Results (OOF balanced accuracy on synthetic folds, seed=42, 5-fold):
  - LGBM+EXT argmax → 0.96208 (baseline 0.96135, Δ = +0.00073)
  - LGBM+EXT prior-reweight → 0.97097 (baseline 0.97065, Δ = +0.00032)
  - **LGBM+EXT tuned log-bias → 0.97124** (baseline 0.97097,
    Δ = **+0.00027**)
  - Fold std (argmax) = 0.00068 → Δ is within 1σ noise but
    directionally positive on every fold.
  - Best bias: Low +0.1324, Medium +0.6689, High +3.4008 (Low
    relaxed ~0.1 vs baseline; Medium/High essentially unchanged).
- Transfer check (train on 8k original, eval on 630k synthetic):
  tuned bal_acc = 0.96278 — only 0.00819 below the synthetic-only
  5-fold OOF despite 63× less training data. Verdict: DGPs overlap
  almost completely; the small concat delta reflects the 10k cap at
  1.6 % of the training pool, not DGP divergence.
- Implications for gap to pack: with EXT our OOF is 0.97124
  (expected LB ~0.96997 given the −0.00125 calibration gap). Pack
  is 0.98114, leader 0.98219. Stacking seed-bag (+0.001) + XGB
  blend (+0.002) + HP/ordinal (+0.001) → best-case ~0.975 OOF →
  ~0.974 LB, still ~0.007 short. The pack likely has a recipe-level
  win we haven't located (HP search at scale, a DGP exploit, or a
  smarter weighting of the external data).
- LB delta: still 1/10 spent.
- Next bet: seed-bag **LGBM+EXT** (not vanilla LGBM) as the new base.
  Then XGBoost with the same EXT concat, then blend. Consider one
  more LB submission of `submission_lgbm_ext_tuned.csv` to confirm
  the small OOF delta transfers — but only after the seed-bag is in,
  since the seed-bag result would be a stronger submit candidate.

### 2026-04-20 — DGP reverse-engineered, closed-form rule submitted

- Goal: find the synthetic-generator rule. Hypothesis: the original
  10k dataset is integer-rule-generated on a small feature subset,
  and the synthetic 630k is the same rule + label-noise near the
  thresholds.
- Changed: `scripts/dgp_formula.py` implements the rule; REPORT.md
  §7 documents derivation; `submissions/submission_dgp_formula.csv`
  built and submitted.
- Rule (perfect on 10k original, 100.000000 % accuracy):
  ```
  dry     = Soil_Moisture < 25
  norain  = Rainfall_mm   < 300
  hot     = Temperature_C > 30
  windy   = Wind_Speed_kmh > 10
  nomulch = Mulching_Used == "No"
  Kc      = 2 if Crop_Growth_Stage in {Flowering, Vegetative} else 0
  score   = 2*(dry + norain) + (hot + windy + nomulch) + Kc
  Low if score<=3 ; Medium if 4<=score<=6 ; High if score>=7
  ```
- Derivation: RF importance collapsed to 6 features; unconstrained
  DT hit 100 % train with 66 leaves at depth 11; tree split
  thresholds clustered on round numbers (25 / 300 / 30 / 10); the
  2⁵ × 4 = 128-cell lookup table over (dry,norain,hot,windy,nomulch)
  × stage had **0 mixed-label cells**. Per-cell inspection revealed
  water axes carry 2× the weight of demand axes and stage acts as
  a +2 bump for active transpiration.
- Synthetic train: rule hits raw acc **0.98364**, bal_acc **0.96097**
  on all 630k rows. Error pattern is strictly boundary-band: rows
  with score 1–3 mis-predicted → Medium (5,269); rows at score 4
  → Low (1,507) or High (1,758); rows 7–9 → Medium (1,692). No
  cross-band errors. Confirms the synthetic = original rule + a
  near-threshold label-flip process.
- LB delta: submitted pure rule → **public = 0.95835**, rank ~N/A
  (below the tied pack). Train bal_acc 0.96097 − LB 0.95835 = 0.00262,
  consistent with the −0.00125 OOF↔LB gap from the tuned LGBM.
- Budget: 2/10 used today, 8 remaining.
- Read-out: the rule alone doesn't beat tuned LGBM (0.96972) because
  LGBM already implicitly learns it. The pack at 0.98114 must be
  using the rule's structure AND a mechanism to recover boundary-
  band flips — either (a) distance-to-threshold features that let
  a model learn where the noise is, or (b) a per-row noise inversion
  specific to the synthetic generator.
- Next bet: add the DGP indicators (score, dry, norain, hot, windy,
  nomulch, Kc) AND distance-to-threshold continuous features
  (Soil_Moisture−25, Rainfall_mm−300, Temperature_C−30,
  Wind_Speed_kmh−10) to LGBM. If the noise is a learnable function
  of distance-to-boundary, tuned OOF should break 0.975+.

### 2026-04-20 — LGBM+DGP: +0.00174 OOF, +0.00165 LB

- Goal: inject DGP indicators + distance-to-threshold features into
  LGBM to let the model learn the boundary-band noise.
- Changed: `scripts/benchmark_dgp.py` (7 DGP cols + 8 signed/abs
  distance cols = 15 extra features, 26 total); artefacts
  `scripts/artifacts/{oof,test}_lgbm_dgp.npy`,
  `bench_dgp_results.json`; submissions
  `submission_lgbm_dgp_{argmax,tuned}.csv`.
- Results (OOF bal_acc, 5-fold, seed=42):
  - argmax                → 0.96349 (baseline 0.96135, Δ = +0.00214)
  - prior-reweight argmax → 0.97250 (baseline 0.97065, Δ = +0.00185)
  - **tuned log-bias      → 0.97271** (baseline 0.97097, Δ = +0.00174)
  - Best bias: Low +0.03, Medium +0.67, High +3.40 — Low collapsed
    from +0.23 to +0.03, meaning the DGP indicators already push
    Low logits to the correct side; the bias only still needs to
    lift Medium/High.
- LB delta: submitted tuned → **public = 0.97137** (prior best
  0.96972, Δ = **+0.00165**). OOF↔LB gap −0.00134, calibration
  holds.
- Budget: 3/10 used today, 7 remaining.
- Gap to pack 0.98114 closes to ~0.00977 (was 0.01142).

### 2026-04-20 — flip detector + gated attempts (diagnostics clean, execution null)

- Goal: model the 10,304 residual flips directly. Test whether they
  are feature-predictable at all, and if yes, build a gated
  rule+model pipeline to exploit them.
- Changed: `scripts/flip_detector.py`, `scripts/gated_pipeline.py`
  (v1 uses LGBM+DGP as direction model), `scripts/gated_pipeline_v2.py`
  (v2 uses specialist trained only on flipped rows). Artefacts
  `scripts/artifacts/{flip_detector,gated_pipeline,gated_pipeline_v2}_results.json`.
- Diagnostic findings (5-fold OOF):
  - **Flips are highly feature-predictable**: binary LGBM on
    `is_flipped` → AUC **0.89932**. Top gain: `dgp_score` (27 M),
    then `Rainfall_mm`, `Temperature_C`, `Previous_Irrigation_mm`,
    `Humidity`, `Soil_pH`, `Sunlight_Hours`, `EC`,
    `Field_Area_hectare`, `Organic_Carbon` — each ~1 M. The
    "unused" features carry the noise model.
  - **Flip direction is near-deterministic**: 3-class LGBM trained
    only on the 10,304 flipped rows → OOF raw **0.99689**, bal
    **0.99368**. Confusion has 12 Low↔High and 20 High↔Low total,
    zero Medium confusions.
  - **LGBM+DGP is awful on flipped rows in isolation**: raw 0.15111,
    bal 0.12016 on the flipped subset (vs 0.99958 raw on clean
    rows). When trained on full 630k, the 1.6 %-minority signal is
    diluted out; LGBM+DGP only hides this via bias tuning.
- Execution (OOF tuned log-bias):
  - gated v1 (LGBM+DGP direction) → 0.97249 (baseline 0.97271,
    Δ = **−0.00022**, regression within noise).
  - gated v2 (specialist direction) → **0.86765** tuned,
    best hard-gate t=0.80 → 0.93931 bal, still below rule 0.96097.
- Why v2 fails: the specialist was trained only on rows where
  `rule != label`, so by construction it *always* disagrees with
  the rule. On the ~3 % of clean rows the detector false-flags at
  any reasonable threshold, the specialist predicts the wrong
  neighbour with 100 % confidence. Flip base rate (1.6 %) is too
  small to absorb the FP cost.
- Parked as ruled-out (for now): specialist-override gating. Works
  in principle — the flip structure IS learnable — but requires a
  direction model that is correct on BOTH clean and flipped rows.
- Next bet: sample-weighted LGBM on full 630k with weight ≈ 60 on
  flipped rows (equalise their total loss contribution with the
  clean majority). Or stacking: add P(flip) + specialist probs as
  columns to LGBM+DGP features. Either way, need a single model
  that is rule-accurate on clean rows AND specialist-accurate on
  flipped ones.

### 2026-04-20 — boundary LGBM (null) and kNN sanity (negative)

- Boundary LGBM (`scripts/boundary_lgbm.py`): train 3-class LGBM only
  on `dgp_score ∈ {1..9}` rows (596k), force Low for score=0 (33k).
  Score=0 covers 5.4 % of train and is 100 % Low without exception.
  Result: tuned OOF **0.97284** vs LGBM+DGP 0.97271 (Δ = +0.00013,
  within fold std 0.00088). The carve-out doesn't move the needle
  because LGBM was already handling score=0 rows trivially on the
  full data. On boundary rows specifically, bal_acc 0.96343 —
  nearly identical to LGBM+DGP's implicit boundary accuracy.
- kNN (`scripts/knn_six_features.py`): 5-fold k=50 KNeighbors on
  z-scored 4 continuous + binary mulch + Kc stage value. OOF argmax
  **0.94433**, tuned log-bias **0.95436**. Strictly below the rule
  (0.96097). kNN smooths across the rule's hard thresholds, losing
  crisp score→label mapping. Clean negative: the DGP is
  piecewise-constant with integer thresholds, and any smoother
  (kNN, MLPs with standard regularisation, kernel methods) is
  worse than trees that can place splits on the exact threshold
  values.

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
- **Ruled out**:
  - **Equal-weight z-score fusion of water-balance axes** (H2) is
    worse than the single-feature Soil_Moisture rule (H1). Any future
    hand-weighted score needs per-axis weights proportional to
    informativeness, not uniform.
  - **Blending MNLogit into LGBM** adds 0.00000 at any mixing weight.
    Linear model is too weak (0.78 vs 0.97) to contribute orthogonal
    signal; parked as possible stacking feature only.
  - **CatBoost as a standalone competitor** — fold-1 argmax 0.96000 ≈
    LGBM/XGB, 23 min/fold training cost, killed after fold 1. Could
    revisit as a 4th blend member only if compute budget allows late.
  - **Hand-engineered domain features inside LGBM** — 8 cols from F2
    / H3 pulled tuned OOF to 0.97045 vs baseline 0.97097
    (Δ = −0.00052, within 1σ fold noise of 0.00088). Trees at 127
    leaves already discover these interactions; prebuilt versions add
    no new splits. Revisit only at a much smaller leaf budget or on a
    tiny training subset.
- **Confirmed (new)**:
  - **Original Irrigation Prediction dataset is well-aligned with the
    synthetic DGP.** Transfer check: LGBM trained on 8k original,
    evaluated on 630k synthetic → tuned bal_acc 0.96278 (gap 0.00819
    vs 5-fold baseline). Categorical vocabularies match exactly;
    numeric distributions align within ~1 % except Rainfall_mm
    (~15 % lower mean in original); priors agree to 3 decimals.
    Concatenating 10k rows into training adds only +0.00027 though,
    because 10k ≪ 630k — the ceiling is bounded by data volume, not
    DGP mismatch.
- **Confirmed (new)**:
  - **The synthetic DGP is the original closed-form rule + boundary-
    band noise.** Closed form on 6 features (Soil_Moisture, Rainfall,
    Temperature, Wind, Mulching, Crop_Growth_Stage) perfectly fits the
    10k original dataset and hits raw 0.98364 / bal 0.96097 on the
    630k synthetic train. All 10,304 flips are strictly one-step
    (Low↔Medium or Medium↔High, never Low↔High) and sit in
    score-bands 1–9. Score 0 rows (5.4 % of train) are 100 % Low
    without exception.
  - **Flip noise is driven by the 10 "unused" features.** Binary
    is_flipped LGBM → OOF AUC 0.89932; top gain after `dgp_score`
    is `Rainfall_mm`, `Temperature_C`, `Previous_Irrigation_mm`,
    `Humidity`, `Soil_pH`, `Sunlight_Hours`, `EC`,
    `Field_Area_hectare`, `Organic_Carbon`. Signature of a tabular
    generative model (TabDDPM, VAE, or GAN) whose decision surface
    is fuzzy around sharp thresholds.
- **Ruled out (new)**:
  - **Specialist-override gating.** Direction model trained only on
    flipped rows is 99.37 % bal_acc on the flipped subset but 0 %
    on clean rows (by construction — always disagrees with rule).
    Gated blend regresses from 0.97271 → 0.86765; at best hard-gate
    t=0.80, 0.93931 — still below rule alone. FP cost from the
    flip detector at any usable threshold outweighs the TP gain
    given the 1.6 % flip base rate. Fix: direction model must be
    correct on BOTH clean and flipped rows.
  - **Boundary-only carve-out.** Restricting LGBM training to
    `dgp_score ∈ {1..9}` rows and forcing rule for score=0 gives
    0.97284 tuned OOF vs 0.97271 (Δ = +0.00013, within fold
    std). Score 0 rows are already trivially handled at full data;
    removing them changes nothing.
  - **kNN on the 6 DGP features.** k=50 KNeighbors OOF tuned-bias
    0.95436, below the rule 0.96097. Smoothing destroys the
    integer-threshold structure. Generalisation: any kernel / MLP
    / kNN / GP on the DGP feature set is bounded *below* the rule,
    because the rule's thresholds are exact and these methods
    interpolate across them.
- **Parked**:
  - Seed recovery / DGP archaeology on the synthetic generator — high
    effort, unclear payoff with only 10 days; revisit if stuck above
    0.9815.
  - Deep-dive into what functional form the synthetic generator
    uses to sample the noise. We know it's a function of 10
    "unused" features (AUC 0.90 detector) and the flip is always
    one step. If we could invert that function we'd get ~99 %+
    bal_acc. Inversion requires either (a) a direction model that
    is correct on both populations (open), or (b) training-data
    seed / architecture leak (unlikely).

## Playbook

The reusable Kaggle playbook lives at
<https://github.com/chris0leite-ui/kaggle-claude-code-setup> (branch
`claude/kaggle-playbook`). Kickoff steps, workflow norms, and
methodology are maintained there — update that repo when a transferable
lesson surfaces.
