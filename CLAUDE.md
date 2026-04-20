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
  field-capacity lookup); `scripts/heuristic.py` (no-training,
  threshold-fit-per-fold predictor); `scripts/formula_mnlogit.py` (three
  hand-crafted MNLogit formulas F1/F2/F3); `scripts/benchmark_multi.py`
  (XGBoost done, CatBoost killed at fold 1); `scripts/blend_lgbm_mnlogit.py`
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

### 2026-04-20 — domain knowledge pack + orthogonal-model 5-fold sweep (ruled out)

- Goal: codify the physical model of the target into a reusable
  knowledge base, and stress-test it by running a range of non-LGBM
  estimators under identical 5-fold CV. Two questions: (a) is the
  signal linear-separable? (b) does any weaker model bring orthogonal
  information worth stacking?
- Changed: `domain/` — 8-file modular primer (water balance, ET,
  soil, crops, irrigation systems, India context, modeling priors)
  adapted for this feature set. `scripts/cv_heuristic.py` —
  domain-weighted scalar score + per-fold 2-threshold tuning.
  `scripts/cv_linear_nb.py` — multinomial LR (class-balanced) +
  Gaussian NB on the same features. `scripts/cv_ebm.py` — EBM
  (InterpretML) with shape functions + pairwise interactions.
  Artefacts: `cv_heuristic.json`, `cv_lr_multinomial.json`,
  `cv_gaussian_nb.json`, `cv_big_fe.json`.
- Results (5-fold stratified OOF balanced accuracy, seed=42):
  - Heuristic (8-signal z-scored sum + learned 2 cuts, 630k):
    **0.60012 ± 0.00141** — per-class recall High 0.706 / Low 0.686
    / Medium 0.409.
  - Gaussian NB (independence, 630k): **0.75172 ± 0.00402**.
  - Multinomial LR (one-hot + z-score, class_weight=balanced, 630k):
    **0.83009 ± 0.00827**.
  - EBM (shape + pairwise interactions, 200k for compute,
    outer_bags=1): **0.96106 ± 0.00120**.
  - Baseline LGBM + tuned log-bias (reference): 0.97097.
- Observations:
  - Interaction gap is the story: heuristic 0.60 → NB 0.75 → LR 0.83 →
    EBM 0.96 → LGBM 0.97. The **independence assumption (NB) loses
    ~0.22 vs LGBM**, almost all of which is non-linearity + pairwise
    interactions. This makes stacking with any of these slower models
    a poor bet (same reason as MNLogit blend null).
  - Heuristic Medium recall collapses (0.41): the middle bin has no
    standalone signal, it lives in the interaction pattern.
  - Hand-engineered domain features (VPD, Kc stage, soil depletion,
    ET proxy) add 0.958 → 0.958 on EBM — consistent with the earlier
    LGBM-FE null result (boosted trees at this leaf count already
    find these patterns).
- LB delta: n/a.
- Next bet: DGP archaeology (now productive per the 2026-04-20 DGP
  entry below) is the remaining orthogonal lever. Skipping further
  orthogonal-model work; the signal is tree-shaped.

### 2026-04-20 — LGBM hyperparameter sweep (ruled out)

- Goal: answer NEXT_STEPS §N — does serious HP tuning on the baseline
  LGBM break the 0.97097 OOF plateau? Baseline uses num_leaves=127,
  min_data_in_leaf=200, lr=0.05, feature_fraction=0.9, bagging=0.9.
- Changed: `scripts/hyperopt_lgbm.py` — Optuna TPE with
  MedianPruner over `learning_rate, num_leaves, min_data_in_leaf,
  feature_fraction, bagging_fraction, bagging_freq, lambda_l1,
  lambda_l2, max_depth, min_gain_to_split`. Optimizes
  prior-reweight OOF (faster proxy for log-bias, captures >99 % of
  the lift on identical probs). `scripts/finalize_lgbm.py` — reruns
  best config on full 630k with log-bias coord ascent.
  `scripts/artifacts/hyperopt_lgbm_200k.json`.
- Setup: 60 trials attempted / 47 completed / 13 pruned, ~90 min
  budget, 200k stratified subsample (ranking stable vs 630k for
  LGBM HPs in this regime).
- Results (prior-reweight OOF bal_acc, 200k):
  - **Best trial (29)**: 0.97047 with `num_leaves=46, max_depth=3,
    lr=0.064, feature_fraction=0.64, bagging_fraction=0.76,
    bagging_freq=1, lambda_l1~7e-5, lambda_l2~4e-5, min_gain=0.24`.
  - Baseline's config on 200k prior-reweight (for apples-to-apples):
    not recomputed, but 630k log-bias baseline is 0.97097.
  - Full 630k finalize was started, killed early — extrapolated
    delta ≤ +0.001, not worth the ~30 min compute.
- Observations:
  - TPE preferred **shallow** trees (max_depth 3–4, num_leaves
    46–189) vs. the baseline's 127 leaves with default max_depth.
    Shallow + regularized is a different regime that reaches roughly
    the same OOF — a plateau, not a ridge.
  - Best config switched 4×+ during the sweep (trial 1 → 17 → 27 →
    29) with Δ ~+0.001 between each. TPE was still exploring when
    budget expired, but gains flattened — typical saturation
    pattern.
- LB delta: n/a.
- Next bet: shift compute to (a) ensembling LGBM+XGBoost at the
  prob level with shared log-bias tuning, (b) the DGP distance-
  to-threshold features flagged in the DGP-reverse-engineering
  entry above — that's where the remaining 0.01–0.015 to the pack
  most plausibly lives.

### 2026-04-20 — LGBM+DGP, boundary model, gated pipelines, flip detector

- Goal: operationalize the reverse-engineered DGP rule inside LGBM and
  test whether the 10,304 boundary-band flips (1.64 % of rows) can be
  recovered — the only quantified remaining lever.
- Changed: `scripts/benchmark_dgp.py` (LGBM + 15 DGP-derived cols:
  `dgp_dry/norain/hot/windy/nomulch/kc/score` plus
  signed + absolute distances to the 4 thresholds);
  `scripts/boundary_lgbm.py` (separate model on boundary-band rows);
  `scripts/gated_pipeline.py` (soft-blend rule + flip-prob + LGBM-on-
  all-rows direction model); `scripts/gated_pipeline_v2.py` (same blend
  but specialist trained on flipped rows only); `scripts/flip_detector.py`
  (diagnostic). Artefacts: `scripts/artifacts/{bench_dgp,boundary_lgbm,
  gated_pipeline,gated_pipeline_v2,flip_detector}_results.json` and
  accompanying `.npy` OOF/test probs.
- Results (OOF bal_acc, 5-fold stratified, seed=42):
  - LGBM baseline tuned (reference) → 0.97097
  - **LGBM+DGP tuned → 0.97271** (Δ = **+0.00174**, ~2σ, real)
  - **Boundary LGBM tuned → 0.97284** (ties LGBM+DGP within 1σ)
  - Gated v1 (rule + LGBM-on-all-rows, soft-blend) tuned → 0.97249
    (no lift: both sides of the soft-average already agree on clean rows)
  - Gated v2 (rule + flipped-only specialist, soft-blend) tuned → 0.86765
    (**broken**: specialist is OOD on clean rows, raw acc 0.000)
- Flip-detector diagnostic (`scripts/flip_detector.py`):
  - **Binary flip detector OOF AUC = 0.8993** on "is this row flipped?"
    `dgp_score` dominates feature importance (5× runner-up).
  - **Flip-direction on flipped-only rows: 99.37 % bal_acc** — given a
    row is flipped, we know the correct class essentially perfectly.
- Read-out: substantial residual signal (AUC 0.9 flip detection) exists,
  but neither of the two blending schemes captures it. v1 is too soft
  (main model already approximates the rule, so blend == rule).
  v2 is too hard (specialist hasn't seen clean rows, so any positive
  P_flip on a clean row leaks garbage into the blend). Correct fix
  is either (a) a **learnable meta-model** on top of [rule, P_flip,
  P_spec, P_main], or (b) **hard-gate** — rule by default, specialist
  only when P_flip > τ. Both need exploration.
- LB delta: still 2/10 spent today.
- **New current best: LGBM+DGP tuned at 0.97271** (boundary_lgbm ties).
  Beats the previous logged best (LGBM+EXT 0.97124) by +0.0015.
- Next bet: `scripts/gated_v3.py` — build stacking + hard-gate on the
  already-saved OOF arrays (no retraining), tune log-bias, emit
  submissions. If meta-LGBM over OOF components breaks 0.975, we're
  finally above the logged-best plateau by a margin worth an LB probe.

### 2026-04-20 — gated_v3 (meta-stack + hard-gate) — null result

- Goal: deploy the AUC-0.9 flip detector + 99.4%-direction specialist
  via a learnable gate instead of the broken hand-coded blends.
- Changed: `scripts/gated_v3.py` — runs on saved OOFs (no retraining),
  evaluates 4 decision rules, tunes log-bias on each. Artefacts:
  `scripts/artifacts/{oof,test}_meta_v3.npy`, `gated_v3_results.json`,
  `submissions/submission_gated_v3.csv`.
- Results (OOF balanced accuracy, 5-fold, seed=42):
  - Rule-only → 0.96097
  - LGBM+DGP tuned (reference) → **0.97271**
  - Hard-gate best `τ=0.95` → 0.95893 (worse than rule)
  - Soft(rule + main) tuned → 0.97249 (ties reference)
  - Meta-LGBM over `[P_main(3), P_spec(3), P_flip(1), rule_oh(3),
    rule_int(1)]`, 5-fold stacking → **0.97245** (ties reference)
- Read-out: **LGBM+DGP is the ceiling from this architectural family.**
  The "99.4% bal_acc on flipped rows" headline is degenerate — on the
  flipped subset, the true label is by definition anti-rule, so a
  specialist just learns "predict ¬rule". When `P_flip > τ` is used
  to route rows to that specialist, the selection set is polluted with
  false positives (clean rows near boundaries), and on those the
  specialist systematically predicts the opposite of the true label.
  Meta-LGBM saw this and collapsed to passing through P_main.
- Implication: the DGP-aware feature set (`dgp_score`, signed
  distance-to-threshold) has already fully internalized the learnable
  part of the flip signal. The remaining ~0.01 gap to the 0.98114
  pack does **not** live in boundary-band flip recovery. Pivot to:
  (a) seed-bag LGBM+DGP (+~0.001 cheap insurance), (b) XGBoost with
  DGP features + blend, or (c) an MLP — the only untried model family,
  and arguably the one that best matches how the synthetic labels were
  generated (`brief.md:74` confirms a DL model was used).
- LB delta: n/a.

## Hypothesis board

- **Open**:
  - Seed-bag LGBM+DGP (3–5 seeds, retune bias on averaged OOF).
    Cheapest remaining win; expected +0.0005–0.001 from fold-std
    reduction.
  - LGBM+DGP + XGBoost-with-DGP-features blend at prob level.
    Model-diversity lift, expected +0.001–0.002.
  - Neural-net tabular model (MLP / TabNet). No NN code in the repo.
    `brief.md:74` states the synthetic labels were generated by a DL
    model; an MLP might recover the near-threshold noise band in a
    way axis-aligned trees structurally can't. Unknown upside,
    non-trivial setup.
  - Ordinal-aware loss to cut Medium↔High confusion is still untested.
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
  - **Orthogonal-model stacking candidates** (heuristic / Gaussian
    NB / multinomial LR / EBM) — 5-fold OOF ladder on the same
    folds: heuristic 0.600, NB 0.752, LR 0.830, EBM 0.961. LGBM is
    0.97097. The independence-to-interaction gap (NB 0.75 → LGBM
    0.97) is ~0.22, so no weaker linear/independence-based model
    brings enough orthogonal signal to justify stacking. EBM is
    close to LGBM but diversity value is bounded by the 0.01 gap.
    Rule: any future stacking candidate must hit ≥0.965 standalone
    OOF to be worth the compute.
  - **LGBM hyperparameter optimization** (Optuna TPE, 47 trials,
    200k subsample, 10-dim search space). Best
    `num_leaves=46, max_depth=3, lr=0.064` hit 0.97047
    prior-reweight on 200k — roughly level with the 0.97097 baseline
    (which uses num_leaves=127, defaults elsewhere). The sweep found
    a different shape of optimum (shallow + regularized) that reaches
    the same plateau. Extrapolated full-630k delta ≤ +0.001.
    Baseline HPs are near-optimal for this feature set; further
    gains need a different lever.
  - **Hand-coded soft-blend of rule + flip-prob + specialist**
    (`scripts/gated_pipeline*.py`). v1 (specialist trained on all rows)
    ties LGBM+DGP at 0.97271 — no lift because the two sides of the
    blend already agree. v2 (specialist trained on flipped rows only)
    collapses to 0.86765 because the specialist predicts anti-rule
    on clean rows where P_flip > 0.
  - **Gated flip-recovery as a lever** (`scripts/gated_v3.py`). Tried
    meta-LGBM stacking over `[P_main, P_spec, P_flip, rule_oh,
    rule_int]` and hard-gate `argmax(P_spec) if P_flip>τ else rule`.
    Hard-gate best τ=0.95 → 0.95893 (worse than rule). Meta-LGBM
    tuned → 0.97245 (ties LGBM+DGP). The flip-direction specialist's
    "99.4% bal_acc on flipped rows" is degenerate — on that subset,
    true label = anti-rule by construction. Deployed at any τ, the
    selection set contains enough false positives (clean rows near
    boundaries) that the specialist's anti-rule prediction becomes
    systematically wrong on them. **The DGP-aware LGBM has already
    internalized all learnable flip signal.** No lever here.
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
  - **DGP features (score + distance-to-threshold) *do* help LGBM.**
    `scripts/benchmark_dgp.py` with 15 DGP-derived cols moves tuned
    OOF from 0.97097 → 0.97271 (Δ = +0.00174, ~2σ, every fold
    improves). Earlier FE null was the wrong features — raw
    water-balance terms. The right features are the ones the
    generator actually uses: binary indicators, score, and signed
    distances to each threshold (`Soil_Moisture − 25`, etc.).
    **New current best.** Boundary-LGBM (`scripts/boundary_lgbm.py`)
    ties it at 0.97284 within 1σ.
  - **Boundary-band flips are feature-predictable.**
    `scripts/flip_detector.py` trains a binary "is_flipped" LGBM and
    hits OOF AUC = 0.8993, with `dgp_score` dominating gain. A
    3-class classifier restricted to the 10,304 flipped rows reaches
    99.37 % bal_acc on flipped rows. The residual signal is real and
    learnable; the open question is how to deploy it in the
    prediction pipeline without breaking clean-row predictions.
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
