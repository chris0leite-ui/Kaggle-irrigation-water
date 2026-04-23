# CLAUDE.md

Guidance for Claude Code when working in this repository.

## ⚠️ LB SUBMISSION RULE — ALWAYS ASK FIRST

**Never upload a submission CSV to Kaggle without explicit user
confirmation for that specific submission.** Building candidate
CSVs locally and reporting their OOF scores is fine; running
`kaggle competitions submit` (or equivalent) is not — it burns
from the 10/day budget and once final-selected, from the 2 final
slots. Always present the candidate + its OOF score + the
expected LB outcome to the user and wait for a go-ahead before
submitting. This rule applies even when a blend's OOF beats the
current best — the LB is an adversarial split and OOF-to-LB
calibration can drift.

## ⚠️ NEVER SUGGEST PUBLIC-CSV / OTHER-PEOPLE'S-SUBMISSIONS BLENDING

**Do not propose, scaffold, or recommend blending other people's
submission CSVs** (pulling high-scoring public-notebook submissions
as Kaggle Dataset inputs and ensembling them, hard-vote / weighted
blend over rival CSVs, pseudo-labeling from someone else's
submission, etc.) as a path to a higher LB. **Do not frame it as
"the pack's mechanism" or "the only realistic path to 0.98+".**
People have reached 0.98+ without blending others' results; our
job is to find the own-pipeline lever they found, not to mirror
the public-notebook blend trick. If the user explicitly asks for
public-CSV blending, point them back to this rule and ask for
confirmation before proceeding.

Any session notes, reports, or hypothesis-board entries that
previously listed public-CSV blending as a "strategic option"
should be treated as stale guidance and ignored — the updated
rule is no-suggest.

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

### 2026-04-20 — 128-cell empirical Bayes + LGBM-dist replication (stacking null + replication)

- Goal: test ideas #1 (128-cell empirical Bayes) and #2 (LGBM with
  signed/abs distance-to-threshold features) from the brainstorm,
  and check whether a blend of the two beats either component.
- Changed: `scripts/empirical_bayes_cell.py` (pack 6 rule features
  into a single cell-id in `[0,128)`, estimate `P(y | cell)` with
  Laplace smoothing out-of-fold, save OOF+test probs +
  `submission_eb_cell_tuned.csv`). `scripts/benchmark_dist.py`
  (LGBM with 43 features: original 19 + DGP score/bool indicators
  + 4 signed distances + 4 absolute distances + score-band
  distances + min-axis distance + 4 pairwise interactions).
  `scripts/blend_eb_dist.py` sweeps α ∈ [0,1] in prob and log
  space. Artefacts: `oof_eb_cell.npy`, `test_eb_cell.npy`,
  `oof_lgbm_dist.npy`, `test_lgbm_dist.npy`,
  `eb_cell_results.json`, `bench_dist_results.json`,
  `blend_eb_dist_results.json`.
- Results (5-fold stratified OOF bal_acc, seed=42, 630k):
  - Rule argmax (pure DGP formula): 0.96097 (prior).
  - **EB-cell tuned log-bias**: **0.96339** — the Bayes-optimal
    ceiling given only the 6 rule features.
  - **LGBM+dist tuned log-bias**: **0.97266** — matches the prior
    `benchmark_dgp.py` result (0.97271) within fold noise
    (σ ≈ 0.00088). Confirms the +0.00174 DGP-aware lift is
    reproducible; `benchmark_dist.py` is a feature superset of
    `benchmark_dgp.py` with the same effective performance.
  - EB-cell + LGBM-dist prob blend: monotonic in α → pure LGBM
    (α=1.0) wins. EB brings zero orthogonal signal.
- Observation: the 128-cell cube uses the same 6 features the
  LGBM already splits on near-optimally — the model has no trouble
  recovering per-cell class distributions from interaction splits.
- Read-out: the ~0.008 gap between EB-cell (0.96339) and LGBM-dist
  (0.97266) is the **information in the 13 non-rule features**
  (Soil_pH, Humidity, Sunlight_Hours, Organic_Carbon, EC, Field_Area,
  Previous_Irrigation, Region, Crop_Type, Soil_Type, plus Mulching
  and Stage already in the rule). Any future "noise-model" approach
  has to either capture those features or beat LGBM at using the
  distance-to-threshold signal, not just restate the rule.

### 2026-04-21 — DGP is a learnable NN function, not a noise process

- Goal: answer whether the ~10k "flipped" rows are a Bernoulli-style
  noise process layered on the rule (as we'd been modeling), or a
  deterministic output of the host's label-generating NN
  (`brief.md:74`) applied to features the rule ignores.
- Changed: `scripts/eda_dgp_residuals.py` (self-contained HTML EDA at
  `plots/eda/dgp_residuals.html`) plus an in-notebook statistical
  test on score=3 rows. No new model code yet.
- Findings:
  - **Zero exact feature-vector duplicates in 630k rows.** Consistent
    with a continuous-feature generator (VAE/diffusion), not with a
    rule + Bernoulli-flip process.
  - **Non-rule features are significantly different between flipped
    and non-flipped rows.** At score=3 (4,899 flips / 102k rows):
    - `Previous_Irrigation_mm`: d = +0.107 (mean 64.9 vs 61.3, p=5e-14)
    - `Humidity`:               d = +0.076 (mean 62.0 vs 60.6, p=8e-8)
    - `Electrical_Conductivity`: d = +0.037 (p=0.011)
    - `Field_Area_hectare`:     d = +0.035 (p=0.019)
    - `Soil_pH`, `Organic_Carbon`, `Sunlight_Hours`: ~0, n.s.
    Effect sizes are small, but the sample size (100k) makes them
    crushingly significant. Direction is agronomically sensible
    (higher humidity + more recent irrigation → label bumps from
    Low to Medium).
  - **Per-cell majority predictor gives raw 0.98384 / bal 0.95983.**
    Only 1 of 64 rule-cells has a synthetic majority different from
    the rule (covering 308 rows, 0.05%). So the "noise" isn't
    cell-level flipping — it's within-cell variation driven by
    continuous position and non-rule features.
  - **LGBM+DGP error geometry confirms**: errors have median
    |distance-to-threshold| 0.79–0.87 of correct rows on moist / rain
    / temp, but 1.03 on wind → wind distance is uninformative of
    errors. 81 % of LGBM errors sit at scores 3 (4,849) and 6 (3,541)
    — the two class-boundary scores.
  - **LGBM+DGP recovers only 19 % of rule flips (1,969 / 10,304) and
    introduces 3,151 new errors** on rule-correct rows. Net: LGBM
    tuned has *more* total errors (11,486) than the rule alone
    (10,304). It only wins on bal_acc because bias tuning redistributes
    errors toward the Medium class to lift High recall.
- Read-out: the DGP is a **deterministic function** (the host's NN),
  not rule + IID noise. Properties:
  1. Flip recovery has no irreducible-noise floor — theoretical
     ceiling is 100 %.
  2. The NN's decision boundary is a smooth curved manifold in the
     full feature space. Axis-aligned trees are structurally
     handicapped; they need many splits to approximate a curve each
     NN neuron represents.
  3. Non-rule features (`Previous_Irrigation_mm`, `Humidity`, etc.)
     carry deterministic signal the NN learned from the original
     10k. LGBM has them as inputs but hasn't fully integrated them
     into boundary-level decisions (effect size 0.1 gets washed out
     by tree regularisation).
  4. The 0.98114 pack is almost certainly **reproducing the NN's
     decisions** (with FE or a DL model), not denoising a stochastic
     process.
- Implication for strategy: reframe from "how do I denoise labels?"
  to "how do I approximate the label-generating NN?" Two consequences:
  1. **MLP / tabular NN is now the top bet**, not parked. Structural
     match to the DGP.
  2. **Pairwise FE of rule × non-rule features** (Humidity × Soil_Moisture,
     Previous_Irrigation × Rainfall_mm, Field_Area × score, etc.) may
     let LGBM recover the NN-learned correlations more cleanly.

### 2026-04-21 — balanced-ensemble methods (ruled out)

- Goal: test whether per-base-learner majority undersampling
  (BalancedRandomForest, EasyEnsemble, RUSBoost from `imbalanced-learn`)
  beats LGBM+DGP's 0.97271, or contributes orthogonal signal in a blend.
  Motivated by the multi-class-imbalance research report flagging
  "rebalance at training time" as the last unexplored data-level lever.
- Changed: `scripts/benchmark_balanced_ensembles.py` (now deleted —
  null result). Same 5-fold stratified split, same 34-col DGP-enriched
  feature set, same coord-ascent log-bias decision rule.
- Configs chosen to avoid known failure modes:
  - BRF 400 trees, `sampling_strategy='all'`, `replacement=True`,
    `min_samples_leaf=50`.
  - EasyEnsemble 10 outer × inner AdaBoost(`DecisionTreeClassifier(max_depth=5)`,
    40 iter, lr=0.3). Default stump-based inner collapses on 3-class.
  - RUSBoost 200 iter, `DecisionTreeClassifier(max_depth=5)`, lr=0.3.
    Default stumps produce SAMME bal_acc=0.333.
- Results (OOF bal_acc, 5-fold, seed=42, tuned log-bias):
  - LGBM+DGP (ref)       0.97271
  - EasyEnsemble         0.96932  (Δ = −0.00339)
  - RUSBoost             0.96666  (Δ = −0.00605)
  - BalancedRF           0.96535  (Δ = −0.00736)
  - LGBM × Easy linear   0.97279 at w=0.80 (Δ = +0.00008)
  - LGBM × Easy geo      0.97278 at w=0.70 (Δ = +0.00007)
  - LGBM × BRF / RUS     collapse to pure LGBM or +0.00001
  - 3-way LGBM+Easy+BRF  0.97279 at (0.8, 0.2, 0) — collapses to
    pairwise, BRF gets zero weight.
- Observations:
  - Balanced-ensemble probs are already nearly class-balanced out of the
    box (inter-class bias deltas 0.03–0.14), so coord-ascent log-bias
    has almost nothing to correct — argmax and tuned are within
    0.0007–0.002 of each other. LGBM's sharper imbalanced probs
    respond much better to log-bias tuning (+0.0092 from tuning).
  - EasyEnsemble trades Medium recall for High recall (97.0% High)
    vs LGBM+DGP's profile, but the High-recall bump does not survive
    blending — log-bias on LGBM already finds the same operating point
    on macro-recall.
  - BRF is strictly dominated in every blend config.
- Read-out: **per-tree/per-base-learner majority undersampling is not
  a distinct lever from post-hoc log-bias on this feature set.** Both
  are mechanisms for picking a balanced-accuracy-optimal operating
  point on a fixed model. LGBM+DGP + log-bias already occupies it.
  The broader lesson matches the 2026-04-21 DGP finding: the ceiling
  isn't a calibration problem, it's a **model-class** problem. Axis-
  aligned trees — rebalanced or not — bottleneck on the same smooth
  NN decision boundary.
- Budget impact: zero LB submissions spent. Still 2/10 used for the
  day (both from 2026-04-20).
- Next bet: unchanged — MLP / tabular NN with balanced softmax or
  LDAM loss remains the top open hypothesis.

### 2026-04-21 — per-score experts + noise-inversion head + GCE loss (three nulls)

- Goal: test brainstorm ideas #3 (noise-inversion head), #5 (GCE
  noise-robust loss), #8 (per-score expert models) to see whether
  any of the unexplored orthogonal levers beats LGBM-dist (0.97266).
- Changed: `scripts/score_experts.py` (one LGBM per score bin,
  dropping rule cols, routing val rows by their score);
  `scripts/noise_inversion.py` (three per-rule-label LGBM heads,
  rule cols removed so each head specialises on P(y_obs | rule, x));
  `scripts/lgbm_gce.py` (custom multiclass GCE objective q=0.7,
  class-major grad/hess, same feature set as benchmark_dist).
- Results (OOF bal_acc, 5-fold stratified, seed=42):
  - **Per-score experts (#8)**: 0.97149 tuned (−0.00117 vs LGBM-dist).
    Splitting 630k into 10 score bins (~80–120k each) loses more to
    per-expert data shortage than specialisation recovers.
  - **Noise-inversion head (#3)**: 0.96768 tuned (−0.00498 vs
    LGBM-dist). Dropping rule cols removed distance information; the
    rule=High head (21 k rows) is especially starved.
  - **LGBM + GCE q=0.7 (#5)**: 0.96500 tuned — buggy. best_iter=1 on
    every fold (training stalls after round 1); log-bias then
    rescues argmax from flat prior-dominated probs. Grad/hess scaling
    of the custom objective is almost certainly off; parked pending
    a proper debug.
- Observation: split-and-ensemble approaches don't add orthogonal
  signal over a single 630k-row LGBM that already has the
  score/distance features — trees find the same per-score partitions
  for free at no data cost.

### 2026-04-21 — LGBM-dist seed-bag (small positive)

- Goal: cheap variance reduction on top of LGBM-dist — 5 seeds,
  averaged OOF, retune log-bias on the mean. Target: +0.0005–0.001.
- Changed: `scripts/seed_bag_dist.py` — same 5-fold split, same
  43-feature LGBM-dist config, seeds `[42, 7, 123, 2024, 9999]`.
  Artefacts: `oof_lgbm_dist_bag.npy`, `test_lgbm_dist_bag.npy`,
  `seed_bag_dist_results.json`,
  `submission_lgbm_dist_bag_tuned.csv`.
- Results (OOF tuned bal_acc):
  - Per-seed range 0.97255 → 0.97274 (spread 0.00019).
  - **5-seed bag**: **0.97289** — beats every individual seed
    (clean 5/5 one-sided win, small but real).
  - Δ vs single-seed baseline = +0.00024.
- Read-out: LGBM at `num_leaves=127, bagging_fraction=0.9` on 630k
  rows is nearly deterministic across seeds, so bagging variance
  reduction has little room. The gain is real but bounded. New best
  candidate on disk at this point: OOF 0.97289.

### 2026-04-21 — XGBoost-dist + LGBM-bag blend (CURRENT BEST, LB 0.97170)

- Goal: real model-family diversity on the 43-feature LGBM-dist set
  — LGBM leaf-wise vs XGBoost level-wise hist — to break past the
  0.97289 bag plateau.
- Changed: `scripts/benchmark_xgb_dist.py` (XGBoost multi:softprob,
  `max_depth=7, min_child_weight=5, subsample=0.9,
  colsample_bytree=0.9, tree_method=hist, enable_categorical=True`,
  early_stopping_rounds=100) and `scripts/blend_lgbm_xgb_dist.py`
  (α ∈ [0,1] sweep in prob and log space, log-bias tuned per blend).
  Artefacts: `oof_xgb_dist.npy`, `test_xgb_dist.npy`,
  `xgb_dist_results.json`, `blend_lgbm_xgb_dist_results.json`,
  `submission_xgb_dist_tuned.csv`, `submission_blend_lgbm_xgb_dist.csv`.
- Results (OOF tuned bal_acc, 5-fold stratified, seed=42):
  - **XGBoost-dist standalone**: **0.97304** (+0.00038 vs single
    LGBM-dist 0.97266, +0.00015 vs 5-seed LGBM-dist bag 0.97289).
  - Prob-blend α sweep: best ≈ 0.50–0.65 → 0.97322, monotone-up to
    middle then monotone-down past it — signal is real, not a
    single-point fluke.
  - **Log-blend α=0.45 (LGBM 0.45 / XGB 0.55) → 0.97327 tuned** —
    **new current best**, beats both standalones at every interior
    α in both spaces.
  - Lift ladder vs baseline 0.97097:
      single LGBM-dist        0.97266  (+0.00169)
      LGBM-dist 5-seed bag    0.97289  (+0.00192)
      XGBoost-dist standalone 0.97304  (+0.00207)
      **LGBM-bag ⊗ XGB blend 0.97327  (+0.00230)**
- Read-out: real model-family diversity is worth ~1.5× as much as
  seed bagging on this problem — first experiment on this lineage
  that moves OOF cleanly via orthogonal signal rather than variance
  reduction.
- LB delta: submitted `submission_blend_lgbm_xgb_dist.csv` →
  **LB public = 0.97170** (**new LB best**). Δ vs LGBM+DGP's LB =
  +0.00033. Δ vs baseline LGBM's LB = +0.00198.
- Calibration ladder (OOF → LB gap widens with OOF):
    single tuned LGBM       0.97097 → 0.96972  gap 0.00125
    LGBM+DGP                0.97271 → 0.97137  gap 0.00134
    **bag + XGB blend       0.97327 → 0.97170  gap 0.00157**
  Gap grew +0.00032 across the ladder — modest OOF selection
  overfit (log-bias coord ascent + α sweep + model picking) but
  still below 1σ fold std (0.00088). Treat OOF above 0.972 as a
  proxy with ~0.0015 discount to predicted LB.
- LB budget: 3 submissions spent cumulatively on this lineage
  (baseline, LGBM+DGP, blend) + 1 DGP-rule probe on main (0.95835)
  = 4 total. 6 LB submissions remaining today.
- Next bet: (a) seed-bag XGB too, blend 2 bags; (b) CatBoost or
  ExtraTrees as a 3rd leg — model-family diversity compounding;
  (c) stack the blend's OOF probs as meta-features into a final
  LGBM meta-model; (d) rule × non-rule pairwise FE applied to
  both LGBM-dist AND XGB-dist, then re-run the bag + blend.

### 2026-04-21 — hinge-loss / max-margin lever ruled out

- Goal: follow up on community discussion
  [692754](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/692754)
  by @broccoli-beef. The post shows the 10k original is linearly
  separable in a 9-binary-feature space (`Soil<25, Temp>30, Rain<300,
  Wind>10, Mulching=Yes, Crop=Flowering/Harvest/Sowing/Vegetative`),
  enumerates every integer linear model `|w|≤10, 1≤θ≤10` that
  separates it, and observes each model has a different hinge loss.
  Conjecture (ours): under the classical max-margin / VC-bound argument,
  the lowest-hinge-loss solution should transfer best to the 630k
  synthetic — i.e. hinge loss is a free tie-breaker picking the model
  closest to the host's NN decision surface.
- Changed: `scripts/enumerate_integer_models.py` reproduces the
  discussion's OR-Tools CP search, computes multiclass hinge loss per
  solution on the 10k, scores **every separating model** on 630k
  synthetic, saves per-model predictions + ranked table to
  `scripts/artifacts/integer_separating_models.csv`,
  `integer_models_summary.json`, and `integer_models_topk_*.npy`.
  One-liner: `Soil<26` in the discussion's display column is just a
  label — the actual separating inequality is `Soil_Moisture < 25`
  (a threshold sweep confirms `<25` gives exact 100 %, `<25.5` gives
  99.5 %, `<26` gives 99.0 %).
- Results:
  - **CP emits exactly 743 distinct integer models, all with
    train_acc_orig = 1.00000**, reproducing the discussion's count.
  - Hinge loss on 10k: range **0.0000** (many tied SVM-style max-margin
    solutions) to **0.2981** (the compact cdeotte-style solution:
    `w=[2,1,2,1,-1,0,-2,-2,0], θ=3`).
  - **All 743 models produce IDENTICAL predictions on the 630k
    synthetic** — agreement rate across top-50 = 1.0000, bal_acc_syn
    = 0.96097 and raw_acc_syn = 0.98364 to 5 decimals for every
    solution. Spearman(hinge, bal_acc_syn) is undefined (zero variance
    on bal_acc). The max-margin argument collapses because every
    synthetic row maps to one of the 128 unique discrete cells
    (`2^5 × 4`), every cell's label is unambiguous in the 10k, and
    every separating linear classifier is forced to agree on the
    cell-labeling. Wider margin (scaling `(w,θ) → (2w, 2θ)`) does
    not move any cell across the boundary.
  - Cdeotte's rule is structurally identical to our DGP rule; the
    LinearSVC posted in the discussion is just a `2×` scale of it.
- Implications:
  1. **Hinge loss is NOT a useful tie-breaker in this competition.**
     The ceiling for any 9-binary-feature linear classifier is
     0.96097, set by the cell-labeling, not by margin choice. Any
     "pick the best rule" approach plateaus here.
  2. The remaining 2.6 % residual (10,304 flipped rows) lives
     **entirely within the 128 cells**, as already flagged by the
     2026-04-21 per-cell-majority analysis (raw 0.98384 / bal 0.95983,
     only 1/64 cells has a cell-majority flip). The flip signal is
     within-cell variation driven by continuous non-rule features
     (`Humidity`, `Previous_Irrigation_mm`, etc.), confirmed earlier.
  3. The 0.98114 LB pack's edge is therefore in **within-cell
     resolution** (model capacity on the continuous features) — not in
     rule/weight choice, not in margin, not in ensembling over
     separating solutions. Consistent with the MLP-plateau commit on
     main (e889f0c): a 50 k-param MLP can't match LGBM+DGP on this
     rule-structured feature set, and rule-level ensembling (this
     work) adds exactly zero orthogonal signal. **Pairwise rule ×
     non-rule FE remains the top open bet.**
- LB delta: n/a (0 LB spend this session; 2/10 total, 8 left today).
- Next bet: the within-cell angle hasn't been exhausted. Two
  adjacent experiments are cheap and still live:
  1. **Rule × non-rule pairwise FE** (already the top bet from main's
     e889f0c) — the CP enumeration confirms it's the right target.
  2. **Within-cell MLP / per-cell logistic** on `Humidity,
     Previous_Irrigation_mm, Electrical_Conductivity, Field_Area,
     Soil_pH, Organic_Carbon, Sunlight_Hours` restricted to the rows
     of each of the 128 cells. By construction orthogonal to any
     rule-level ensemble, and targets exactly the 10,304 within-cell
     flips.

### 2026-04-21 — rule × non-rule pairwise FE (null result)

- Goal: execute the top-ranked Open bet from the hypothesis board —
  add 8 pairwise products targeting the non-rule features that showed
  significant Cohen's d on flipped rows (2026-04-21 DGP-residuals
  EDA), re-train LGBM-dist bag + XGB-dist, re-run the blend.
- Hypothesis: the flip-band residuals are a smooth NN function of
  `(Previous_Irrigation × Rainfall, Humidity × Soil_Moisture,
  Humidity × Temperature, EC × Soil_Moisture, Field_Area × score)`.
  Giving trees explicit products of rule × non-rule pairs should
  replace many weak splits with a single strong one and let the
  model trace the smooth decision surface.
- Changed: `scripts/seed_bag_dist_fe.py` (LGBM 5-seed bag on 51
  features = 43 dist + 8 pairwise), `scripts/benchmark_xgb_dist_fe.py`
  (XGB on same 51 features), `scripts/blend_lgbm_xgb_dist_fe.py`
  (α sweep + log-bias tuning). 8 new cols: `humidity_x_sm`,
  `humidity_x_sm_dist`, `prev_irrig_x_rf`, `prev_irrig_x_rf_dist`,
  `prev_irrig_minus_rf`, `vpd_proxy` (= `Temperature_C *
  (100 − Humidity)/100`), `ec_x_sm`, `field_area_x_score`.
- Results (OOF bal_acc, 5-fold stratified, seed=42):
  - LGBM-dist bag (no FE, reference)    0.97289
  - **LGBM-dist-FE bag**                 **0.97270**   Δ = −0.00019
  - XGB-dist (no FE, reference)         0.97304
  - **XGB-dist-FE**                      **0.97313**   Δ = +0.00009
  - Non-FE blend (current best)          0.97327
  - **FE blend (log-α=0.05)**            **0.97320**   Δ = **−0.00007**
  - Prob-blend sweep: best α=0.05 → 0.97317. Log-blend best
    α=0.05 → 0.97320. Both pick essentially pure XGB-FE (95 %),
    because LGBM-FE's signal is redundant. Monotonically decreasing
    from α=0.05 through α=1.0.
- Read-out: the pairwise FE changes nothing at the ensemble level.
  All three deltas (LGBM, XGB, blend) sit well inside the ~0.00088
  fold-std noise band. This is the third tree-FE null in a row:
    - 2026-04-20 LGBM+FE (8 water-balance cols):  Δ = −0.00052
    - 2026-04-21 128-cell empirical Bayes blend:  Δ = 0
    - 2026-04-21 rule × non-rule pairwise FE:     Δ = −0.00007
  Trees at 127 leaves (LGBM) / max_depth=7 (XGB) already discover
  these interactions as splits; prebuilt products add no new
  information. Crucially, the optimal blend weight SHIFTED from
  α=0.45 (LGBM-bag / XGB balanced) without FE to α=0.05 (nearly
  pure XGB) with FE — the added LGBM features didn't just fail to
  help LGBM, they also broke LGBM's complementarity with XGB.
- Implication: "trees can't see the interaction" is definitively
  the wrong diagnosis for the ~0.01 gap to the 0.98114 pack. The
  pack's edge lives either (a) in a fundamentally different model
  class (a NN that matches the host's label generator structure),
  or (b) in within-cell continuous-feature modelling that avoids
  axis-aligned splits entirely (per-cell logistic/MLP).
- LB delta: n/a (0 LB spend; 6 remaining today). Candidate
  submission on disk (`submission_blend_lgbm_xgb_dist_fe.csv`) is
  strictly worse than the current LB-0.97170 submission, so no LB
  probe is warranted.
- New current best: unchanged — **LGBM-dist 5-seed bag × XGB-dist
  blend OOF 0.97327 / LB 0.97170**. Submission:
  `submissions/submission_blend_lgbm_xgb_dist.csv`.
- Next bet: within-cell per-cell logistic / MLP (Open #5 → now #1),
  which targets the only remaining architecturally distinct lever.
  The 8 non-rule continuous features are the only way the flip
  signal can enter the model; tree-shaped models plateau regardless
  of how they encode the interactions.

### 2026-04-21 — score-routing + spec-{6,7,8} hybrid: NEW LB BEST 0.97224

- Goal: test whether routing rows where the rule is near-100 %
  accurate to the rule (at predict time), and training XGB only on
  ambiguous rows, produces a better balanced-accuracy-optimal
  pipeline than training on all 630k.
- Motivation: the rule-error-rate-per-score table (computed this
  session) shows the flip mass is concentrated in a narrow band:
  ```
  score  rows     rule errors  err%      (rule predicts)
  0      33,767   0           0.000%    Low
  1      115,457  5           0.004%    Low
  2      122,220  365         0.299%    Low
  3      102,157  4,899       4.80%     Low (boundary)
  4      117,837  1,520       1.29%     Medium
  5      79,203   274         0.35%     Medium
  6      38,416   1,549       4.03%     Medium (boundary)
  7      15,026   1,360       9.05%     High
  8      2,680    330         12.31%    High
  9      3,237    2           0.062%    High
  ```
  Scores {0,1,2,5,9} are very clean. Scores {3,6,7,8} carry >95 %
  of rule-errors.
- Changed: `scripts/xgb_dist_routed.py` (routing {1,2}),
  `xgb_dist_routed_v2.py` ({0,1,2,9}), `xgb_dist_routed_v3.py`
  ({0,1,2}), `xgb_dist_routed_v4.py` ({0,1}),
  `xgb_specialist_678.py` (specialist XGB on scores {6,7,8}),
  `hybrid_routed_spec.py` (glue: main routed-{1,2} overridden by
  spec on {6,7,8}), `xgb_per_class_specialists.py` (three
  specialists, one per rule class). Artefacts in
  `scripts/artifacts/`; submissions in `submissions/`.
- Routing-set ablation (tuned OOF bal_acc, 5-fold stratified):
  ```
  baseline XGB-dist (no routing)        0.97304
  route {1,2}     (v1)                  0.97333   +0.00029
  route {0,1,2}   (v3)                  0.97332   +0.00028   ≈ v1
  route {0,1}     (v4)                  0.97326   +0.00022   < v1
  route {0,1,2,9} (v2)                  0.97319   +0.00015   < v1
  ```
  Clean pattern:
  - Score 2 is net-positive to route (122 k × 99.7 % Low rows
    waste XGB boosting capacity on a near-trivial split).
  - Score 0 is a wash (too few examples at 33 k to matter).
  - Score 9 is net-negative to route (3.2 k High rows is 15 % of
    the entire High training pool; removing them hurts High
    calibration more than rule-routing gains).
  - General rule: **only route if (a) rule ≥ 99.5 % on the score
    AND (b) the class the rule predicts is over-represented in
    the non-routed training set**.
- Specialist on scores {6,7,8} (`xgb_specialist_678.py`):
  - Domain: 56,122 train rows, 69 % Medium / 31 % High (0 % Low).
  - 5-fold stratified-on-global-y XGB trained only on spec domain.
  - Specialist argmax bal_acc on its domain: 0.95198
  - Main XGB argmax bal_acc on same domain: 0.95088
  - Δ spec − main = +0.00109 (small but clean)
- **Hybrid pipeline (`hybrid_routed_spec.py`): new current best.**
  - Override routed-{1,2} XGB predictions on scores {6,7,8} with
    the specialist's predictions. Retune log-bias on the hybrid
    OOF (coord-ascent).
  - routed-{1,2} alone:                                  0.97333
  - routed-{1,2} + spec on {6,7,8}:  **0.97352**  Δ = +0.00019
  - Hybrid variant routed-{0,1,2} + spec {6,7,8}: 0.97352 (tied —
    score 0 routing doesn't touch {6,7,8} rows, no change).
- Per-class specialists (`xgb_per_class_specialists.py`): **null.**
  Three specialists, one per rule-class:
  ```
  Low-spec   (scores 0-3, 374k rows, 98 % Low)   dom bal_acc 0.505
  Med-spec   (scores 4-6, 235k rows, 98 % Med)   dom bal_acc 0.389
  High-spec  (scores 7-9,  21k rows, 92 % High)  dom bal_acc 0.849
  Fused OOF (per-row routed to matching specialist): **0.97226**
  Δ vs hybrid 0.97352: −0.00126
  ```
  Reading: specialization only helps when the domain has genuine
  class ambiguity. Low-domain (98 % Low) and Medium-domain (98.5 %
  Medium) specialists collapse into "predict the majority", so
  bal_acc on their minority flips is random (~0.5). Only High-spec
  made real use of specialization (+0.349 vs rule), because its
  domain is actually 3-class. The {6,7,8} spec works for the same
  reason — it's the only sub-domain where class distribution is
  balanced enough for a 3-class classifier to extract signal. Rule:
  **specialize on sub-domains with 20–80 % minority class**, not on
  sub-domains dominated by a single class.
- **LB submissions** (two hybrid variants submitted):
  - `submission_xgb_hybrid_routed_spec.csv` (routed-{1,2}):
    OOF 0.97352 → **LB public = 0.97224**. Gap 0.00128.
  - **`submission_xgb_hybrid_v3_routed012_spec678.csv`
    (routed-{0,1,2}): OOF 0.97352 → LB public = 0.97271.** Gap
    **0.00081** (narrowest we've seen). +0.00047 LB over the {1,2}
    variant despite identical OOF — the v3 variant is the new
    current best.
  - **Why v3 > v1 on LB despite OOF tie**: on training, all 33 767
    score-0 rows are truly Low, so XGB (v1) and rule (v3) agree on
    argmax → no OOF delta. On the **hidden test set**, XGB must
    extrapolate; it occasionally misfires on OOD score-0 rows
    while the rule is deterministic and correct 100 % of the time.
    Routing trades learned behaviour for a provably optimal
    deterministic one — robustness pays off on the hidden split.
  - New rule: **when a rule is ≥ 99.99 % accurate on a score, prefer
    routing over learning** even if OOF shows zero delta; it cuts
    test-time variance.
  - Updated calibration ladder:
    ```
    single tuned LGBM             0.97097 → 0.96972   gap 0.00125
    LGBM+DGP                      0.97271 → 0.97137   gap 0.00134
    bag + XGB blend               0.97327 → 0.97170   gap 0.00157
    routed-{1,2} + spec-{6,7,8}   0.97352 → 0.97224   gap 0.00128
    **routed-{0,1,2} + spec-{6,7,8} 0.97352 → 0.97271  gap 0.00081**
    ```
  - Pack 0.98114 still +0.00843 above. Leader 0.98219 still +0.00948.
  - Δ vs prior LB best (blend 0.97170): **+0.00101** cumulative.
- LB budget: 3/10 spent today (blend at 08:07, hybrid at 12:08,
  v3 hybrid at 12:29), 7 remaining.
- Read-out / next bets:
  1. The routing-sweet-spot is {1,2} or {0,1,2} tied. The spec-on-
     {6,7,8} is the real lift.
  2. Next architectural lever: **seed-bag the routed-XGB** (5 seeds,
     mirrors earlier LGBM-bag work, expected +0.0001–0.0003).
  3. **Blend routed-XGB-bag with LGBM-bag** — LGBM-bag artefacts
     need to be regenerated (~17 min). Expected +0.0002–0.0005.
  4. **Specialist-bag on {6,7,8}** — 56 k rows × 5 seeds is
     cheap. Expected +0.0001.
  5. **Spec on {3}** (4.8 % err rate, 102 k rows, 95 % Low / 5 %
     Medium). Worth trying since the class distribution is 95/5,
     not 98/2, and the 5 % minority is meaningful (4.9 k flips).
     Parallel structure to spec-{6,7,8}.
  6. Within-cell per-cell MLP remains the largest orthogonal lever
     (unexplored; expected +0.0005–0.002).

### 2026-04-21 — per-cell LR + specialist-augmented-with-original (two nulls)

- Goal: execute the two top bets for the "stacking exploration"
  branch: (a) per-cell logistic regression on within-cell continuous
  features as the within-cell architectural lever, and (b) augment the
  {6,7,8} specialist's training data with the 982 rule-clean rows from
  the 10k original dataset that have score in {6,7,8}.
- Changed: `scripts/per_cell_lr.py` (128-cell LR on 7 non-rule
  continuous features, Laplace-EB fallback for small/single-class
  cells), `scripts/per_cell_lr_blend_rule.py` (rule ⊗ LR sweep +
  error-overlap diagnostic), `scripts/xgb_specialist_678_aug.py`
  (synthetic-{6,7,8} ∪ original-{6,7,8} training with configurable
  sample weight), `scripts/hybrid_routed_spec_aug.py` (4-variant
  hybrid comparison).

- **Per-cell LR result (null)**:
  - With `class_weight='balanced'`: recovers **47.6% of rule-wrong
    rows** (4,908 / 10,304) but introduces **196,368 new false
    positives** on rule-right rows. Standalone OOF 0.73082.
  - Without balanced weights (correctly learns per-cell posteriors):
    standalone **0.96280 tuned** (vs EB-cell 0.96339, just below).
    Rule ⊗ LR log-blend tops at **0.96286** at α=0.20 (+0.00189 over
    rule-only; fully explained by log-bias tuning on a slightly
    richer prior — not by new signal).
  - LR recovers only 3.86% of rule-wrong rows after recalibration.
    Hard-gate over-rule at any τ ∈ {0.5,…,0.9} stays below rule-only.
  - Read-out: within-cell continuous features **do not carry
    orthogonal signal at LR capacity**. The rule's cell-majority
    prediction already uses all the information LR could extract.
    Same lesson as the 128-cell empirical Bayes null from 2026-04-21:
    any predictor that only sees a cell's row-level context through
    non-rule continuous features plateaus at ~0.963. MLP won't rescue
    it — same feature set, same per-cell data budget; the bottleneck
    is information, not model capacity.

- **Specialist augmentation result (null)**:
  - Original dataset has 982 rows with `dgp_score ∈ {6,7,8}` (666
    Medium + 316 High, all rule-correct). Synthetic {6,7,8} has
    56,122 rows with 13% rule-error rate.
  - Specialist-aug w=1.0 OOF on spec-domain: **0.95149** vs baseline
    specialist **0.95198** (Δ = −0.00049).
  - Specialist-aug w=0.3 OOF on spec-domain: **0.95142** (Δ = −0.00056).
  - Hybrid-level comparison (routed-{0,1,2} main + spec override on
    {6,7,8}, tuned log-bias per variant):
    ```
    main_only                    0.97332  (Δ = +0.00000)
    hybrid_spec_base    (ref)    0.97352  (Δ = +0.00020 vs main)
    hybrid_spec_aug_w1.0         0.97323  (Δ = -0.00010 vs main, −0.00029 vs hybrid)
    hybrid_spec_aug_w0.3         0.97326  (Δ = -0.00006 vs main, −0.00026 vs hybrid)
    ```
  - Both augmented variants are worse than both pure main AND the
    non-augmented hybrid. The 982 clean original rows pull the
    specialist's decision boundary toward the rule, eroding the
    specialist's flip-recovery edge — which is precisely the signal
    the hybrid relies on. Downweighting to 0.3 doesn't rescue it.
  - Read-out: **the specialist's value is that it trains ONLY on
    flip-rich synthetic data.** Adding rule-correct examples (even
    at 1.75% of the training pool) is net-negative for the override
    because the specialist becomes less different from the main XGB
    on exactly the rows where we want it to disagree. New rule:
    **don't augment specialist training with clean data if the
    specialist's purpose is to deviate from a clean predictor.**

- LB delta: n/a (0 LB spend; 3/10 cumulative, 7 remaining).
- Current best unchanged: hybrid_spec_base (routed-{0,1,2} + spec
  on {6,7,8}) at OOF **0.97352** / LB 0.97271.

### 2026-04-21 — model-stacking-exploration session: 9 nulls + routing-lever refinement

- Goal: a disciplined sweep over every plausible non-FE lever to close
  the +0.00843 gap to the 0.98114 pack (later revealed to be an
  ensemble of public-notebook submissions, not a modeling trick).
  Hypothesis board Open items systematically tested.
- Changed: 10+ new scripts in `scripts/`, all on
  `claude/model-stacking-exploration-s2osn`. Key adds:
  `per_cell_lr.py`, `per_cell_lr_blend_rule.py`, `xgb_specialist_3.py`,
  `xgb_specialist_46.py`, `xgb_specialist_678_aug.py`,
  `hybrid_routed_spec_aug.py`, `rule_distillation.py`,
  `benchmark_te_orig.py`, `benchmark_te_oof.py`,
  `benchmark_catboost_dist.py`, `xgb_dist_routed_v6.py`,
  `xgb_dist_routed_v7.py`, `pseudo_label_hybrid.py`.

- **Nine nulls** (ordered by when they landed):
  ```
  per-cell LR rule-blend               +0.00189  (just log-bias recal)
  specialist-{6,7,8} aug w=1.0         -0.00029  (hybrid)
  specialist-{6,7,8} aug w=0.3         -0.00026  (hybrid)
  spec-3 (95/5 domain)                  0.00000  (sub-heuristic)
  TE-from-original + LGBM              +0.00004
  OOF-TE + flip-rate + LGBM            +0.00005
  rule-distillation (10k features)     -0.00047
  spec-{4,6} (98/1/1 domain)            0.00000  (sub-heuristic)
  routed v6 {0,1,2,5}                  -0.00012
  routed v7 decoupled (all train + infer route)  -0.00044
  ```

- **The one new insight** — the routing lever is training-distribution
  rebalancing, not inference determinism or capacity-freeing:
  ```
  vanilla XGB-dist (train all, no route)       0.97304
  v3  (drop {0,1,2} train + route infer)       0.97332   ← best
  v6  (drop {0,1,2,5} both)                    0.97320
  v7  (train all, route {0,1,2,5} at infer)    0.97288
  ```
  V7 isolated the inference-routing component: train-on-all + route-
  infer is **worse** than vanilla XGB. This falsified the "anchor-row"
  theory (score-5 rows structurally informative for {6,7,8}
  boundary). The real lever: v3's training filter removes 271k easy-
  Low rows, which implicitly rebalances XGB's class prior — a pre-hoc
  version of what log-bias does post-hoc. Score 5 fails because
  dropping 79k clean-Medium rows unbalances in the wrong direction.

  Related lesson: **training-distribution engineering (remove easy
  rule-trivial rows) is more powerful than inference-routing for
  boosted trees.** Rule-route at inference for LB robustness only;
  the OOF lever lives entirely in the training data composition.

- **Three heuristics falsified / confirmed**:
  - Specialist 20-80 band confirmed empirically: {6,7,8} 69/31 lifts
    (+0.00019); {0-3} 98/2, {4-6} 98/1/1, score-3 95/5 all null.
    Minimum minority threshold is ~20%.
  - Routing heuristic gained a 3rd condition: routed rows must not
    be structurally informative (score-5 failure).
  - TE + LGBM fully falsified at num_leaves=127 on cat cards 2-6
    (TE-orig 0.97270, TE-oof 0.97271 both matching vanilla 0.97266).

- **Rival-approach analysis**: public notebook at LB 0.98114 pulls
  other people's submission CSVs as Kaggle Dataset inputs
  (`5(4)-0.98074.csv`, `5(5)-0.98057.csv`, `5(7)-0.98057.csv`, etc.)
  and ensembles them. The 0.98114 pack is a public-notebook-blend
  ceiling, not a modeling breakthrough. For our own pipeline, the
  ceiling is ~0.975–0.976 via compound own-pipeline diversity, not
  ~0.98. Re-framed the +0.01 target from "missing lever" to "stack
  more own OOFs".

- **Still running at session snapshot time**: CatBoost-dist (~1h10m
  wall, expected ~5 more min), pseudo-labeling hybrid (~30m wall
  expected). Results will land on `claude/model-stacking-exploration-
  s2osn` and can be merged separately.

- Current best unchanged: hybrid_spec_base (routed-{0,1,2} +
  spec-{6,7,8}) at OOF 0.97352 / LB 0.97271. LB budget: 3/10 used.

### 2026-04-21 — session wrap-up: new OOF best 0.97362 + artifacts for cross-branch blending

- Goal: close out the CatBoost + pseudo-labeling experiments and find
  one more lift via architectural blending.

- **NEW CURRENT BEST: hybrid × LGBM×XGB log-blend @ w_hyb=0.75 →
  OOF 0.97362 (+0.00010 vs hybrid alone).** First lift of the session
  after 12 nulls. Submission on disk:
  `submissions/submission_hybrid_lgbmxgb_blend.csv`. Blend is
  `0.75 × hybrid_v3 + 0.25 × (LGBM-dist × 0.45 + XGB-dist × 0.55)` in
  log space. Jaccard hybrid vs LGBM×XGB = 0.8053 (above our prior
  "skip" threshold) but blend still works — complementary error
  magnitudes rescued the borderline Jaccard.

- **CatBoost-dist standalone: 0.97128 (−0.00138 vs LGBM-dist).**
  Weakest of the three. Native ordered TE didn't help on 43-feature
  dist set. Jaccards with LGBM / XGB: 0.736 / 0.756 — both below
  0.80. Best 3-way blend `(L=0.4, X=0.5, C=0.1)` = 0.97320, **worse
  by 0.00007** than the 2-way LGBM×XGB. New negative-result rule:
  **low Jaccard is necessary but NOT sufficient for a useful blend**.
  CatBoost's unique errors landed on rows LGBM/XGB got right — any
  weight > 0 dragged the blend toward its wrong answers.

- **Pseudo-labeling τ=0.95 hybrid: −0.00020 null.** 226,749 test
  rows (84 %) pass confidence threshold, split 60/36/4 Low/Med/High.
  Augmented training: 630 k → 856 k per fold. Pseudo-hybrid tuned OOF
  0.97332 vs baseline 0.97352. Probable cause: hybrid's boundary-band
  errors get encoded in pseudo-labels; adding them with wrong class
  pushes decision surface in the wrong direction on exactly the rows
  the hybrid already mis-predicts.

- **Cross-container blending setup**: committed 5 OOF + 5 test `.npy`
  artifacts (~104 MB) to `scripts/artifacts/` via targeted `.gitignore`
  exception entries, plus `OOFS.md` manifest documenting fold
  convention (`StratifiedKFold seed=42`, 5-fold), class order
  (Low=0, Medium=1, High=2), load/tune/blend recipes, and regen
  instructions. Allows another container/branch to `git checkout
  main` and blend with our predictions via `np.load()`.

- Final tally this session: **13 experiments, 12 nulls, 1 lift.**
  Current best OOF 0.97362 (single-digit bps above 0.97352). LB best
  still 0.97271 (new blend unsubmitted; expected ~0.9728 at our ~0.0008
  OOF→LB gap). LB budget: 3/10 used.

- Lessons logged to LEARNINGS.md:
  - **Jaccard necessary but not sufficient for blend.** CatBoost
    Jaccard 0.74 with LGBM/XGB but blend hurt → need complementary
    error magnitudes, not just non-overlap.
  - **Pseudo-labeling compounds boundary errors when the labeler is
    systematically wrong on the boundary.** τ=0.95 was not high
    enough to filter out the hybrid's Medium↔High mistakes.
  - Routing heuristic 3rd condition already logged earlier in the
    session (training-distribution, not inference determinism or
    structural anchors).

### 2026-04-21 — soft-blend greedy forward: NEW LB BEST 0.97296

- Goal: regenerate saved OOFs for the top models (they were lost when
  the container was re-hydrated) and run a proper prob-space blend
  with OOF-gated evaluation, to see whether ensembling over our OWN
  pipelines produces real LB lift without adding a new model class.
- Context: the "stack more own OOFs" framing from the prior entry
  (rival-notebook pack is CSV ensembling) pointed at this — and we
  had zero `.npy` artefacts on disk, so even hard-vote on submission
  CSVs was limited to 0.99+ pairwise agreement with no way to
  OOF-score candidates. Blanket rule added to LEARNINGS.md: every
  training script must save `oof_*.npy + test_*.npy` as first-class
  outputs.
- Changed:
  - `scripts/blend_submissions.py` — hard-vote harness over saved
    CSVs; 7 strategies (plurality, weighted, Borda, veto, rule-
    deferred, High-supermajority, pairwise-veto). Surfaced the
    rare-class-preservation insight: blends that DEMOTE the rare
    class under macro-recall are likely LB-negative, even if they
    have similar or better OOF.
  - `scripts/hybrid_v3_reconstruct.py` — reassembles the hybrid_v3
    OOF from routed_v3 main + spec_678 (matches 0.97352 logged).
  - `scripts/blend_ensemble.py` — full soft-blend pipeline
    (standalone + pairwise α-sweep + equal-weight + greedy forward
    + logistic meta-stack with class_weight=balanced). Uses a
    vectorized `fast_bal_acc` that's 7.7x faster than
    `sklearn.balanced_accuracy_score` on 630k rows. Wide log-bias
    grid for the High class (up to +6 since optimum is ~+3.4).
  - `scripts/blend_greedy_finalize.py` — reproduces the greedy
    winner with a sensitivity sweep around the best weights.
  - `scripts/blend_high_weighted.py` — class-asymmetric variant
    that keeps hybrid_v3's Low/Medium probs and upweights consensus
    High prob from other models.
- Results (OOF tuned bal_acc, 5-fold stratified):
  - `lgbm_baseline`       0.97097 (reference)
  - `lgbm_dgp`            0.97271  rec_H=0.9603
  - `xgb_dist`            0.97304  rec_H=0.9631
  - `xgb_dist_routed_v3`  0.97332  rec_H=0.9657 ← highest High recall
  - `xgb_hybrid_v3`       0.97352  rec_H=0.9639 (reference for blend)
  - log mean of 6         0.97354
  - pair hybrid × routed  0.97366  (but DEMOTES 343 High rows on test — bad)
  - **greedy log-blend:**
    **hybrid_v3 (0.45) + routed_v3 (0.40) + spec_678 (0.15) = 0.97375**
    rec_H=0.9654 tuned bias=[0.132, 0.569, 3.401] +114 High on test
  - meta-stack LR-balanced   0.97348 (underperforms — components too
    correlated for a 12-feature logistic to add signal)
- **LB probe**: `submission_blend_greedy_w045_040_015.csv` uploaded.
  **LB public = 0.97296** (vs prior best 0.97271). Δ LB = +0.00025,
  matching the OOF prediction almost exactly. OOF→LB gap 0.00079,
  consistent with 0.00081 on hybrid_v3. No OOF overfit — the greedy
  log-blend found real signal.
- New calibration ladder:
  ```
  single tuned LGBM              0.97097 → 0.96972   gap 0.00125
  LGBM+DGP                       0.97271 → 0.97137   gap 0.00134
  bag + XGB blend                0.97327 → 0.97170   gap 0.00157
  routed-{1,2}+spec-{6,7,8}      0.97352 → 0.97224   gap 0.00128
  routed-{0,1,2}+spec-{6,7,8}    0.97352 → 0.97271   gap 0.00081
  **greedy 3-way log-blend       0.97375 → 0.97296   gap 0.00079**
  ```
  Δ vs prior LB best: +0.00025. Pack 0.98114 still +0.00818 above;
  leader 0.98219 still +0.00923 above. The own-pipeline stacking
  ceiling (~0.975-0.976 per the rival-analysis note) remains the
  expected upper bound for this approach family.
- LB budget: 4/10 used today, 6 remaining.
- Meta lessons (captured in LEARNINGS.md):
  1. `oof_*.npy + test_*.npy` are first-class outputs of every
     training script — not debug artefacts. Losing them to a
     container rehydrate cost ~45 min of regeneration on a day
     when blending was the entire goal.
  2. Committee pairwise agreement is the cheapest diagnostic for
     blend potential — 0.99+ means hard-vote blends are capped at
     ~0.003 lift at best, and the rare-class-preservation check
     determines whether the lift is positive or negative at all.
     Document in every blend design: "Δ rare-class count vs best
     standalone = ???".
  3. Greedy forward-selection (start from best standalone, add the
     component whose log-blend at the OOF-best α most improves
     tuned bal_acc) is the no-hyperparameter ensemble baseline
     that out-performed both the logistic meta-stack and the
     equal-weight average on this problem.
  4. Model-family diversity (LGBM × XGB) is worth ~+0.00015 — real
     but bounded. Within-family seed bagging and specialist
     overrides (+0.00020 each) are comparable levers; combining
     all three via greedy gets you to +0.00023 over the best
     single pipeline without adding a new model class.
  5. Cross-lineage blending is bounded by the anchor-model overlap.
     Main's `hybrid_lgbmxgb_blend` (OOF 0.97362) and our greedy
     (0.97375) both anchor on `xgb_hybrid_v3`; pairwise log-blend
     picks w_ours=0.95 → OOF 0.97376 (+0.00001, null). Two blends
     that share the dominant component don't compound — you need
     DIFFERENT anchors to get orthogonal signal.

### 2026-04-21 — training-data-quality experiments (3 nulls)

- Goal: test whether training-data-level changes — heavy-weight
  original augmentation and (target × dgp_score) stratified CV —
  lift the XGB-dist base model past its 0.97304 OOF. Follow-up
  after the soft-blend ceiling at ~0.9738.
- Changed: `scripts/data_quality_experiments.py` runs 4 configs on
  the same XGB-dist pipeline (43-feature dist set, same XGB HPs):
  baseline, orig w=20 target-strat, no orig score-strat, and both
  combined. Each saves `oof_xgb_dist_{config}.npy` + test counterpart.
- Results (OOF tuned bal_acc, 5-fold, seed=42):
  ```
  baseline (reproduced)                      0.97304   (--)
  orig w=20, target-strat                    0.97278   −0.00026
  no orig, (target × score) strat            0.97278   −0.00026
  orig w=20 + score-strat (combined)         0.97249   −0.00055
  ```
  All three configs net-negative. Baseline exactly reproduced
  (fold-for-fold argmax) so the deltas are real, not noise.
- **Diagnosis (heavy orig aug)**: 10k original is rule-perfect
  (no NN flips) while synthetic train AND test both contain
  10,304 deterministic NN flips. Biasing training toward the
  rule-perfect original pulls the decision surface AWAY from the
  flip signal. Fold-by-fold argmax shows 5/5 folds below baseline
  avg −0.00044. Per-class: rec_M drops (−0.00066 on argmax),
  rec_H basically flat. The flip signal lives in Medium↔High
  boundary, and that's exactly where the model loses capacity when
  it's anchored on rule-perfect data.
- **Diagnosis (score-stratified CV)**: Fold variance drops from
  σ ~0.0008 to σ ~0.0002 (stratification works on per-fold
  calibration) but tuned OOF is unchanged at 0.97278. At 630k
  rows, the default StratifiedKFold(shuffle=True, seed=42) already
  produces well-balanced score-bin distributions per fold by sheer
  sample size — explicit stratification adds zero information.
- **Diagnosis (combined)**: the two changes compound negatively;
  the combined score-strat + w=20 config drops −0.00055, worse
  than either individually.
- Meta-lesson: the LB-target signal comes from fitting the DGP NN
  flips that live in the 630k synthetic data. External "clean"
  data at any weight > 1× per row is counterproductive when the
  test set shares the same noise process as the train. Rule added
  to LEARNINGS.md: "When your train and test share a deterministic
  noise process absent from external data, external data at any
  weight > prior is net-negative."
- LB budget: unchanged at 4/10 used today. No submission warranted
  from these experiments since all configs landed below current
  LB-verified best (greedy log-blend OOF 0.97375).
- Next-bet status: training-data-quality ruled out. The ~0.9738
  OOF ceiling for our tree-ensemble family appears to be the
  genuine plateau; further lift would need a structurally
  different model class (MLP retry with larger capacity, or a new
  feature view).

### 2026-04-21 — end-of-day session wrap-up

- **Leaderboard final state**: **LB 0.97296** via
  `submission_blend_greedy_w045_040_015.csv` (greedy log-blend of
  hybrid_v3 0.45 + routed_v3 0.40 + spec_678 0.15). This is the
  verified LB best and should be locked as one of the 2 final-
  selection submissions.
- **Final-selection candidates** (pick 2 of 2 before competition
  close):
  1. **Primary**: `submission_blend_greedy_w045_040_015.csv`
     (OOF 0.97375 / LB 0.97296). Current best.
  2. **Safe fallback**:
     `submission_xgb_hybrid_v3_routed012_spec678.csv`
     (OOF 0.97352 / LB 0.97271). Minimal variance, clean pipeline.
     Good hedge if the blend overfits on private LB.
  If a genuinely different model (MLP with capacity) lands a lift
  before the deadline, swap the safe fallback for that.

- **Single highest-ROI remaining experiment: large-capacity
  tabular NN.** Rationale:
  1. The label-generation process is a deterministic NN (per
     2026-04-21 DGP residuals EDA). Our best trees plateau at
     ~0.9738 OOF regardless of FE/blending/data-quality lever.
  2. A 3-layer 50k-param MLP previously plateaued at 0.966, but
     that's 1-2 orders of magnitude below a serious tabular NN.
     Capacity-bound, not structurally wrong.
  3. Every non-NN lever explored this session (DQ, cross-lineage
     blend, class-asymmetric High mixing, meta-stack) converged
     to within ±0.00002 of the same OOF ceiling — signature of
     an architectural bottleneck, not a tuning one.

  **Concrete action plan for next session**:
  - Target: FT-Transformer (1-3M params) or NumEmb + wide MLP
    (500k params) on the 43-feature dist set.
  - Bootstrap: `./bootstrap.sh` (data rehydrate).
  - Pre-check after 1 fold: compute OOF error Jaccard vs
    `oof_xgb_hybrid_v3.npy`. Gate decision:
    - Jaccard ≥ 0.90: kill, NN is mimicking the tree ensemble.
    - Jaccard < 0.85: run all 5 folds, then blend into greedy.
    - 0.85 ≤ Jaccard < 0.90: run all 5 folds, but treat the blend
      lift ceiling as +0.00015 not +0.001+ (same as the MNLogit /
      balanced-ensemble diagnosis rule).
  - Expected: +0.001 to +0.003 LB if the NN is genuinely
    orthogonal; 0 if it plateaus at the tree ceiling.
  - Budget: ~1-2 hours compute (GPU strongly preferred).

  Second-priority experiment if NN plateaus: **seed-bag the
  greedy log-blend** (3 seeds × same weight vector). Variance
  reduction on the current best. Expected +0.0001-0.0003 LB.
  Cheap (~60 min) and guaranteed-safe.

### 2026-04-21 — binary 'is High?' head + hybrid blend: NEW OOF BEST 0.97398

- Goal: brainstorm #1 (High-class lever). High has 3x leverage under
  balanced accuracy (1/3 of macro-recall), so a dedicated binary head
  specialising on `P(High | x)` may lift the hybrid's High posterior.
- Changed: `scripts/binary_high_head.py` — XGBoost `binary:logistic`
  with 43-feature dist set, same 5-fold split as all other OOFs (seed
  42). Three blend variants (prob-mix, geo-mix, logit-add) swept
  against `oof_hybrid_lgbmxgb_blend.npy`, each with coord-ascent
  log-bias. Artefacts: `oof_xgb_bin_high.npy`, `test_xgb_bin_high.npy`,
  `oof_hybrid_binhigh.npy`, `test_hybrid_binhigh.npy`,
  `binary_high_head_results.json`,
  `submissions/submission_hybrid_binhigh_tuned.csv`.
- Binary head OOF AUC = **0.99866** (5 folds, 526-713 best_iter).
  Distance features + rule indicators separate High trivially; 3.3%
  prior class has a crisp decision boundary.
- Blend sweep results (OOF tuned bal_acc):
  ```
  baseline hybrid_lgbmxgb_blend                0.97362

  prob-mix:     w=0.00  0.97362
                w=0.35  0.97396  (+0.00034, peak)
                w=1.00  0.97352

  geo-mix:      w=0.00  0.97362
                w=0.35  0.97396  (+0.00034, peak)
                w=1.00  0.97352

  logit-add:    lam=0.00  0.97362
                lam=+0.60 0.97398  (+0.00036, peak, OVERALL BEST)
                lam=+2.00 0.97383
                lam=-1.00 0.66360  (destroys probs as expected)
  ```
  All three sweeps produce clean unimodal curves — not single-point
  flukes. Prob-mix and geo-mix agree on the optimal weight (w=0.35),
  logit-add squeezes another 0.00002 at lam=+0.60 (equivalent
  strength, different parameterisation of the same intervention).
- **New current best: 0.97398 OOF** (logit-add lam=+0.60).
  Delta vs hybrid +0.00036. Inside 1sigma fold-std (~0.00088) in
  absolute terms, but the smooth monotonic sweep structure confirms
  the signal is real, not selection noise.
- Confusion matrix at tuned operating point:
  ```
           Low  Medium   High    per-class recall
  Low   368354    1561      2    99.578%
  Medium  5120  229697   4257    96.078%
  High       0     727  20282    96.540%
  ```
  Compared to hybrid baseline (not re-run but trivially similar):
  High recall lifted to 96.54% — this is what the binary head bought.
  Medium is now the weakest leg (96.08%); Medium->High confusions
  dominate the remaining error mass (4257 out of ~11k total errors).
- Implication for next bets: **High-class lever still has meat**. The
  binary head's +0.00036 came from pushing a few hundred boundary
  rows from Medium to High correctly. The lever is not exhausted —
  a second High-specialist (different feature subset or different
  seed) may stack further. More importantly, brainstorm #7 (non-rule
  features only) now has a concrete mechanism hypothesis: if those
  features carry the NN-flip signal, a non-rule-feature-only head
  blended similarly could push another bucket of Medium->High flips.
- **LB probe: submitted at 17:44, result 0.97212** — worse than current
  LB best (`submission_blend_greedy_w045_040_015.csv` submitted
  earlier today by parallel session, 0.97296 on LB / 0.97375 OOF).
  OOF->LB gap for binhigh = 0.97398 − 0.97212 = **0.00186**, far wider
  than the greedy blend's 0.00079. **The OOF gain did not transfer.**
- **Root cause: selection overfit on top of an already-tuned
  pipeline.** This experiment optimised:
  1. Binary-head XGB (4k round early stopping on fold val bal_acc).
  2. Log-bias coord-ascent on hybrid baseline (already done).
  3. Three blend parameterisations × ~20-30 grid points = ~75
     candidates, each with its own log-bias coord-ascent.
  4. Argmax over all sweep points.
  Each nested tuning on OOF compounds small selection biases that
  don't exist on the hidden LB. The prior hybrid had already been
  OOF-tuned (blend weights, log-bias) — layering another round of
  OOF tuning added ~0.0011 OOF-only inflation on top of a baseline
  that had ~0.0015 worth of same.
- **Rule: when adding a new component on top of a stack that is
  already OOF-tuned (blend weights + log-bias), expect the real LB
  delta to be ~1/3 of the OOF delta.** Current-best OOF 0.97398 →
  LB 0.97212 (0.00186 gap, 5.2x above the 0.00036 OOF lift). The
  greedy 3-way blend submitted earlier by parallel session was the
  right reference baseline (0.97375 OOF / 0.97296 LB, 0.00079 gap)
  because it tuned fewer hyperparameters per additional component.
- **Revised current best**: `submission_blend_greedy_w045_040_015.csv`
  at LB **0.97296** (submitted 16:09 today, not by this branch).
  Our OOF best (0.97398) is on disk but LB-inferior to the greedy.
- LB budget: **5/10 used today** (was under-counted earlier; the
  greedy sub by parallel session counts), 5 remaining.
- Calibration ladder update:
  ```
  hybrid_lgbmxgb_blend          0.97362 -> ~0.9727 expected (not subbed)
  greedy 3-way log-blend        0.97375 -> 0.97296   gap 0.00079 <- LB BEST
  hybrid + binhigh logit_add    0.97398 -> 0.97212   gap 0.00186 <- OVERFIT
  ```
- Next bet: instead of piling more tuned blends on top, run
  brainstorm #7 (non-rule-features-only flip predictor) — it's
  architecturally orthogonal, so new information not new
  OOF-selection. And consider adding binhigh to the *greedy*
  blend pipeline (not the hybrid_lgbmxgb_blend) with minimal
  additional tuning to see if the High-head signal survives the
  selection-tightened baseline.

### 2026-04-21 — binhigh lever falsified on greedy stack (fixed-bias sweep)

- Goal: test whether the +0.00036 OOF lift from binhigh survives
  honest tuning, by adding it to the LB-validated greedy blend with
  a single parameter (logit-add lam on the High column) and the
  greedy's already-fitted log-bias reused as-is.
- Changed: `scripts/greedy_binhigh_minimal.py` — reconstructs greedy
  from committed components (hybrid_v3 = routed_v3 with spec_678
  override on dgp_score ∈ {6,7,8}, then 0.45 hybrid + 0.40 routed +
  0.15 spec log-blend), fits log-bias once, sweeps lam ∈ {0, 0.05,
  …, 0.50} with that bias FIXED. Artefacts: `oof_greedy_blend.npy`,
  `test_greedy_blend.npy`, `greedy_binhigh_minimal_results.json`.
- Results (OOF bal_acc at fixed greedy bias = [0.1324, 0.5689, 3.4008]):
  ```
  greedy baseline (lam=0)      0.97375  (matches prior LB-0.97296 sub)
  lam=0.05                     0.97372  (−0.00002)
  lam=0.10                     0.97364  (−0.00011)
  lam=0.15                     0.97330  (−0.00044)
  lam=0.20                     0.97302  (−0.00072)
  lam=0.30                     0.97246  (−0.00129)
  lam=0.50                     0.97168  (−0.00207)
  ```
  **Monotonic decrease.** Binary-High head adds zero information to
  the greedy stack; with tuned bias reused as-is, any positive lam
  strictly hurts OOF.
- **Binhigh lever is DEAD** as a greedy-stack add-on. The earlier
  +0.00036 OOF lift on `hybrid_lgbmxgb_blend` was a log-bias
  artefact: retuning bias after injecting P(High) lets coord-ascent
  push the High threshold up, which inflates OOF without new signal.
  That's exactly why it lost 0.00084 LB vs greedy.
- Gap math now reconciled:
  ```
  greedy   OOF 0.97375 − LB 0.97296 = 0.00079  (honest calibration)
  binhigh  OOF 0.97398 − LB 0.97212 = 0.00186  (overfit by 0.00107)
  ```
  The 0.00107 overfit = all of the log-bias-retune inflation.
- No LB submission (fixed-bias sweep strictly negative). Budget
  unchanged at 5/10 used, 5 remaining.
- New rule: **when adding a component to a tuned blend, sweep with
  fixed baseline bias first.** If fixed-bias OOF doesn't improve,
  the component is redundant with the blend — retuning bias on top
  will manufacture a fake lift that vanishes on LB.
- Next bet: brainstorm #7 (non-rule-features-only flip predictor).
  Architectural not tuning — tests whether the NN-generator's flip
  signal hides in `Humidity, Prev_Irrig, EC, Soil_pH, Organic_C,
  Sunlight, Field_Area, Region, Crop_Type, Soil_Type`, which trees
  on the rule features alone can't fully access.

### 2026-04-21 — non-rule-features-only blend: NEW LB BEST 0.97352 (+0.00056)

- Goal: brainstorm #7. The NN label generator (`brief.md:74`) likely
  used non-rule features to perturb labels away from the rule. A
  model restricted to just those features captures exactly that
  perturbation signal, orthogonal by construction to tree models
  that are dominated by the 6 rule features.
- Changed: `scripts/nonrule_features_only.py` — XGBoost 3-class
  `multi:softprob` on 13 non-rule features only (`Soil_Type, Soil_pH,
  Organic_Carbon, Electrical_Conductivity, Humidity, Sunlight_Hours,
  Crop_Type, Season, Irrigation_Type, Water_Source, Field_Area_hectare,
  Previous_Irrigation_mm, Region`), same 5-fold split (seed=42) as all
  other OOFs. Fixed-greedy-bias sweep over log-blend α. Artefacts:
  `oof_xgb_nonrule.npy`, `test_xgb_nonrule.npy`, `nonrule_results.json`,
  `submission_greedy_nonrule_blend.csv`.
- Standalone (non-rule features only): OOF argmax = 0.42965,
  tuned = 0.56966 — barely above random. Model learns almost nothing
  class-predictive from these features alone.
- Fixed-bias log-blend sweep (greedy tuned baseline = 0.97375,
  bias = [0.1324, 0.5689, 3.4008]):
  ```
  alpha_nonrule=0.00  OOF = 0.97375  Δ = +0.00000  (baseline)
  alpha_nonrule=0.05  OOF = 0.97383  Δ = +0.00008
  alpha_nonrule=0.10  OOF = 0.97400  Δ = +0.00026
  alpha_nonrule=0.15  OOF = 0.97421  Δ = +0.00047   ← peak
  alpha_nonrule=0.20  OOF = 0.97419  Δ = +0.00044
  alpha_nonrule=0.25  OOF = 0.97397  Δ = +0.00022
  alpha_nonrule=0.30  OOF = 0.97379  Δ = +0.00004
  alpha_nonrule=0.40  OOF = 0.97262  Δ = -0.00113
  alpha_nonrule=0.50  OOF = 0.96998  Δ = -0.00377
  ```
  Clean unimodal peak at α=0.15, symmetric curve. FIXED bias throughout —
  no retune compensation. The signal is real, not calibration-manufactured.
- Confusion-matrix deltas at α=0.15 (blend − greedy):
  ```
                    Low recall    Medium recall   High recall
  greedy (ref)      0.99566       0.96013         0.96544
  greedy + nonrule  0.99554       0.95785         0.96925
  delta            -0.00012      -0.00228        +0.00381
  ```
  Non-rule blend trades ~540 Medium rows for ~80 High flips. Net
  positive because High has 3× leverage under balanced accuracy.
  Mechanism: non-rule features (especially `Humidity`,
  `Previous_Irrigation_mm`, `Region`) carry the NN-generator's flip
  signal that axis-aligned trees on rule features can't fully access.
- **LB probe: submitted at 18:26, result 0.97352** — **new LB best**,
  +0.00056 vs greedy's 0.97296.
- Calibration ladder update:
  ```
  hybrid_lgbmxgb_blend          0.97362 -> LB (not submitted)
  greedy 3-way log-blend        0.97375 -> 0.97296   gap 0.00079
  hybrid + binhigh (overfit)    0.97398 -> 0.97212   gap 0.00186
  **greedy + nonrule α=0.15**   **0.97421 -> 0.97352   gap 0.00069 ← NEW BEST**
  ```
  **Gap shrunk from 0.00079 to 0.00069** — honest architectural lever,
  opposite of the binhigh experiment where gap blew up on retune.
  Confirms the methodology: fixed-bias fixed-sweep over a new model
  family is a reliable way to validate lifts before LB.
- Hypothesis confirmed: **the NN label generator does perturb labels
  via non-rule features**. The effect is small (~80 flips per 630k
  rows) but real. Any further gains on this lever should stack
  cleanly because the non-rule model isn't using rule features at
  all — there's no information leak with the greedy ensemble.
- LB budget: **6/10 used today**, 4 remaining.
- Next bets unlocked by this result:
  1. **Second non-rule model** (LGBM or CatBoost variant, or different
     seed) — bag the non-rule predictor, then blend. Expected
     +0.00005–0.0002 cheap variance reduction.
  2. **Brainstorm #8 (two-stage rule-base + non-rule correction)** —
     explicitly predict `y − rule_pred` instead of y from non-rule
     features. Now well-motivated since we know the lever works.
  3. **Non-rule model with rule_pred or dgp_score as an input** —
     lets the model learn "predict rule unless non-rule features
     suggest otherwise". Hybrid of the two frames.
  4. **Stack with the existing binhigh head** — binhigh and nonrule
     attack different rows (binhigh = amplify rule-strong rows,
     nonrule = correct rule-wrong rows). The second overfit didn't
     mean the first was worthless — they may stack.

### 2026-04-21 — nonrule + rule_pred + dgp_score (null, confirms orthogonality)

- Goal: test whether augmenting the 13 non-rule features with
  `rule_pred` (categorical, 3 classes) and `dgp_score` (int 0-9)
  lets XGB learn corrections like "rule says Low but Humidity +
  Prev_Irrig pattern → actually Medium" that pure nonrule can't
  express. Risk: the model simply parrots `rule_pred`, losing the
  architectural orthogonality that makes #7 work.
- Changed: `scripts/nonrule_with_rulepred.py` — 3-class XGB on 15
  features (13 non-rule + rule_pred cat + dgp_score num), same
  5-fold split (seed=42), fixed-greedy-bias sweeps. Artefacts:
  `oof_xgb_nonrule_rulepred.npy`, `test_xgb_nonrule_rulepred.npy`,
  `nonrule_rulepred_results.json`.
- Results (OOF, 5-fold, seed=42):
  - Standalone argmax = 0.96052 (rule's ceiling), tuned = 0.96481.
    Above pure rule's 0.96097 — the model DID learn non-rule
    corrections on top of the rule signal.
  - Onto greedy alone: peak α=0.05 → 0.97382 (+0.00007 vs 0.97375,
    within fold noise). Every α > 0.05 strictly hurts.
  - Onto base (greedy + XGB-nonrule @0.15): β=0 peak 0.97421,
    monotonic decrease.
  - 3-way (XGB-nonrule + this + greedy): best at a=0.15, b=0.05,
    g=0.80 → 0.97418 (Δ = −0.00003 vs base).
  - Error Jaccard (new vs XGB-nonrule) = 0.037 — they make very
    different errors (inter 12k, union 333k; new model has
    6-7× fewer errors overall because it uses rule).
- **Architectural confirmation**: the non-rule lever works precisely
  BECAUSE it ignores rule features. Adding `rule_pred` pulls the
  model's predictions close to greedy (which also uses rule
  features) — the different errors vs XGB-nonrule are exactly the
  errors greedy already corrects. So the blend adds redundancy,
  not orthogonality.
- New rule: **diversity from "ignoring a feature class" is
  additive to a model that DOES use that feature class; but
  diversity from "using the same feature class differently" is
  usually redundant in a blend**. XGB-nonrule wins by being
  rule-free, not by being a tree.
- No submission (fixed-bias sweep capped below +0.0003). LB budget
  unchanged at 6/10 used, 4 remaining.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.
- Next bets unchanged from previous entry (seed-bag, pseudo-label,
  self-distillation, rule × non-rule pairwise FE on greedy).

### 2026-04-21 — nonrule-lever stacking batch: LGBM + weighted-shift + featsubset + EBM (four nulls)

After the non-rule-features-only lever hit LB 0.97352 (+0.00056), this
session tested four follow-ups to stack more diversity into the
same lever. All four null — the non-rule signal is fully captured
by the single XGB-nonrule model on 13 features; no additional
architecture or feature view adds orthogonal bits at this base.

- **LGBM variant of nonrule** (`scripts/nonrule_lgbm_blend.py`).
  Standalone OOF argmax 0.42924 / tuned 0.56791 — tracks XGB-
  nonrule (0.42965 / 0.56966) to 3 decimals. Onto greedy alone:
  peak α=0.20 → 0.97415 (+0.00041, below XGB's +0.00047).
  2D sweep XGB_nr + LGBM_nr + greedy: best (0.05, 0.15, 0.80) →
  0.97421 ties the base. 1D stacking: β=0 wins. LGBM and XGB
  produce near-identical predictions on 13 non-rule features —
  leaf-wise vs level-wise tree construction not enough diversity.

- **Weighted-shift retry** (`scripts/nonrule_shift_weighted.py`).
  Sample_weight=100 on shift≠0 rows. Model learns flip
  discrimination now (y-argmax 0.76 vs vanilla's 0.96 parrot-
  rule) but standalone tuned 0.95892 — WORSE than the rule
  (0.96097). Blend sweep monotone negative from α=0. Upweight
  100x overshot: model predicts too many rows as flipped,
  degrading clean-row predictions. Would need HP tuning on the
  weight.

- **Feature-subset bagging (#+ user idea)**
  (`scripts/nonrule_featsubset_bag.py`). 5 XGB sub-models, each
  on a different 4-feature subset of 7 top non-rule features
  (Humidity, Prev_Irrig, EC, Field_Area, Region, Crop_Type,
  Soil_Type). Log-mean ensemble standalone tuned 0.53720 —
  BELOW both XGB-nonrule full (0.56966) and every individual
  subset except D (0.40620, weakest). Onto greedy alone: peak
  α=0.15 → 0.97383 (+0.00009, way below XGB's +0.00047). Onto
  base monotone negative. 3-way XGB+ens+greedy also null. Each
  individual subset at β=0.10 onto base: all −0.00011 to
  −0.00031. Diagnosis: the 5 subsets share too many features
  (each feature in 3 subsets), ensemble converges to a weaker
  version of XGB-nonrule-full. Feature-subspace diversity on
  only 7 features doesn't have room.

- **EBM variant** (`scripts/nonrule_ebm_blend.py`). Fold 1 took
  **1742s (29 min)**, argmax bal 0.42421 — identical to XGB (0.42913)
  and LGBM (0.42730). Killed after fold 1: (a) 5 folds would cost
  2.5+ hours, (b) fold-1 argmax parity with LGBM/XGB means EBM
  won't add blend diversity at this feature set — same ceiling,
  different architecture. Saved for potential revival only if a
  lever shows up that makes the compute justifiable.

- **Summary of the stacking batch**: XGB-nonrule-full on 13
  features is the single best expression of the non-rule lever.
  LGBM, EBM, feature-subset, and shift-weighted all track or
  underperform it. The diversity we need has to come from
  somewhere OTHER than "different model on the same non-rule
  features" — likely from either different features (rule ×
  non-rule cross FE still untested on greedy), a different fold
  split (seed-bag), or a genuinely new data source.

- LB budget: **6/10 used today** (unchanged). Current best:
  `submission_greedy_nonrule_blend.csv` OOF 0.97421 / LB 0.97352.
- Next bet: seed-bag XGB-nonrule (5 seeds), OR try
  rule_pred-as-feature for nonrule (which we'd rejected as
  architectural leak but is worth a fixed-bias probe), OR go
  broader: test-time augmentation, self-distillation, or
  pseudo-labeling via current best.

### 2026-04-21 — two-stage shift-correction (brainstorm #8, null)

- Goal: predict ordinal shift `y - rule_pred + 2 ∈ {0..4}` from
  non-rule features only, convert to y-probs via the rule offset
  map, blend into greedy with fixed bias. Hypothesis: by baking
  rule_pred into the target, the model concentrates capacity on the
  NN-perturbation residual instead of re-learning the class prior.
- Changed: `scripts/nonrule_shift_correction.py` — 5-class
  `multi:softprob` XGB on 13 non-rule features, same 5-fold stratified
  split on y (seed=42). Conversion `shift5_to_y3(p_shift, rule_pred)`
  with clipping at y=[0,2]. Artefacts: `oof_xgb_shift5.npy`,
  `test_xgb_shift5.npy`, `oof_xgb_shift_to_y.npy`,
  `test_xgb_shift_to_y.npy`, `shift_results.json`.
- Observed shift distribution on train (after conversion):
  `shift=-1: 0.52%`, `shift=0: 98.36%`, `shift=+1: 1.12%`. **No shift
  of ±2** — the NN never flips two classes.
- Results (OOF tuned bal_acc, 5-fold, seed=42, fixed greedy bias):
  - Standalone shift->y: argmax 0.96097, tuned 0.96097 — matches the
    rule's ceiling. Model converged to "parrot rule_pred".
  - Onto greedy: α=0.00 peak 0.97375, α=0.05 0.97372 (−0.00002),
    α=0.15 0.97326 (−0.00049). Monotone negative.
  - Onto greedy+nonrule (current LB best): α=0.00 peak 0.97421,
    α=0.10 0.97393 (−0.00028). Also monotone negative.
- Diagnostic: best_iter 59-108 rounds (vs 1100+ for direct-y
  nonrule #7). Early stopping saturated on "predict shift=0 always"
  — the 98.36% majority dominates 5-class log-loss and the rare
  shift-±1 signal never gets enough gradient to matter.
- **Lesson**: the shift framing is structurally fragile when the
  majority class dominates ≥95% of the target. Direct-y 3-class
  keeps the model learning per-row Low/Medium/High discrimination
  across all 630k rows; shift framing lets it collapse to a
  one-class predictor. Would need either (a) heavy sample-weight
  upweighting of shift-±1 rows, (b) stratified balanced sampling,
  or (c) binary classifier on "is flipped?" + direction head.
- No LB submission (fixed-bias sweep strictly negative). LB budget
  unchanged at 6/10 used, 4 remaining.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.
- Next bet: brainstorm #7 follow-up #1 — seed-bag the non-rule
  model (5 seeds, ~15 min). Cheapest variance reduction on the
  only architecturally-diverse leg we have. Or #4 — stack the
  binhigh head with non-rule in the greedy pipeline (still on
  fixed bias, just a 2-parameter sweep); binhigh's diagonal
  ~0.99 AUC on High was never fully tested with honest tuning.

### 2026-04-21 — rank-sum / Borda blend (null, first of brainstorm batch)

- Goal: falsify the "sum" lever. All prior blends (LGBM×XGB,
  hybrid×blend) were prob-space or log-space. Rank-averaging is
  calibration-invariant; if the per-model confidence-scale was
  limiting prob blends, rank-avg should lift.
- Changed: `scripts/rank_blend.py` — per-column rank-normalisation to
  `[0, 1]`, averaged across model subsets, softmaxed row-wise, coord-
  ascent log-bias. Three aggregators (rank_avg, rank_wavg weighted
  by standalone bal_acc, Borda via softmax), four subsets (all 4
  committed OOFs, no-hybrid, hybrid+xgb_v3, hybrid+all_base).
  Artefact: `scripts/artifacts/rank_blend_results.json`.
- Results (OOF tuned bal_acc, 5-fold stratified, seed=42):
  ```
  baseline hybrid_lgbmxgb_blend (current best)   0.97362
  baseline xgb_dist_routed_v3                    0.97332
  baseline xgb_vanilla_dist                      0.97304
  baseline lgbm_te_orig                          0.97270

  rank_avg_all4                                  0.96800
  rank_avg_no_hybrid                             0.96810
  rank_avg_hybrid+xgb_v3                         0.96739
  rank_wavg_all4                                 0.96800
  rank_wavg_no_hybrid                            0.96808
  borda_softmax_all4                             0.96800
  borda_softmax_no_hybrid                        0.96810
  ```
  All 12 rank/Borda variants land at **0.96739–0.96810** —
  **−0.0055 to −0.0062 below current best**, far worse than every
  base learner and clearly outside fold-noise.
- Mix sweep (α = rank-weight in a `α·rank + (1−α)·prob` blend of
  hybrid + xgb_v3):
  ```
  α=0.00  0.97368   ← pure prob-avg, tiny +0.00006 over hybrid
  α=0.10  0.97362   ← ties hybrid
  α=0.50  0.97340
  α=1.00  0.96739   ← pure rank, null
  ```
  α=0.00 found a +0.00006 crumb (simple 50/50 prob-avg of hybrid +
  xgb_v3 edges the current best) but that's within fold-std noise
  (~0.00088) and has nothing to do with rank aggregation — it's just
  a different point in prob space.
- Read-out: **rank aggregation throws away absolute-probability
  information that log-bias tuning needs.** Balanced-accuracy tuning
  for a 3-class problem requires per-class calibrated probabilities
  to shift operating points; a rank distribution squashes class
  posteriors to nearly-uniform after row-softmax, losing the sharp
  separation LGBM/XGB provide on clean rows. Calibration-invariance
  isn't actually a benefit here — the component models already
  produce comparable probability scales because they train on the
  same loss.
- New rule: **for 3-class balanced-accuracy problems with
  log-bias-tuned decision rules, rank-space blending is strictly
  dominated by prob/log-space blending.** Don't retry rank-avg
  variants. Keep prob and log blends for component-model fusion.
- LB delta: n/a (0 LB spend; 3/10 cumulative).
- Current best unchanged: `oof_hybrid_lgbmxgb_blend` at OOF 0.97362 /
  LB-best 0.97271. First of the brainstorm batch — moving to bet #1
  (binary "is High?" head) next.

### 2026-04-22 — NN lever closed: 5 MLP variants all null, seed-bag LB regression

- Goal: exhaust the "large-capacity tabular NN" hypothesis that sat at
  the top of the Open bets list. Prior 50k-param MLP on a parallel
  branch hit 0.966 standalone / blend null; today's work scales
  capacity 20×, tests 4 structural variants, and pushes to Kaggle GPU
  to remove compute as an excuse.
- Infrastructure: Kaggle Kernels API for free T4/P100 GPU. Kernel
  metadata + boot-time `torch==2.5.1+cu121` shim to handle P100 (sm_60)
  incompatibility with pre-installed torch 2.10. Two private datasets
  uploaded: `irrigation-greedy-blend-oof` (oof_greedy_blend.npy +
  test_greedy_blend.npy + oof_xgb_nonrule.npy) as the fold-1
  error-Jaccard gate reference + blend baseline. Three kernels pushed
  (5 MLP variants total).

  All kernels share: 5-fold StratifiedKFold(shuffle=True,
  random_state=42) pinned for OOF alignment with on-disk OOFs. Same
  Balanced Softmax loss (Menon 2021) with per-fold prior recomputation
  on filtered training subsets. Same AdamW + cosine schedule. Fold-1
  error-Jaccard kill gate vs greedy (0.90 abort / 0.85 warn) and vs
  xgb_nonrule (since xgb_nonrule is in our LB-best stack at α=0.15).

- Results table (OOF tuned bal_acc, 5-fold, seed=42):
  ```
  variant           params  feat                  standalone  J-greedy  J-nonrule  blend vs greedy  blend vs greedy+nonrule
  v5 full            1.0M   43                    0.96494     0.676     0.032      monotone −      monotone −
  v6 nonrule-only    150k   13 (6 cat + 7 num)    0.43338     0.015     0.350      monotone −      monotone −
  v7 top-3 numerics   15k   3                     0.42393     0.015     0.353      monotone −      monotone −
  v8 spec {6,7,8}    200k   43 (on 56k rows)      0.64 ungated / 0.9358 on-domain (vs xgb_spec_678 0.9520)   override monotone −
  v9 routed {0,1,2}  1.0M   43 (on 359k rows)     0.96477     0.689     0.032      monotone −      monotone −
  ```
- Diagnoses per variant:
  - **v5 full features [768,512,384,256]**: 1M params, 30 epochs,
    dropout 0.25. Fold-1 Jaccard 0.668 vs greedy looked promising,
    but blend null in both prob and log space. |E_mlp|=12,005 vs
    |E_greedy|=8,909 — MLP's different errors are also MORE numerous,
    and its disagreements with greedy are more often MLP-wrong than
    MLP-right. Classic "Jaccard necessary but not sufficient".
  - **v6 non-rule features only [256,192,128,96]**: direct NN analog
    of xgb-nonrule (LB-winning lever at +0.00056). Standalone 0.433
    matches xgb-nonrule's 0.430 argmax — same ceiling. But 384k
    errors makes the blend sweep catastrophically negative.
  - **v7 top-3 flip-significant numerics [128,96,64]**: Humidity,
    Previous_Irrigation_mm, Electrical_Conductivity only. Standalone
    0.424, blend null. Even tighter information bottleneck didn't
    produce orthogonal signal after normalizing for weakness.
  - **v8 specialist {6,7,8} [384,256,192,128]**: 56k rows (45k per
    fold) with dropout 0.40. On-domain bal_acc 0.9358 **below** xgb
    specialist's 0.9520. MLP data-starved on the small sub-domain;
    XGB's axis splits generalize better at this scale.
  - **v9 training-data routed (exclude score {0,1,2})** [768,512,384,256]:
    359k train rows, at inference route score-{0,1,2} to rule. Standalone
    OOF 0.96477 — identical to v5 (0.96494). The "easy-row gradient
    domination" hypothesis that explained xgb_dist_routed_v3's
    LB-winning +0.00047 is **falsified for MLPs**: Balanced Softmax
    + uniform CE already handles class imbalance, so removing 271k
    trivial Low rows doesn't shift MLP behavior. Training-data
    engineering is a tree-specific lever, not an NN-universal one.
- Collective read-out: NN architectural plateau at ~0.965 for
  full-feature variants is insensitive to:
  - 20× capacity span (50k → 1M params)
  - Feature-set width (3 / 13 / 43 columns)
  - Training-data policy (all / filter / specialist)
  - Domain-restricted specialization
  With every degree of freedom exercised and still null, the NN
  lever is architecturally exhausted on this problem. Any further
  NN capacity scaling (FT-Transformer, tabular ResNet, ensemble of
  seeds) is unlikely to break the pattern — this is not a
  capacity-or-optimizer problem, it is an information-bottleneck
  one that no feature-independent NN can route around.

- Second result this session — **seed-bag greedy LB regression**
  (submitted):
  - Local experiments: `xgb_dist_routed_v3_seed7.py`,
    `xgb_specialist_678_seed7.py`, `xgb_spec_3.py` (all fold_seed=42
    pinned, XGB_SEED=7 for seeded training).
  - `seed_bag_greedy_analysis.py` bagged routed + spec across seeds
    {42, 7}, rebuilt hybrid (routed overridden by spec on {6,7,8}),
    rebuilt greedy at (0.45, 0.40, 0.15) log-blend.
  - OOF: **0.97385** tuned (Δ = +0.00010 vs seed=42 greedy's 0.97375,
    within fold-std noise σ=0.00088 but directionally positive).
  - LB (submitted 05:43): **0.97284** — REGRESSION −0.00012 vs
    single-seed greedy LB (0.97296). OOF→LB gap widened from
    0.00079 to 0.00101.
  - Diagnosis: XGB at our hyperparams is near-deterministic across
    seeds (per-seed routed_v3 OOF range 0.97332→0.97342 = 0.00010
    spread). A 2-seed bag has too little variance to reduce; the OOF
    "lift" is calibration artifact on the log-bias coord-ascent,
    not signal. Rule added: **below-1-fold-std OOF lifts from
    near-deterministic bags should be treated as non-signal on LB.**

- Third result — spec-3 null (as predicted by 20-80% heuristic):
  - `xgb_spec_3.py` specialist on the 102k-row score=3 domain (95%
    Low / 5% Medium / 0% High). Spec-domain bal_acc 0.5040 vs rule's
    0.5 floor. Hybrid override −0.00011 vs greedy; soft-blend sweep
    monotone negative.
  - Rule-confirmation: **specialists need 20–80% minority mass**.
    95/5 with zero High is below threshold; Low-spec + Medium-spec
    per-class specialists from main's session had the same failure.

- LB state: best unchanged at **LB 0.97352**
  (`submission_greedy_nonrule_blend.csv`). 1/10 LB spend today, 9
  remaining.
- Calibration ladder update:
  ```
  single tuned LGBM                 0.97097 → 0.96972   gap 0.00125
  LGBM+DGP                          0.97271 → 0.97137   gap 0.00134
  bag + XGB blend                   0.97327 → 0.97170   gap 0.00157
  routed-{0,1,2}+spec-{6,7,8}       0.97352 → 0.97271   gap 0.00081
  greedy 3-way log-blend            0.97375 → 0.97296   gap 0.00079
  hybrid + binhigh (overfit)        0.97398 → 0.97212   gap 0.00186
  **greedy + nonrule α=0.15         0.97421 → 0.97352   gap 0.00069**  ← LB BEST
  seed-bag greedy                   0.97385 → 0.97284   gap 0.00101  (null)
  ```
- Strategic read: own-pipeline ceiling confirmed at OOF ~0.974 /
  LB ~0.9735. Every architectural + representation + data-policy
  lever has been exercised. The remaining +0.008 to the 0.98114
  pack requires public-CSV blending (the pack's actual mechanism),
  which is a strategic choice, not a modeling one. If we stay on
  own-pipeline, 0.97352 is very likely our final LB floor.

### 2026-04-22 — NN-on-original as features (idea 1, null in two modes)

- Goal: execute the user-reframed idea — train our own NN on the 10k
  rule-perfect original and apply to synthetic features. Under the
  right framing (host's synthetic labels = host's NN on synthetic,
  where that NN was trained on the original), our own NN-on-original
  should partially reproduce the flip pattern through its smooth
  decision boundary.
- Changed: `scripts/nn_orig_features.py` (5-arch MLP ensemble trained
  on 10k original, predicts on 630k train + 270k test);
  `scripts/blend_nn_orig_greedy.py` (fixed-bias log-blend sweep vs
  greedy); `scripts/xgb_dist_with_nn_feats.py` (adds 3 NN prob cols
  to the 43-feature dist set, retrains XGB, blends).
- Protocol iteration #1 (full 43-feature dist set incl. `rule_pred`
  and `dgp_score`): every arch collapsed EXACTLY to the rule ceiling
  (orig train acc 1.0, synth bal_acc 0.96097 to 5 decimals for all
  5 archs). NNs trivially parrot `rule_pred` when it's in the input.
  Killed — fundamental flaw, features had to be restricted.
- Protocol iteration #2 (continuous features only: 11 raw numerics +
  4 signed dist + 4 abs dist + min_axis_abs + 2 pairwise products +
  8 categoricals, total 22 num + 8 cat). NN must now re-discover the
  rule from smooth signals only.
  - Ensemble 5-arch (7k–73k params each, 150 epochs, ~2 min total CPU):
    orig train bal 0.999+, synth tuned **0.9448**, error Jaccard vs
    greedy **0.3716** (very low, ens errs 21,097 vs greedy 8,909).
  - **Blend sweep (fixed greedy bias)**: peak at α=0 (no blend);
    monotone negative from α=0.02 (−0.0002) through α=0.50 (−0.0118).
    **Null**.
  - Why: Jaccard 0.37 = error orthogonality, but ens has 2.4× more
    errors than greedy; weighting in any NN prob drags the blend
    toward the NN's wrong answers faster than it helps.
- Protocol iteration #3 (NN probs as 3 new tree features on top of
  XGB-dist): XGB retrained on 46-feature set (43 dist + 3 NN probs).
  Standalone tuned **0.97306** vs vanilla XGB-dist 0.97304 (Δ =
  +0.00003). Error Jaccard vs greedy = **0.9537** (basically no
  diversity). Blend sweep peak α=0.40 at 0.97376 (+0.00001 vs
  greedy, null).
  - Why: XGB at max_depth=7 already splits optimally on signed dist
    + dgp_score; the NN's 3-dim prob is a re-encoding of that signal
    with additional noise from the NN's smoothing errors. Trees
    correctly learn to ignore it.
- **Idea 1 is ruled out in both framings** (prob blend, tree feature).
  Lesson: our 5-arch small-MLP ensemble trained on 10k rule-perfect
  rows does NOT reproduce the host NN's specific flip pattern. The
  "smooth approximation" character of a NN is narrowly determined by
  its architecture × the 10k anchor points, and our architectural
  envelope doesn't cover the host's specific function. Without
  matching the host's architecture + training recipe, the
  NN-on-original is just a noisier restatement of the rule.
- LB budget: 1/10 used today (unchanged — no LB probe justified, all
  fixed-bias sweeps were < +0.0005 threshold).
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.
- Idea 2 (pretrain-finetune MLP) is architecturally distinct —
  whole model, not just predictions — and still open. Scaffolded
  as Kaggle kernel `kaggle_kernel/kernel_pretrain_ft/` to run on GPU.

### 2026-04-22 — pretrain-finetune MLP (idea 2, null — NN lever stays closed)

- Goal: test whether pretraining on 10k original (rule-perfect) then
  fine-tuning on 630k synthetic breaks the MLP plateau at 0.965. The
  hypothesis: v5 MLP plateau was an optimization issue (NN never
  settled into the rule basin via joint rule+flip training); a
  rule-aligned initialization from pretrain should let fine-tune
  refine *toward* the host NN rather than from scratch.
- Changed: `kaggle_kernel/kernel_pretrain_ft/mlp_pretrain_ft.py`
  (v5 architecture 1M params + phase-1 pretrain 30 ep on 10k orig,
  CE + orig Balanced Softmax prior; phase-2 fine-tune 15 ep per
  fold on synth, LR 1e-4 = 10× lower than pretrain, synth Balanced
  Softmax). Uploaded 10k original as private dataset
  `chrisleitescha/irrigation-prediction-original` (l3llff public
  dataset rejected by kernel push API). Kernel v2 on T4 GPU,
  ~6 min total (30 s pretrain + ~70 s finetune per fold × 5).
  `scripts/blend_mlp_pretrain_ft.py` runs fixed-greedy-bias blend
  sweep against both greedy (0.97375) and greedy+nonrule (LB-best
  0.97421).
- Standalone results (OOF bal_acc, 5-fold stratified, seed=42):
  - Per-fold val bal_acc: 0.9633, 0.9645, 0.9653, 0.9622, 0.9625
    (σ ≈ 0.0012, tight).
  - OOF argmax **0.96358** / tuned **0.96361** — essentially same as
    v5 full-feature MLP (0.9649). Pretrain effect: **null** at the
    standalone level.
  - Fold-1 error Jaccard vs greedy = **0.6626** (below 0.85 warn;
    passed the kill gate). ft errs = 12,524 vs greedy's 8,909.
- Blend sweep (fixed greedy bias = [0.1324, 0.5689, 3.4008]):
  - vs greedy (0.97375): monotone negative from α=0.02 (−0.00006)
    through α=1.0 (−0.047). Peak at α=0.0. **Null.**
  - vs greedy+nonrule LB-best (0.97421): same pattern, α=0.02
    (−0.00022), monotone negative. Peak at α=0.0. **Null.**
- Diagnosis: the MLP plateau at ~0.965 is not a basin-finding or
  capacity problem — pretrain init did not move the standalone
  ceiling. The ~3,615 extra MLP errors vs greedy's error set (Jaccard
  0.65, ft has 41 % more errors) are MLP-wrong on rows greedy got
  right; any positive α weights those wrong answers into the blend
  and hurts. Same mechanism as v5's blend-null: **Jaccard 0.66 is
  orthogonal ENOUGH to look promising but the magnitude of extra
  MLP errors (41 %) defeats the blend lift.**
- Meta-read: ideas 1 (NN on orig as features, 3 variants) and 2
  (pretrain-finetune MLP) both null. Combined with the 2026-04-22
  NN-lever closure (5 MLP variants v5-v9), this is the 10th MLP-style
  null on the problem. The pattern is consistent: **MLPs trained on
  the 43-feature dist set plateau at ~0.965 regardless of
  initialization, capacity, training-data policy, or pretrain
  strategy**; and **MLPs trained on 10k rule-perfect original do not
  reproduce the host NN's flip pattern** whether their predictions
  are used as probs or tree features. The host-NN-reverse-engineering
  approach is exhausted via plain MLP architectures on our current
  feature set. Pivot back to compounding own-pipeline levers or the
  strategic option (public-CSV blending).
- LB budget: unchanged, 1/10 used today. No submission from either
  idea — both fixed-bias sweeps were strictly negative, well below
  the +0.0005 LB-probe threshold.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.

### 2026-04-22 — Tier 1+2 own-pipeline levers exhausted (6 nulls)

- Goal: systematically test the remaining Tier 1 + Tier 2 levers on
  top of the LB-best greedy+nonrule baseline (OOF 0.97421 / LB 0.97352)
  after NN/DGP-reversal was closed. User constraint: no public-CSV
  blending.
- Changed: `scripts/per_score_log_bias.py`, `scripts/seed_bag_nonrule.py`,
  `scripts/pseudo_label_v2.py`, `scripts/error_analysis_greedy_nonrule.py`
  and the corresponding spec-6 / self-distill probes;
  `kaggle_kernel/kernel_ftt/ft_transformer.py` (4-block × 8-head FT-
  Transformer, d_token=192, 20 epochs, ~1h50min wall on T4). Blend
  script `scripts/blend_ft_transformer.py` for the fixed-bias sweep.
- Results (all vs LB-best OOF 0.97421 at fixed greedy bias):
  ```
  per-score log-bias (30 params, nested CV)   NULL  Δ = −0.00031
  seed-bag XGB-nonrule (5 seeds)              NULL  blend Δ ≈ 0
  pseudo-label v2 (τ=0.99, Low-only)          NULL  monotone neg
  spec-6 override (cell-2 targeted)           NULL  monotone neg
  self-distill XGB (teacher=greedy+nonrule)   NULL  Jaccard 0.93
  FT-Transformer standalone                   0.96780 (+0.003 vs v5)
  FT-Transformer blend vs LB-best             NULL  monotone neg
  ```
- Notable findings:
  1. **Per-score bias overfits**: full-fit at 0.97429, nested at 0.97391.
     Coord-ascent over 30 params × 31-point grid picks dramatic biases
     on score 4 (High +0.8) and score 5 (Low −1.9) that don't generalize.
     Lesson: global 3-param log-bias is at the right granularity for
     this problem; finer-bin decomposition overfits without massive
     per-bin data.
  2. **FT-Transformer is the first NN to break the MLP plateau.** OOF
     tuned 0.96780, +0.003 over v5's 0.9649. Attention-based
     architecture finds a different attractor than the v5-v9 plain
     MLPs — fold-1 error Jaccard vs greedy = **0.614** (the lowest
     NN Jaccard of the competition). But FT-T has 12,634 errors
     (+42 % over greedy's 8,909) and that magnitude mismatch defeats
     the blend math at every α > 0.01. Both sweeps (vs greedy, vs
     LB-best) monotone negative from α=0.02.
  3. **Self-distillation saturates at argmax**: best_iter=75 (vs
     ~600 for vanilla XGB), student Jaccard 0.93 with teacher.
     Classic distillation-for-diversity failure mode: the student
     quickly learns argmax and doesn't develop orthogonal errors.
  4. **Error analysis** (run on greedy+nonrule CM): 74 % of errors at
     score 3 (cell rule=Low→true=Medium, n=5041, driven by high
     Rainfall_mm, Cohen's d=+0.557) and score 6 (cell rule=Medium
     pushed-to-High, n=4163, driven by low Soil_Moisture, d=−0.526).
     Per-score bias and spec-6 both failed to address these cleanly;
     the errors live in within-cell feature continuous signal that
     trees + MLP + transformer all encode similarly.
- LB delta: n/a. 1 LB submission spent today (unchanged since NN
  session). 9 LB remaining.
- **Final-candidate assessment**:
  - Primary: `submission_greedy_nonrule_blend.csv` (LB 0.97352).
    Confirmed.
  - Safe fallback: `submission_xgb_hybrid_v3_routed012_spec678.csv`
    (LB 0.97271).
  - No candidate from this session outperforms either. **Tier 1 + 2
    own-pipeline levers are exhausted on this baseline.**
- Remaining untried own-pipeline bet: rule×non-rule pairwise FE on
  the greedy base (untested; previously null on hybrid_lgbmxgb_blend).
  Not expected to exceed +0.0005 based on adjacent-experiment deltas.
  CORN / Frank-Hall ordinal decomposition was the other candidate;
  executed 2026-04-22 and closed as null (see entry below).

### 2026-04-22 — Frank-Hall ordinal decomposition (null vs LB-best)

- Goal: test the one remaining unexecuted Tier-2 lever — two binary
  XGB heads for ordinal y (Low<Medium<High), recomposed Frank-Hall
  style with post-hoc monotone clip. Hypothesis: Bernoulli loss on
  each ordinal cut focuses capacity on ONE boundary at a time
  (Low↔Medium, Medium↔High), which should produce a different
  decision surface than multi:softprob. Motivated by error analysis
  showing 74 % of greedy+nonrule errors land on score=3 (Low↔Medium)
  and score=6 (Medium↔High) cells.
- Changed: `scripts/ordinal_corn.py` — head A `P(y>=Medium)` +
  head B `P(y>=High)`, both on the 43-feature dist set, 5-fold
  stratified (seed=42) for OOF alignment. Monotone clip enforces
  `P(y>=High) <= P(y>=Medium)`. Artefacts
  `oof_xgb_corn{,_head_a,_head_b}.npy`, `test_xgb_corn*.npy`,
  `ordinal_corn_results.json`. Wall time ~2 min.
- Head diagnostics (5-fold OOF AUC): head A 0.99788 ± 0.00013,
  head B 0.99865 ± 0.00011. Both near-perfect binary separators;
  the ordinal cut is well-learned. Monotone clip kicked in on
  0.14 % of OOF rows and 0.06 % of test rows — a trivial
  correction.
- Standalone (OOF bal_acc): argmax **0.96396**, tuned
  **0.97354**. In the same band as xgb_hybrid_v3 (0.97352) and
  xgb_dist_routed_v3 (0.97332). Different error trade vs LB-best
  (greedy+nonrule): Medium recall +0.0050 (96.283 vs 95.785), High
  recall -0.0072 (96.207 vs 96.925). CORN trades High for Medium —
  the wrong direction under macro-recall because High has 3× the
  leverage.
- Fixed-bias blend sweeps:
  ```
  vs greedy (0.97375, LB 0.97296)
      alpha=0.25  0.97397  Δ=+0.00022
      alpha=0.30  0.97399  Δ=+0.00024
      alpha=0.40  0.97400  Δ=+0.00025   ← peak
      alpha=0.50  0.97396  Δ=+0.00021

  vs greedy+nonrule (0.97421, LB 0.97352)   ← LB-best
      alpha=0.15  0.97429  Δ=+0.00008
      alpha=0.20  0.97428  Δ=+0.00007
      alpha=0.25  0.97429  Δ=+0.00008
      alpha=0.30  0.97430  Δ=+0.00008
      alpha=0.40  0.97430  Δ=+0.00009   ← peak
      alpha=0.50  0.97418  Δ=-0.00003
  ```
- Read-out: **null vs LB-best.** The +0.00009 peak is inside the
  fold-std noise band (~0.00088) and below the +0.0002 threshold
  for an LB probe. Expected LB if submitted ≈ 0.97352 + ~0.00008 =
  ~0.9736, indistinguishable from current LB-best at the noise
  floor. The +0.00025 lift over the greedy baseline is cleaner but
  still translates to expected LB ~0.97321, **below** LB-best.
  **No submission warranted.**
- Why the null: the Frank-Hall decomposition IS architecturally
  orthogonal to multi:softprob — the two models make materially
  different error trades (Medium↔High vs High↔Medium). But the
  greedy+nonrule blend already occupies the macro-recall-optimal
  point for this feature set; CORN's trade moves error mass in the
  direction of Medium at the expense of High, which hurts balanced
  accuracy even if standalone tuned is on par with the best
  xgb_hybrid variants. Third consistent signal this week that the
  ~0.974 OOF / ~0.9735 LB ceiling is architecturally invariant
  across tree-based families on this feature set.
- LB delta: n/a. Budget unchanged (1 used today, 9 remaining).
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  at OOF 0.97421 / LB 0.97352. CORN artefacts committed for
  cross-branch reuse (`oof_xgb_corn.npy`, `test_xgb_corn.npy`).

### 2026-04-22 — TabPFN v2 as new blend leg (null)

- Goal: test the last unexercised architectural lever — a pretrained
  tabular foundation model (TabPFN v2.2.1). Hypothesis: TabPFN is
  pretrained on millions of synthetic tabular DGPs, including the
  regime the host used to generate the 630k synthetic train set.
  Every NN we've trained from scratch (v5-v9, FT-Transformer,
  pretrain-finetune) plateaued in either standalone OOF or blend.
  A foundation model conditions on a small context each fold
  instead of gradient-fitting, and may bring genuine orthogonal
  signal that survives the magnitude-mismatch blend trap.
- Changed: `scripts/ordinal_tabpfn.py` — TabPFN v2 classifier on
  the 43-feature dist set, SUBSAMPLE=1500 stratified training rows
  per fold (CPU compute cap), N_ESTIMATORS=1, same 5-fold
  StratifiedKFold(seed=42) for OOF alignment. Chunked test
  prediction (5k-row batches). Fold-1 error-Jaccard gate vs
  greedy + LB-best (0.90 abort). Wall time: 5 folds × ~30 min
  on 16-core CPU = ~2h30min. `tabpfn==2.2.1` pre-license version
  (v7.x requires Prior-Labs API token).
- Standalone results (OOF bal_acc):
  - Per-fold argmax: 0.9602 / 0.9615 / 0.9631 / 0.9605 / 0.9452
    (σ ≈ 0.0065, fold 5 dragged down by subsample draw variance).
  - **OOF argmax = 0.95811, tuned = 0.96209.** Below XGB-dist
    (0.9726) and the MLP ceiling (0.9649) — the 1500-row context
    cap hurts vs TabPFN's 10k sweet spot.
  - Fold-1 Jaccard gate: vs greedy 0.8486, vs LB-best 0.8445.
    Passed the 0.90 abort threshold, landed in the 0.85-0.90
    "warn" band (blend lift ceiling ~+0.00015 per CLAUDE.md rule).
  - Full-OOF Jaccard: vs greedy 0.8139, vs LB-best 0.8081.
    **TabPFN errors = 10,376 vs LB-best = 8,891 (+16.7 %)** —
    better than FT-Transformer's +42 % overshoot but still too
    many to blend productively.
- Fixed-bias blend sweeps (same greedy bias, no retune):
  ```
  vs greedy (0.97375)
      alpha=0.000  0.97375  Δ=+0.00000   ← peak (blend degrades the moment TabPFN enters)
      alpha=0.025  0.97372  Δ=-0.00002
      alpha=0.050  0.97359  Δ=-0.00015
      alpha=0.500  0.97000  Δ=-0.00375

  vs greedy+nonrule LB-best (0.97421)
      alpha=0.000  0.97421  Δ=+0.00000
      alpha=0.025  0.97424  Δ=+0.00002   ← peak (within fold noise)
      alpha=0.050  0.97412  Δ=-0.00009
      alpha=0.500  0.97000  Δ=-0.00422
  ```
  vs LB-best: monotonic decrease past α=0.025. The +0.00002
  "peak" is indistinguishable from noise (fold σ ~0.00088). No
  submission warranted. vs greedy: best is α=0, TabPFN strictly
  hurts the blend.
- Per-class recall (standalone tuned):
  ```
                Low    Medium    High
  recall     0.9957   0.9665   0.9238
  ```
  Low and Medium are competitive with LB-best; High recall is
  substantially worse (0.9238 vs LB-best's 0.9693). The small
  context size (1500 rows × 3.3 % High = ~50 High examples per
  fold) is the bottleneck — TabPFN can't see enough High rows
  to model the rare class well.
- Read-out: **NN lever remains closed across every architecture
  family tested on this feature set.** TabPFN trained on millions
  of pretrained tabular DGPs, with an architecturally-distinct
  in-context learning mechanism, still plateaus at the same
  ~0.96 band and can't beat the magnitude-mismatch blend trap.
  Combined with the 10+ prior NN nulls (MLP v5-v9,
  FT-Transformer, pretrain-finetune MLP, 3 NN-on-orig variants),
  this is definitive: **no NN architecture on this feature set
  produces errors with both the right orthogonality AND the
  right magnitude to lift a greedy+nonrule blend**. The own-
  pipeline ceiling at LB ~0.9735 appears structural.
- Caveat: TabPFN v2 sweet spot is 10k training rows. We used
  1500 for CPU-compute feasibility. A GPU run at
  SUBSAMPLE=10000, N_ESTIMATORS=4 could lift standalone to
  maybe 0.97, but given the Jaccard 0.81 and error-magnitude
  pattern, the blend outcome is unlikely to change — the
  underlying decision surface TabPFN finds is different enough
  from trees to make different errors but not so different that
  the errors are complementary to the greedy+nonrule stack.
- LB delta: n/a. Budget unchanged (1 used today, 9 remaining).
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  at OOF 0.97421 / LB 0.97352. TabPFN artefacts committed
  (`oof_tabpfn.npy`, `test_tabpfn.npy`) for cross-branch reuse.

### 2026-04-22 — careful XGB HP tuning (all 3 components, 80-trial Optuna): LB null

- Goal: full HP sweep over the three XGB components underpinning
  the LB-best pipeline — `xgb_dist_routed_v3`, `xgb_specialist_678`,
  `xgb_nonrule` — to settle whether careful per-component tuning
  beats their default HPs. The 2026-04-20 LGBM HP Optuna sweep was
  null but on a different feature set; XGB had never been swept
  comprehensively on this problem.
- Changed: `scripts/hp_common.py` (shared FE + HP space + objective),
  `scripts/hp_{dist_routed,spec_678,nonrule}.py` (per-component
  Optuna TPE, inner 80/20 split, MedianPruner, 80 trials each),
  `scripts/refit_best_hp.py` (5-fold outer-CV refit with tuned HPs),
  `scripts/blend_tuned_greedy.py` (rebuild greedy + nonrule blend,
  fixed-bias α sweep, nested-CV weight search).
- HP space: `lr ∈ [0.02, 0.15]`, `max_depth ∈ [4, 10]`,
  `min_child_weight ∈ [1, 30]`, `subsample/colsample ∈ [0.6, 1.0]`,
  `reg_alpha/reg_lambda ∈ [1e-8, 10]`, `gamma ∈ [1e-8, 5]`. Objective:
  prior-reweight bal_acc on inner val (argmax bal_acc for spec_678,
  since Low is absent from spec domain).
- Acceptance gate: inner-val lift > 1 fold-std (0.001) for each
  component. All three passed.
- Inner-val HP results (per-component Δ on 80/20 inner split):
  ```
  spec_678    baseline 0.94954 -> 0.95146   +0.00193  (max_depth=4, lr=0.072, reg_lambda=7e-4)
  dist_routed baseline 0.96836 -> 0.97055   +0.00218  (max_depth=4, lr=0.028, reg_alpha=2.98)
  nonrule     baseline 0.55442 -> 0.56461   +0.01019  (max_depth=4, lr=0.026, reg_alpha=5.38)
  ```
  Common pattern: **shallow trees (max_depth=4 vs baseline 7), slower
  learning, much heavier L1 regularization**. Consistent with the
  LGBM Optuna finding — same "shallow + regularized" alternative
  optimum regime, but with apparent real lift this time.
- Outer 5-fold OOF refit (standalone, tuned HPs):
  ```
  dist_routed            0.97332 -> 0.97405   +0.00073  (realized 33% of inner predict)
  spec_678 spec-domain   0.95198 -> 0.95258   +0.00060  (realized 31%)
  nonrule tuned          0.56966 -> 0.57611   +0.00645  (realized 63%)
  ```
  Realized/predicted ratio 30-65% — typical Optuna selection-bias
  compression. Lifts shrunk but remained positive on all three.
- Blend rebuild at production weights (0.45 hybrid + 0.40 routed +
  0.15 spec) with fixed greedy log-bias + fixed-bias α sweep:
  ```
  greedy tuned OOF              0.97431   (baseline 0.97375, +0.00056)
  α=0.15 (production)           0.97455   (baseline 0.97421, +0.00034)
  α=0.20 (sweep peak)           0.97461   (+0.00040)
  nested-CV mean (unbiased)     0.97470 ± 0.00091   (+0.00049)
  ```
  Alpha sweep a clean unimodal curve peaking at α=0.20. Nested-CV
  inner folds consistently preferred w_routed 0.45-0.55 (higher
  than production 0.40) and α=0.20-0.25.
- LB probe (submitted both candidates at 18:01 UTC):
  ```
  prodAlpha015  OOF 0.97455 -> LB 0.97336  gap 0.00119  Δ LB -0.00016 vs best
  peakAlpha020  OOF 0.97461 -> LB 0.97331  gap 0.00130  Δ LB -0.00021 vs best
  ```
  **Both tuned variants REGRESSED on LB** despite honest OOF gains.
  OOF→LB gap widened **by 50-60 bps** (baseline gap 0.00069 -> tuned
  gap 0.00119-0.00130).
- Read-out: HP tuning is a **structural generalization null**, not a
  selection-bias null. Unlike binhigh (post-hoc log-bias retune =
  clear selection-bias failure), this experiment used FIXED production
  blend weights AND FIXED α throughout — no stage-wise OOF selection.
  The failure mode is that the Optuna-favored HP regime (shallow
  trees, heavy L1 reg) fits the visible data distribution better but
  generalizes less well to the hidden LB split than the baseline
  regime (moderate depth, light reg). The baseline HPs sit in a
  robustness sweet-spot that inner-val and outer-CV bal_acc don't
  reward.
- New rule (added to LEARNINGS.md candidate): **On this problem,
  inner-val and 5-fold outer-CV reward per-component capacity/
  regularization tradeoffs that do NOT transfer to LB. Treat any HP
  search lift below +0.001 at the blend level as a likely LB null
  even when selection bias is controlled; require blend-level
  lift ≥ +0.001 before burning an LB slot on HP tuning.**
- LB budget: **3/10 used today**, 7 remaining. Cumulative this
  competition: prior count + 2 new = bumped 2.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  at **LB 0.97352 / OOF 0.97421** with baseline HPs.
- Artefacts retained for reference (gitignored `_tuned.npy` arrays
  not committed; the submission CSVs and the
  `blend_tuned_greedy_results.json` are):
  ```
  submissions/submission_greedy_nonrule_tunedHP_prodAlpha015.csv
  submissions/submission_greedy_nonrule_tunedHP_peakAlpha020.csv
  scripts/artifacts/blend_tuned_greedy_results.json
  scripts/artifacts/hp_{dist_routed,spec_678,nonrule}_best.json
  scripts/artifacts/refit_best_hp_results.json
  ```
- Next: own-pipeline lever bank is now fully exhausted. Remaining
  path to improvement is strategic (public-CSV blending) or pivot
  to within-cell continuous feature modeling — previously ruled out
  at linear capacity (per-cell LR = 0.96280, EB = 0.96339), but
  per-cell MLP untested and constrained by per-cell data (largest
  cell ~100k rows, smallest <300).

### 2026-04-22 — OTE scaffolded on top of digit-XGB (LB best 0.97468), seed-bag deferred

- Goal: pull the digit-extraction breakthrough onto this branch and
  scaffold the OTE (Ordered Target Encoding) follow-up — the second
  half of the public-notebook "digit + OTE" recipe.
- Branch state: digit-extraction commits (32e6b42, 46ef32a, fb9d5b7)
  already merged into `claude/plan-next-steps-I32rC` via main. Digit
  scripts (`xgb_dist_digits.py`, `digit_features.py`,
  `lgbm_dist_digits.py`, `blend_digits.py`) and artefacts
  (`oof_xgb_dist_digits.npy`, `test_xgb_dist_digits.npy`,
  `xgb_dist_digits_results.json`) all present locally. No port needed.
- Seed-bag survey across all branches — **5 prior experiments, 4
  null/regressed**:
  1. `seed_bag_dist` (LGBM-dist 5-seed model bag) — OOF +0.00023.
     Small lift, never LB-tested standalone.
  2. `seed_bag_dist_fe` (LGBM-dist+FE 5-seed model bag) — null.
  3. `seed_bag_nonrule` (xgb_nonrule 5-seed model bag) — null
     (model variance below noise).
  4. `seed_bag_greedy` (2-seed model bag of routed + spec) —
     OOF +0.00010, **LB −0.00012** (regression).
  5. `session_b_*` (3-FOLD-seed bag of full greedy+nonrule stack) —
     OOF +0.00040, **LB −0.00055** (worst regression).
  Diagnosis (from CLAUDE.md): XGB at our HPs is near-deterministic
  across model seeds, and fold-seed bagging produces OOF lift via
  log-bias coord-ascent overfit, not signal. Two confirmed rules:
  - "below-1-fold-std OOF lift from near-deterministic bags =
    non-signal on LB" (model-seed bagging)
  - "fold-seed bagging creates OOF lift but not necessarily LB lift"
    (Session B, OOF→LB gap blew up from 0.00069 to 0.00164)
  **Implication**: digit-XGB seed-bagging — initially framed as
  "cheap insurance" in the prior session's hypothesis board — is
  RISKY, not safe. Skipping in favour of OTE.
- OTE scaffold (3 new files):
  - `scripts/ote_features.py`: `OTE` class doing per-row
    K-shuffled cumulative target stats. Per-shuffle inner loop is
    vectorised: factorise key, then for each shuffle compute exclusive
    cumsum per key on the shuffled-onehot, scatter back to original
    row order, average across K. Test transform uses full-train
    per-key lookup (unseen → prior). Smoke-tested on 12-row toy
    (3 cats × 3 classes), correctly produces per-row noisy estimates
    that average to the per-cat full-train mean. Benchmarked at 22 s
    for one 300-key pair × 8 shuffles × 504 k rows.
  - `scripts/xgb_dist_digits_ote.py`: same XGB pipeline as
    `xgb_dist_digits.py` (43 dist + 46 digits) plus 16 OTE keys
    × 3 classes = 48 OTE columns. 5-fold StratifiedKFold(seed=42)
    aligned with every other OOF. Per-fold OTE: fit on tr_idx, apply
    to va_idx (no leak). Test OTE: fit on full train (one-shot
    outside the fold loop). Wall budget estimate ~25-30 min.
  - `scripts/blend_digits_ote.py`: fixed-bias α-sweep over THREE
    baselines — greedy (0.97375), greedy+nonrule (0.97421), and
    digit-XGB (0.97468 LB-best). Same 11-point grid, same Δ ≥
    +5e-4 LB-probe gate. Auto-emits submission CSVs only when
    α > 0 AND Δ > 1e-5; flags BORDERLINE for 1e-5 ≤ Δ < 5e-4.
- OTE design rationale vs prior nulls:
  - `benchmark_te_orig` (10k original source) and `benchmark_te_oof`
    (synthetic 5-fold source) were both null because they used
    fold-level averaging — every val-fold row got the same TE per
    category. OTE produces a DIFFERENT per-row value via the K-shuffle
    cumulative noise, exposing finer category-row structure.
  - The `nonrule` lever already proved non-rule cats carry the
    NN-flip signal (LB +0.00056). OTE is the per-row analog:
    encode each cat's class-conditional probabilities directly as
    numeric features, exposing them to digit-XGB's tree splits
    without requiring the model to discover them via cat one-hots.
- Status: scaffold ready, NOT yet executed. Next step is to run
  `python scripts/xgb_dist_digits_ote.py` (~25-30 min) then
  `python scripts/blend_digits_ote.py` (~10 s).

### 2026-04-23 — OTE-XGB executed: NULL on top of LB-best digit-XGB

- Goal: run the OTE scaffold from the 2026-04-22 entry to test whether
  per-row K-shuffled target encoding lifts digit-XGB (LB 0.97468).
- Changed: `scripts/xgb_dist_digits_ote.py` runs the full pipeline; one
  pre-commit bug fix (`_all_key_specs()` was iterating `SINGLE_CATS`
  strings as characters instead of wrapping each name in a list).
  Training wall: test-OTE 208 s one-shot + 5 folds × (OTE build ~40 s
  + XGB train ~190 s) ≈ 21 min total. 48 OTE cols from 16 keys × 3
  classes, 8 shuffles, alpha=10.
- Standalone OOF (5-fold, seed=42):
  - argmax **0.96465** (digit-XGB 0.96485, −0.00020)
  - prior-reweight not reported but at the same ~0.974 level
  - **tuned log-bias 0.97375** (digit-XGB 0.97449, **−0.00074**)
  - Error count 8,888 (digit-XGB 8,846 — near-identical magnitude)
- Blend sweeps (fixed-bias, α ∈ {0, 0.025, …, 0.5}):
  ```
                                     baseline OOF   peak α   OOF at peak   Δ
  vs greedy                          0.97375        0.50     0.97443       +0.00069
  vs greedy+nonrule (prior LB best)  0.97421        0.50     0.97463       +0.00041
  vs digit-XGB (CURRENT LB BEST)     0.97449        0.00     0.97449        0.00000   ← peak is α=0
  ```
  **Null on top of digit-XGB.** Every α > 0 strictly hurts; sweep is
  monotone-negative (α=0.025: −0.00002, α=0.50: −0.00019). The lifts
  vs greedy / greedy+nonrule are against weaker baselines than the
  current LB best, so they don't translate.
- Jaccard-vs-error-count diagnostic:
  ```
                 errors   Jaccard vs digit-XGB
  OTE-XGB        8,888    0.59
  digit-XGB      8,846    —
  greedy         11,862   0.60
  greedy+nonrule 12,372   0.57
  ```
  OTE-XGB is the SIXTH model to hit the "decent orthogonality
  (0.57-0.65) but similar-or-higher error count" blend-null pattern.
  Same mechanism as MLP-v5 (12,005 errs vs greedy 8,909),
  FT-Transformer (12,634 errs), TabPFN (10,376 errs), pretrain-
  finetune MLP (12,524 errs), and LGBM-digits (8,874 errs,
  Jaccard-0.96). Digit-XGB is the anchor; nothing blends usefully
  on top of it without a fundamentally better error footprint.
- Interpretation: the public-notebook "digit + OTE" recipe does
  NOT work for us as scaffolded. Three possible failure modes:
  1. **Our OTE consumed by digit-XGB's category splits**: XGB's
     `enable_categorical=True` on raw cat cols + the 46 digit cols
     already lets the tree discover per-category per-class
     probabilities without explicit encoding. OTE columns become
     redundant re-statements of what splits already express.
  2. **Wrong key set**: we encoded the 8 standard cats + 6 pairs +
     2 rule cells. The notebook may encode digit columns themselves
     (10-card each) or cell × digit interactions. Untested.
  3. **Wrong smoothing**: alpha=10 + 8 shuffles may over-regularise
     toward the prior. Lower alpha (1, 3) would expose more
     per-category signal; fewer shuffles (1-2) would keep more
     per-row noise.
- LB budget: unchanged at 5/10 used today (no LB submission; OOF
  delta vs LB-best is 0.00000, no probe warranted).
- Current LB best unchanged: `submission_xgb_dist_digits_tuned.csv`
  at LB 0.97468.
- Artefacts committed for cross-branch reuse:
  - `scripts/artifacts/oof_xgb_dist_digits_ote.npy`
  - `scripts/artifacts/test_xgb_dist_digits_ote.npy`
  - `scripts/artifacts/xgb_dist_digits_ote_results.json`
  - `scripts/artifacts/blend_digits_ote_results.json`
  - `submissions/submission_xgb_dist_digits_ote_tuned.csv`
    (standalone tuned; strictly worse LB than digit-XGB)
  - `submissions/submission_greedy_nonrule_ote_blend.csv`
    (borderline +0.00041 vs prior LB best, auto-flagged; LB
    inferior to digit-XGB, not for submission)
- Next bet (smallest delta from current scaffold): re-run OTE with
  alpha=1 and/or n_shuffles=2 to test if over-smoothing is masking
  signal. Cheap — same ~20 min wall. Alternative: OTE on digit cols
  themselves (treat each of 46 digit positions as a 10-card
  categorical). Structurally different target — may expose patterns
  the standard-cat OTE can't.

### 2026-04-23 — OTE variants follow-up: digits-OTE → NEW LB BEST 0.97482

- Goal: test two follow-ups to the default-OTE null from earlier today:
  (1) "light" — alpha=1, shuffles=2 to check if default's alpha=10 +
  shuffles=8 over-smoothed away signal; (2) "digits" — OTE keys = 46
  surviving digit columns (10-card each) instead of 8 standard cats +
  pairs + rule keys, testing a structurally different target aligned
  with the digit-extraction LB-best lever.
- Changed: `scripts/xgb_dist_digits_ote.py` parameterised via
  `OTE_VARIANT` env var; `scripts/blend_digits_ote.py` likewise. Shared
  code path, suffix-based artifact names (`_light`, `_digits`).
- Light variant results (alpha=1, shuffles=2):
  - Standalone: argmax 0.96433, tuned 0.97390 (default tuned 0.97375,
    +0.00015). Lighter regularisation trades argmax for tuned.
  - Blend peaks: vs greedy α=0.40 → 0.97416 (+0.00042); vs
    greedy+nonrule α=0.40 → 0.97485 (+0.00064); **vs digit-XGB
    α=0.025 → 0.97454 (+0.00005 BORDERLINE)**.
  - Verdict: small improvement over default but vs-digit peak moved
    from α=0 to α=0.025 with only +0.00005 lift — within fold noise.
- **Digits variant results (46 digit cols as keys, alpha=10, shuffles=8)**:
  - Standalone: argmax **0.96520** (best of all 3 OTE variants),
    tuned 0.97415. Error count 8,840 — **FIRST OTE variant with fewer
    errors than digit-XGB** (8,879).
  - Jaccard vs digit-XGB: 0.59 (same orthogonality range as other OTE
    variants, but combined with lower error count).
  - Blend sweep vs digit-XGB: **clean unimodal curve with broad
    plateau α=0.30-0.50** at Δ = +0.00019 to +0.00028. Peak at α=0.40
    (OOF 0.97477, Δ=+0.00028). Not a single-point fluke.
- **LB PROBE (user-approved, submitted 05:11 UTC)**:
  - `submission_digit_ote_digits_blend.csv` (0.4 × digits-OTE + 0.6 ×
    digit-XGB log-blend, digit-XGB's bias).
  - **LB public = 0.97482** (new best), gap OOF → LB = **−0.00005**
    (LB slightly above OOF — excellent calibration).
  - Δ vs prior LB-best (digit-XGB 0.97468): **+0.00014**.
  - Gap to pack 0.98114: +0.00632 (was +0.00646).
  - Gap to leader 0.98219: +0.00737 (was +0.00751).
- Updated calibration ladder:
  ```
  single tuned LGBM                 0.97097 → 0.96972   gap 0.00125
  greedy 3-way log-blend            0.97375 → 0.97296   gap 0.00079
  greedy + nonrule α=0.15           0.97421 → 0.97352   gap 0.00069
  digit-XGB standalone              0.97449 → 0.97468   gap -0.00019
  **digits-OTE × digit-XGB α=0.40   0.97477 → 0.97482   gap -0.00005**  ← NEW LB BEST
  ```
- LB budget: **6/10 used today**, 4 remaining.
- Read-out: the **"encode digit columns directly with per-class target
  stats" approach adds orthogonal signal** on top of the raw digit
  features. Mechanism: each digit position has 10 unique values, and
  their per-class probability (e.g. "rows where humidity tens-digit =
  7 are 8% High" — higher than the 3.3% prior) carries signal the
  tree can't express by axis-aligned digit-value splits because each
  split only sees ONE class side at a time. OTE on digit cols gives
  the tree a 3-dim probability per row per digit position, enabling
  single-split decisions on the full per-class distribution.
- Critical heuristic confirmed: **fewer errors AND Jaccard < 0.65 is
  the right pattern for a blend lift.** Every prior OTE variant had
  the right Jaccard but more errors; every prior NN/tree had similar
  Jaccard but 16-42% MORE errors. Digits-OTE is the first to satisfy
  both, and it delivered +0.00014 LB.
- Next bets (in priority order):
  1. **Extend digit-OTE key set**: pair digit cols with cats
     (`dig_Humidity_-1 × Crop_Type`) — high-card keys may unlock more
     per-category per-digit flip signal. Low-risk; same pipeline.
  2. **Seed-bag digits-OTE**: variance-reduction on new LB-best.
     HISTORICALLY RISKY per seed-bag survey but the digit-OTE lever
     is structurally different — worth testing if 1 session budget
     allows.
  3. **Retry default-OTE or light with "digits" keys AND alpha=1**: the
     two wins (alpha=1 calibration lift + digit-key structural signal)
     may compound.
- Artefacts committed: `oof_xgb_dist_digits_ote_{light,digits}.npy`,
  `test_xgb_dist_digits_ote_{light,digits}.npy`, three result JSONs,
  all 6 submissions (standalones + blends) on `claude/plan-next-steps-I32rC`.

### 2026-04-23 — digits_light variant (compound light + digits): null

- Goal: compound the two "winning" OTE ingredients (alpha=1/shuffles=2
  calibration lift from `light`; digit-column keys from `digits`) into
  a single variant. Hypothesis: if the two wins are independent, the
  combined variant should beat digits alone.
- Changed: added `digits_light` branch in `OTE_VARIANT` env var
  handling. 46 single-key OTEs on digit cols with alpha=1, shuffles=2.
- Results (OOF 5-fold, seed=42):
  - Standalone: argmax 0.96486 (digits 0.96520, light 0.96433),
    tuned **0.97371** — **worst of all digit-key variants**.
  - Blend vs digit-XGB (LB-best 0.97449): peak α=0.40 → 0.97469
    (Δ = +0.00020). Strictly below digits variant's +0.00028 at
    same α.
- Diagnosis: alpha=1 on 10-card keys overweights raw observations,
  making per-row encodings noisier than useful. Digit keys need
  alpha=10 smoothing to produce stable per-digit-value class probs;
  alpha=1 converts the OTE from "smoothed per-digit class
  distribution" to "near-unsmoothed per-row neighbor voting",
  which has too much variance. Alpha=1 only helped the CAT-key
  variant because its higher-cardinality keys (50-300 unique values)
  needed less smoothing.
- Rule (generalisable): **OTE alpha should scale with key
  cardinality**. Rough rule of thumb: alpha ≈ n_unique_keys / 10.
  For 10-card digit keys, alpha=1 is too low; for 80-card cat pairs,
  alpha=8 is about right. Our default alpha=10 happens to be
  near-optimal for both cat-pair and digit-key cardinalities in
  this problem.
- No LB probe warranted (null on top of LB-best).
- Next: digits_pairs variant (46 digit singles + 46 (digit ×
  Crop_Type) pairs) — running in background.

## Hypothesis board

- **Current best (LB)**: digits-OTE × digit-XGB log-blend at α=0.40
  → OOF 0.97477, **LB 0.97482** (+0.00014 over prior, −0.00005
  OOF→LB gap). Submission on disk:
  `submissions/submission_digit_ote_digits_blend.csv`. Pack 0.98114
  is +0.00632 above; leader 0.98219 is +0.00737 above. LB budget
  today: 4 remaining (6/10 used: 1 seed-bag null + 2 HP-tuning
  nulls + 2 digit-extraction probes + 1 digits-OTE blend probe).

  Second-best: XGB-dist + digits standalone, tuned log-bias →
  OOF 0.97449, LB 0.97468. Submission:
  `submissions/submission_xgb_dist_digits_tuned.csv`.

  Third-best: greedy + xgb-nonrule log-blend at α=0.15
  → OOF 0.97421, LB 0.97352. Submission:
  `submissions/submission_greedy_nonrule_blend.csv`.

  Fourth-best: greedy log-blend `hybrid_v3(0.45) + routed_v3(0.40) +
  spec_678(0.15)` → OOF 0.97375, LB 0.97296. Submission:
  `submissions/submission_blend_greedy_w045_040_015.csv`.

### Anchor-row ideas (from 2026-04-21 v6 null + refined routing heuristic)

The v6 {0,1,2,5} null (−0.00012) revealed that single-class-pure rows
adjacent to a class boundary act as training anchors for the model's
boundary calibration. Removing score-5 Medium rows destabilized the
Medium↔High boundary on {6,7,8} (Medium→High errors +703 vs v3).
Opens five follow-up ideas:

  **A1. Decoupled routing (v7): train on all, route inference only.**
  If v6's loss was purely the training-side anchor removal, training
  vanilla XGB on all 630k rows and routing {0,1,2,5} only at inference
  should recover v3's OOF. Cheap, direct test. **Launched as
  `scripts/xgb_dist_routed_v7.py`** (in progress).

  **A2. Upweight anchor rows instead of removing.** Give clean-class
  rows near boundaries `sample_weight > 1` (e.g. score-5 at 1.5×,
  score-9 at 1.3×). Strengthens the Medium anchor for {6,7,8}
  calibration. One-line XGB param change; ~15 min run.

  **A3. Soft routing with per-score α.** Replace hard override with
  `pred = α(score) · rule_onehot + (1−α) · XGB_softmax`. Tune α per
  score on OOF: α≈0.98 for {0,1,2,9}, α≈0.85 for {3,5}, α=0 for
  {6,7,8}. Keeps XGB's probability distribution while rewarding
  rule-reliable scores. ~20 min with α sweep.

  **A4. Per-score log-bias tuning.** Current bias is 3 global params;
  tune 30 (10 score bins × 3 classes). Lets the decision rule
  account for score-specific error patterns. Overfitting risk —
  needs nested CV. Expected +0.0003–0.0008.

  **A5. Explicit boundary-row oversampling.** Duplicate rows at class
  boundaries: 2× score-3 Medium rows, 2× score-6 High rows. Forces
  XGB to attend to exactly the rows the rule gets wrong.

- **Open** (ranked by expected ROI after the 2026-04-22 NN-lever
  closure — all remaining own-pipeline bets are expected ≤ +0.0005 LB):

  1. **Per-score log-bias tuning** (30 params = 10 score bins × 3
     classes vs 3 global). Nested CV to avoid overfit; high risk.
     Expected +0.0003–0.0008. ~30 min.
  2. **LGBM leaf-embedding MLP** (tree-distilled features). Train
     LGBM once, extract per-tree leaf indices as categorical features
     for a NumEmb+MLP. Different from v5-v9 because the NN sees
     tree-discovered rule knowledge directly, not raw features.
     Well-documented to lift tabular NNs +0.003–0.008 on problems
     like this. Expected here: +0.0005–0.002 LB if it breaks the NN
     plateau. ~45 min on Kaggle GPU.
  3. **Blend greedy-winner with a distinct-anchor blend.** Our
     greedy and main's `hybrid_lgbmxgb_blend` both anchor on
     `xgb_hybrid_v3` (cross-lineage pairwise null). A blend whose
     anchor is the 5-seed LGBM bag would be structurally different
     — could add +0.00005–0.00015. Contingent on regenerating an
     anchor-free blend.

- **Do not propose public-CSV / other-people's-submission blending.**
  See the top-of-file rule. People have reached 0.98+ without
  blending others' results; the open question is which own-pipeline
  lever they used, not whether to mirror the public-notebook blend.

- **Ruled out this session** (2026-04-21 soft-blend + DQ experiments):
  - Hard-vote plurality/Borda/veto across top submissions (0.99+
    pairwise agreement → <0.005 ceiling, and plain plurality
    demotes the rare class; only "High-supermajority" and rule-
    deferred are geometrically aligned with macro-recall but still
    speculative without OOF gating).
  - Logistic meta-stacker on (P_hv3 + P_routed + P_dgp + P_xgbdist)
    with class_weight=balanced: 0.97348, below greedy log-blend.
    Components too correlated to let 12-feature LR add signal.
  - Cross-lineage blending with main's `hybrid_lgbmxgb_blend`:
    pairwise picks w_ours=0.95 → 0.97376 (null vs our greedy
    0.97375). Shared anchor on hybrid_v3 — two blends that share
    the dominant component don't compound.
  - **Heavy-weight original-dataset augmentation** (w=20 per row):
    −0.00026 on xgb_dist. Medium recall drops −0.00066 on argmax.
    Rule-perfect external data biases the model AWAY from the
    deterministic NN flips that generalize to LB. Safe weight is
    1× per row (prior +0.00027 result); anything heavier hurts.
  - **(target × dgp_score) stratified CV**: tuned OOF unchanged
    (0.97278 both ways); fold variance drops σ ~0.0008 → ~0.0002
    but means nothing for the global OOF. At 630k rows, default
    StratifiedKFold(shuffle=True) is already well-balanced.
- **Confirmed**:
  - Default `argmax` is suboptimal under balanced accuracy when classes
    are imbalanced → prior-reweight + coord-ascent log-bias moves OOF
    from 0.96135 → 0.97097 (+0.0096). Keep this as the decision rule
    for every subsequent model.
- **Ruled out**:
  - **TabPFN v2 as blend leg (2026-04-22)** — tabular foundation
    model (tabpfn==2.2.1, pre-license), 1500-row stratified
    subsample per fold, 5-fold OOF. Standalone tuned 0.96209
    (below XGB-dist 0.9726). Fold-1 Jaccard 0.85 passed abort
    gate but in "warn" band. Full-OOF Jaccard 0.81, TabPFN has
    10,376 errors vs LB-best's 8,891 (+16.7 %). Fixed-bias blend
    sweep vs LB-best peaks at α=0.025, Δ=+0.00002 — within fold
    noise, strictly monotone-negative past. High-class recall
    (0.92) was the weakest leg — 1500-row context = ~50 High
    examples/fold. GPU SUBSAMPLE=10k might lift standalone but
    the Jaccard/magnitude pattern predicts the blend remains null.
    Artefacts `oof_tabpfn.npy`, `test_tabpfn.npy` committed.
    Rule: **in-context foundation models share the same NN-on-
    tabular blend-null failure mode as from-scratch MLPs and
    transformers at this feature set** — the error-magnitude
    mismatch defeats the blend regardless of architecture family
    or training regime. Closes the NN lever definitively.
  - **Frank-Hall ordinal decomposition (2026-04-22)** — two binary
    XGB heads `P(y>=Medium)` + `P(y>=High)` on 43-feature dist set,
    Frank-Hall recomposition with monotone clip. AUC 0.998/0.999 per
    head. Standalone tuned 0.97354 (ties xgb_hybrid_v3). Blend vs
    LB-best greedy+nonrule peaks at **α=0.4, Δ=+0.00009** — inside
    fold noise and below the +0.0002 LB-probe threshold. The
    decomposition makes different error trades (Medium recall +0.005,
    High recall −0.007) but the direction is wrong under macro-recall
    because High has 3× leverage. Artefacts `oof_xgb_corn.npy`,
    `test_xgb_corn.npy` committed for cross-branch reuse. Rule:
    **ordinal-binary decompositions on the DGP-enriched feature set
    plateau at the same OOF band as multi:softprob** — the
    information ceiling is set by the feature set, not the objective.
  - **Equal-weight z-score fusion of water-balance axes** (H2) is
    worse than the single-feature Soil_Moisture rule (H1). Any future
    hand-weighted score needs per-axis weights proportional to
    informativeness, not uniform.
  - **Large-capacity tabular NN (5 MLP variants, 2026-04-22)** — the
    NN lever hypothesis that sat at the top of the Open bets list is
    now closed. Five variants run on Kaggle GPU: v5 full features
    [768,512,384,256] 1M params / v6 13 non-rule features
    [256,192,128,96] 150k / v7 top-3 numerics [128,96,64] 15k / v8
    specialist {6,7,8} [384,256,192,128] 200k on 56k rows / v9
    training-data-routed (exclude score {0,1,2}) [768,512,384,256]
    1M on 359k rows. All standalone + blend-null across prob and
    log space vs both greedy and greedy+nonrule baselines. The
    plateau at ~0.965 for full-feature variants is insensitive to
    20× capacity span, feature slicing, training-data policy, and
    domain specialization. v9 falsified the "easy-row gradient
    domination" hypothesis (MLP with Balanced Softmax + CE already
    handles imbalance; filtering 271k trivial rows has no effect).
    v8 under-performed XGB's axis splits on its own specialist
    domain (0.936 vs 0.952 xgb_spec_678). Implication: not a
    capacity-or-optimizer problem, an information-bottleneck problem
    no feature-independent NN can route around. Any further NN
    capacity scaling (FT-Transformer, tabular-ResNet) is unlikely
    to break the pattern.
  - **Seed-bag greedy at LB (2026-04-22)** — 2-seed bag of routed +
    spec (seeds 42+7), rebuilt hybrid, rebuilt greedy. OOF 0.97385
    (+0.00010 vs single-seed 0.97375), but LB 0.97284 (−0.00012 vs
    single-seed LB 0.97296). OOF→LB gap widened 0.00079 → 0.00101.
    Diagnosis: XGB at our hyperparams is near-deterministic across
    seeds (per-seed spread ~0.00010, below 1-fold-std σ=0.00088).
    Bagging buys nothing when base variance is already below noise.
    New rule: **below-1-fold-std OOF lift from near-deterministic
    bags = non-signal on LB.**
  - **Spec on score {3} (2026-04-22)** — 102k rows (95% Low / 5%
    Medium / 0% High). Spec-domain bal_acc 0.5040 vs rule's 0.5
    floor (null). Hybrid override −0.00011 vs greedy; soft-blend
    sweep monotone negative. Reconfirms **specialist 20–80%
    minority-mass heuristic**: 95/5 with 0% of one class is
    below threshold, and Low/Medium per-class specialists from
    main's session had the same failure.
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
  - **128-cell empirical Bayes as a stacking feature** — standalone
    OOF 0.96339 (vs rule 0.96097, LGBM-dist 0.97266). Prob-space
    blend with LGBM-dist is monotonic in α → pure LGBM wins; EB
    adds zero orthogonal signal because LGBM already splits on the
    same 6 rule features and recovers cell-level class
    distributions via interaction splits. Same lesson as the
    hand-engineered domain features ruled out earlier. Cell
    probabilities only help if paired with a model that doesn't
    already see the 6 rule cols.
  - **Per-score expert LGBMs** (#8) — 0.97149 tuned OOF, below
    both baseline LGBM (0.97097 by +0.00052, within fold noise)
    and LGBM-dist (0.97266 by −0.00117). Partitioning train into
    10 score bins and training binary/3-class specialists per bin
    loses more data per fit than specialisation buys back. LGBM at
    127 leaves already splits on (score, stage) internally, so
    "explicit experts" is redundant.
  - **Noise-inversion head** (#3) — 0.96768 tuned OOF, **−0.00329
    vs baseline**. Three per-rule-label LGBM heads (Low / Medium /
    High routed by rule(x)), with rule cols removed so each head
    specialises on P(y_obs | rule, x). The rule=High head is
    data-starved (~21k rows) and the Low-vs-Medium head trains to
    a near-prior flat vector. Dropping rule cols removes distance
    information the heads desperately need.
  - **Naive GCE loss** (#5, q=0.7) — 0.96500 tuned OOF. Custom
    multiclass objective hits `best_iter=1` on every fold: the
    grad/hess scaling doesn't let LGBM progress past the first
    round. Result is essentially a uniform-prob prediction rescued
    by an aggressive log-bias. Real GCE requires debug on the
    gradient scale and learning-rate; parked until that's done.
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
  - **Balanced-ensemble methods (BalancedRandomForest, EasyEnsemble,
    RUSBoost) on DGP features.** All three land below LGBM+DGP
    0.97271 tuned: Easy 0.96932, RUSBoost 0.96666, BRF 0.96535.
    Pairwise and 3-way blends with LGBM+DGP give Δ ≤ +0.00008, well
    inside the ~0.0009 fold-std noise band; BRF gets zero weight in
    every blend. These methods produce pre-balanced probabilities
    (inter-class bias deltas 0.03–0.14) so log-bias has nothing to
    correct — they and LGBM+log-bias are picking the same balanced-
    accuracy operating point via different mechanisms. **Per-tree
    majority undersampling is not a distinct lever from post-hoc
    log-bias at this feature set.** Rule: balanced-ensemble wrappers
    are not a useful diversity source when log-bias tuning is already
    in the pipeline.
  - **MLP / tabular NN** (plateaued 2026-04-21, details in `REPORT.md`
    and `LEARNINGS.md` from main commit e889f0c; implementation code
    on branch `claude/improve-balanced-accuracy-v1UtX`, not merged).
    3-layer MLP (256→128→64, ~50 k params, embedded cats, 26 DGP-
    enriched numerics): v1 plain CE + log-bias = 0.96437; v3 Balanced
    Softmax (Menon 2021) = 0.96596; v4 LDAM-DRW killed at fold 1
    (effective-number class weights degenerate at n_c ≫ 10 k).
    **Blend with LGBM+DGP: geometric w=0.15 → 0.97276** vs LGBM+DGP
    0.97271 — Δ = +0.00005, well below fold-std noise. Third
    independent blend null (MNLogit, balanced-ensemble, MLP). New
    rule: **blending requires per-row error orthogonality, not just
    standalone OOF ≥ 0.965** (log in `LEARNINGS.md`). MLP is
    capacity-bound on this rule-structured feature set at our
    training budget; revisit only with a significantly larger
    architecture or a structural prior matching the rule (e.g.
    additive / monotone net).
  - **Rule × non-rule pairwise FE** (`scripts/seed_bag_dist_fe.py`,
    `benchmark_xgb_dist_fe.py`, `blend_lgbm_xgb_dist_fe.py`). 8 new
    cols on top of the 43-feature dist set (`humidity_x_sm`,
    `humidity_x_sm_dist`, `prev_irrig_x_rf`, `prev_irrig_x_rf_dist`,
    `prev_irrig_minus_rf`, `vpd_proxy`, `ec_x_sm`,
    `field_area_x_score`) targeting the non-rule features with
    significant Cohen's d on flipped rows (2026-04-21 EDA). OOF:
    LGBM-FE bag 0.97270 (Δ = −0.00019), XGB-FE 0.97313 (Δ = +0.00009),
    blend log-α=0.05 → 0.97320 (Δ = **−0.00007** vs non-FE blend
    0.97327). All deltas are well inside the fold-std noise band
    (~0.00088). Optimal blend weight collapsed from α=0.45 to
    α=0.05 — the added LGBM features didn't just fail to help, they
    also broke LGBM's complementarity with XGB. Third tree-FE null
    in a row (water-balance cols, 128-cell empirical Bayes, pairwise
    rule×non-rule). Rule: trees at 127-leaves / max_depth=7 already
    find pairwise interactions internally; engineered products add
    no new signal regardless of how physically motivated they are.
  - **Extended score-routing to {0, 1, 2, 9}**
    (`scripts/xgb_dist_routed_v2.py`): tuned OOF 0.97319 vs v1
    `{1,2}` at 0.97333. Adding score 9 to routing removes 3,237
    High rows from training — 15 % of the 21 k total High pool.
    Since High is the rare class, losing this many training
    examples hurts High calibration more than the marginal
    rule-routing gain (99.938 % rule accuracy on score 9). Rule:
    **don't route a score to the rule if removing it strips >10 %
    of any class's training pool**. Safe routing set: {1, 2} or
    {0, 1, 2} (tied at 0.97333).
  - **Per-rule-class specialists (Low-spec on 0-3, Medium-spec on
    4-6, High-spec on 7-9)**
    (`scripts/xgb_per_class_specialists.py`). Fused per-row-routed
    OOF = 0.97226 (Δ = −0.00126 vs hybrid 0.97352). Low-domain is
    98 % Low and Medium-domain is 98.5 % Medium, so their
    specialists collapse into "predict majority" with bal_acc ~0.5.
    Only the High-spec (92 %/8 %) made real use of its small
    domain. Rule: **specialize on sub-domains with 20–80 % minority
    class**, not sub-domains dominated by one class. The {6,7,8}
    specialist works for exactly this reason (69 % Medium / 31 %
    High).
  - **Hinge-loss / max-margin tie-breaker over integer separating
    rules** (`scripts/enumerate_integer_models.py`, per discussion
    [692754](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/692754)).
    CP enumeration finds 743 integer models with `|w|≤10, θ≤10` that
    achieve 100 % train_acc on the 10k original. Hinge loss on 10k
    spans 0.0000 → 0.2981. **All 743 produce identical predictions on
    630k synthetic** (agreement 1.0000, bal_acc 0.96097). Cell-labeling
    over the 2⁵ × 4 = 128 discrete cells is fully determined by the
    10k, so any separating linear classifier gives the same
    decision-region map. Ceiling for this representation is 0.96097 —
    the same as cdeotte's rule, the SVM, and our existing DGP rule.
    Residual signal lives in within-cell continuous variation, not
    in weight choice. Related rule: **don't ensemble over linearly
    equivalent models with identical argmax — scale ambiguity ≠
    diversity.**
  - **Per-cell logistic regression on within-cell continuous
    features** (`scripts/per_cell_lr.py`, `per_cell_lr_blend_rule.py`).
    128-cell LR on 7 non-rule continuous features. With
    `class_weight='balanced'`: standalone 0.73082 (catastrophic 196k
    false positives). Without: 0.96280 tuned standalone (on par with
    EB-cell 0.96339), but rule ⊗ LR blend tops at 0.96286 and
    recovers only 3.86% of rule-wrong rows. Within-cell continuous
    features **do not carry orthogonal signal at linear capacity** —
    same lesson as the 128-cell empirical-Bayes null. MLP unlikely
    to rescue it: same feature set, same per-cell data, bottleneck
    is information not model capacity.
  - **Augmenting spec-{6,7,8} training with original-{6,7,8} rows**
    (`scripts/xgb_specialist_678_aug.py`). 982 rule-clean rows from
    the 10k original added to the specialist's training pool in two
    variants. Standalone spec-domain OOF: w=1.0 → 0.95149 (Δ=−0.00049
    vs baseline 0.95198), w=0.3 → 0.95142. **Hybrid-level**: w=1.0 →
    0.97323, w=0.3 → 0.97326, both below both non-aug hybrid 0.97352
    AND pure main 0.97332. Rule: **don't augment specialist training
    with clean data if the specialist's purpose is to deviate from a
    clean predictor** — the 982 rule-correct rows pull the decision
    boundary toward the rule, eroding the flip-recovery edge that
    is the specialist's only reason to exist.
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
  - **DGP features transfer cleanly to the LB.** LGBM+DGP tuned OOF
    0.97271 → LB public 0.97137, gap 0.00134 (within the +0.00010
    the baseline submission's 0.00125 gap set). +0.00165 LB lift
    vs baseline LGBM (0.96972). The OOF→LB calibration is honest
    for DGP-enriched feature sets.
  - **Model-family diversity (LGBM × XGBoost) beats seed bagging
    ~1.5×.** On the 43-feature LGBM-dist feature set, the
    progression single LGBM (0.97266) → 5-seed LGBM bag (0.97289,
    +0.00023) → LGBM-bag × XGB log-blend α=0.45 (0.97327, +0.00038)
    shows model-family blending stacks cleanly on top of seed
    bagging and gives ~1.5× the delta for the same compute budget.
    XGB beats both LGBM standalone and the LGBM bag at every
    interior α in prob and log space — structurally clean lift,
    not a single-point fluke. **LB public 0.97170** (+0.00033 vs
    LGBM+DGP's 0.97137), confirming the OOF lift transfers. New
    rule for this feature set: **LGBM ⊗ XGB is the default
    decision rule, not plain LGBM.**
  - **Score-routing to the rule is net-positive when the class the
    rule predicts is abundant in the non-routed training set.**
    Routing scores {1, 2} (237 k rows, 99.7 % Low rule-accuracy)
    moves XGB-dist from 0.97304 → 0.97333 (+0.00029). Routing
    {0, 1, 2} ties (adds score 0 = 33 k Low rows with 0 errors,
    no effect since Low is already over-represented). Routing
    {0, 1, 2, 9} underperforms by 0.00014 because removing score 9
    strips 3.2 k High rows (15 % of the entire High training pool)
    from XGB's training set; since High is the rare class, this
    hurts High-class calibration more than rule-routing gains.
    Rule: **only route if (a) rule ≥ 99.5 % on the score AND (b)
    the class the rule predicts is over-represented in the
    remaining training set**.
  - **Specialist-on-{6,7,8} + routed main is the current best
    architecture: OOF 0.97352 / LB 0.97271** (routing {0,1,2}
    variant, narrowest OOF→LB gap seen at 0.00081). The {6,7,8}
    domain (56 k rows, 69 % Medium / 31 % High) has ideal class
    ambiguity — a specialist XGB beats the main XGB on this domain
    by +0.00109 bal_acc, and overriding main's predictions with the
    specialist's on these rows lifts global tuned OOF by +0.00019.
    Rule: **target specialists at sub-domains with 20–80 % minority
    class**, not uniform-class sub-domains.
  - **Rule-route even at OOF-ties when rule accuracy is ≥ 99.99%**.
    The {0,1,2} vs {1,2} routing variants tied on OOF (both 0.97352)
    because XGB trained on score-0 rows (100 % Low) learns the same
    Low prediction the rule makes. On the hidden test set, however,
    XGB can misfire on OOD score-0 rows while the rule never does —
    the {0,1,2} variant pulled +0.00047 LB over {1,2} at identical
    OOF. Rule: **when a deterministic predictor is provably correct
    on a score, prefer it even at OOF parity**; it reduces hidden-
    split variance by removing a learned model's failure modes.
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

### 2026-04-22 — TE-continuous-regression (null, definitive)

- Goal: test the user-proposed reframing — avoid discrete residual
  zero-inflation by regressing onto continuous per-class TE target
  values from the 10k rule-perfect original. 3 independent XGB
  `reg:squarederror` boosters on 43-col dist features, TE target
  keyed by (Crop_Type, Soil_Type, Season, Region,
  Crop_Growth_Stage, dgp_score) with Bayesian shrinkage m=30 toward
  per-score prior. 5-fold stratified (seed=42) to align with all
  other OOFs. Fixed-bias log-blend sweep into (a) greedy alone
  (OOF 0.97375) and (b) greedy + xgb_nonrule@0.15 LB-best
  (OOF 0.97421).
- Changed: `scripts/te_targets.py` (TE matrix build, 5522 unique
  cells in original, median 1 row/cell, 27 % fallback-to-score-
  prior on synthetic train and test), `scripts/te_xgb_regression.py`
  (3 boosters/fold, ~13s/fold total wall 69s), `scripts/blend_te_reg.py`
  (11-point grid over both baselines). Artefacts:
  `oof_xgb_te_reg.npy`, `test_xgb_te_reg.npy`,
  `te_xgb_regression_results.json`, `blend_te_reg_results.json`.
- Results (OOF tuned bal_acc, fixed greedy bias):
  ```
  TE-reg standalone argmax                   0.96097  (== rule ceiling)

  vs greedy (base 0.97375):
    alpha=0.000   0.97375   peak
    alpha=0.025   0.97357  -0.00017
    alpha=0.050   0.97325  -0.00050
    alpha=0.400   0.96219  -0.01155   monotone negative

  vs LB-best greedy+nonrule@0.15 (base 0.97421):
    alpha=0.000   0.97421   peak
    alpha=0.025   0.97410  -0.00012
    alpha=0.050   0.97392  -0.00030
    alpha=0.400   0.96169  -0.01253   monotone negative
  ```
- **Diagnostic** (the decisive bit):
  ```
  TE-reg argmax errors = 10,304  (EXACTLY = rule's 10,304 flipped rows)
  greedy  argmax errors =  8,909
  argmax agreement rate (TE-reg vs greedy) = 99.70 %
  rows where they disagree                 =  1,863
    TE-reg right, greedy wrong             =    234
    greedy right, TE-reg wrong             =  1,629
    both wrong (different answers)         =      0
  net TE-reg wins - losses in disagreement = -1,395
  ```
- **Mechanism** (generalisable): the 10k original is rule-perfect
  by construction. Any predictor trained to reproduce original-
  dataset per-class distributions — TE lookup, XGB regression on
  TE targets, empirical Bayes, per-cell LR — converges to a rule-
  equivalent predictor (10,304 errors on the identical rows,
  cell-level). No key granularity or shrinkage tweak changes this:
  the rule IS the optimal predictor of the original dataset's
  labels, and TE-from-original inherits that ceiling exactly.
- **Implication** — **three rule-related levers are now provably
  redundant**: (i) TE-from-original features inside a tree (already
  null in `benchmark_te_orig`, +0.00004), (ii) per-cell empirical
  Bayes on rule cells (already null in `empirical_bayes_cell`), and
  (iii) this continuous regression reformulation. They all produce
  predictors at argmax-equivalence to the rule, which greedy (with
  tuned log-bias) has already transcended by pushing boundary rows
  in the direction that improves macro-recall. Any positive blend
  weight drags the decision back toward the rule's operating point,
  hurting OOF monotonically.
- **New LEARNINGS rule**: "The 10k rule-perfect original is a
  saturation source — any predictor that consumes original labels
  as ground truth (TE, EB, distillation of original, NN-on-original)
  is bounded by the rule's argmax-equivalence class. Use the
  original only for **features that describe marginal
  distributions** (e.g., `Crop_Type × Region` frequencies as an
  auxiliary XGB input), never as **labels**."
- No LB submission (both sweeps strictly negative; deep below the
  +0.0005 LB-probe threshold). LB budget unchanged at 1/10 used
  today.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.

### 2026-04-22 — TE-continuous-regression OOF variant (null + theorem)

- Goal: the user-proposed follow-up to the original-source TE null —
  swap the TE source from the 10k rule-perfect original to synthetic
  train labels (which carry the host's NN flip signal), with
  leave-one-fold-out leak prevention. Same 5 cats × dgp_score key,
  m=30 shrinkage, same 5-fold split (seed=42).
- Changed: `scripts/te_targets_oof.py` (new, OOF TE matrix from
  synthetic, ~16s); `scripts/te_xgb_regression.py` and
  `scripts/blend_te_reg.py` parameterized via `TE_VARIANT={orig,oof}`
  env var (suffix-based filenames; orig artefacts untouched).
  Run: `TE_VARIANT=oof python3 scripts/te_xgb_regression.py` (~5 min,
  4× longer than orig because synthetic TE is a less-smooth target),
  then `TE_VARIANT=oof python3 scripts/blend_te_reg.py` (~10s).
- Density (orig vs oof source):
  ```
                        cells   median rows/cell   synth hit-rate
  orig (10k)             5522   1                 73 %  (27 % score-fallback)
  oof  (5×504k LOFO)   ~10870   ~50               99.8 % (0.2 % score-fallback)
  ```
- Results (OOF tuned bal_acc, fixed greedy bias):
  ```
  orig-TE standalone argmax       0.96097
  oof-TE  standalone argmax       0.96097   (identical)

  vs greedy (base 0.97375):
    orig peak  alpha=0.000  +0.00000   monotone negative onward
    oof  peak  alpha=0.025  +0.00000   then negative

  vs LB-best greedy+nonrule@0.15 (base 0.97421):
    orig peak  alpha=0.000  +0.00000   monotone negative onward
    oof  peak  alpha=0.075  +0.00006   (~8× below 0.0005 LB-probe gate)
  ```
- **Cross-variant diagnostic** (orig vs oof TE-regression OOF):
  ```
  argmax agreement                                 1.00000
  per-row error count                              10,304 (both, identical rows)
  per-class probability L1 delta (oof - orig)      Low 0.010, Med 0.016, High 0.005
  argmax disagreement w/ greedy (oof)              1,863 rows  (== orig)
    oof right, greedy wrong                        234        (== orig)
    greedy right, oof wrong                        1,629      (== orig)
    net oof win                                    -1,395     (== orig)
  ```
- **The argmax-equivalence theorem**: regardless of TE source
  (rule-perfect 10k OR NN-flipped 630k synthetic), the resulting
  per-class probability target has the same per-(cat-tuple x score)
  cell-majority class — because the synthetic NN flips are
  *within-cell minority* events (already established 2026-04-21:
  only 1/64 cells has a synthetic majority different from the rule,
  covering 308 rows / 0.05 %). XGB regressed onto either target
  reproduces the cell-majority at argmax, which IS the rule. The
  flip signal is structurally invisible to a multinomial soft-prob
  target keyed by features that determine the cell.
- **Implication / new LEARNINGS rule**: "TE-as-regression-target
  predictors converge to a rule-equivalent argmax even when sourced
  from flip-rich synthetic data, because the flips are within-cell
  minority and the soft-prob target preserves cell-majority. To
  escape this ceiling, the target must be either (a) the per-row
  flip indicator (binary, not per-class soft-prob) so the model
  learns to OVERRIDE cell-majority, or (b) computed at a
  granularity finer than the cells the rule itself splits on
  (e.g. continuous-bin × cat-tuple subdivisions of each rule
  cell)."
- The +0.00006 lift on LB-best is calibration drift on 1-2 % of
  cell probs where synthetic LOFO has dense data and the original
  has 1-2 rows. Far below LB-probe threshold (+0.0005) and ~15×
  below fold-std noise (~0.00088). No submission warranted.
- LB budget unchanged at 1/10 used today.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.

### 2026-04-22 — TE-regression follow-ups A & B (both null, theorem reinforced)

Closed the two open paths from the argmax-equivalence theorem:
(A) soft flip-correction composite, (B) sub-cell granularity TE.

- **(A) Soft flip-correction**
  (`scripts/flip_correction_blend.py`) — P_flip binary XGB + 3-class
  XGB on flipped rows only, composite prob
  `P(y) = P_flip * P_dir + (1 - P_flip) * onehot(rule)`.
  Results: OOF P_flip AUC = 0.9047; composite standalone argmax =
  0.96146 (essentially rule ceiling, since P_flip is small on 98%
  of rows); blends vs greedy and LB-best both peak at alpha=0
  (i.e., no lift — monotone negative from alpha=0.025).
- **(B) Sub-cell TE target**
  (`scripts/te_targets_subcell.py`, ran via `TE_VARIANT=subcell`
  through existing infra) — TE keyed by
  `(rule_cell × Humidity_bin × Crop_Type)` = 1920 sub-cells, m=15
  shrinkage to per-rule-cell prior, OOF source. Diagnostic: only
  **67/1915 sub-cells (3.5%) have majority class different from
  their parent rule-cell's majority, covering 754 rows (0.12% of
  train)** — the sub-cell majority deviation available is
  structurally tiny. Results: standalone argmax 0.95927 (below
  rule ceiling — XGB makes MORE wrong sub-cell decisions than
  right ones on the 0.12% available flip substrate); vs greedy
  peak +0.00001 at alpha=0.10; vs LB-best peak +0.00003 at
  alpha=0.05. Both well inside fold-std noise.
- **Combined ladder** (all four TE-regression family variants now
  closed, OOF delta vs LB-best 0.97421, fixed bias):
  ```
  orig TE           monotone negative, peak alpha=0 (+0.00000)
  oof  TE           peak alpha=0.075 (+0.00006)
  subcell TE        peak alpha=0.050 (+0.00003)
  flip-correction   monotone negative, peak alpha=0 (+0.00000)
  ```
  All < 0.0005 LB-probe gate; all < fold-std noise (~0.00088).
- **Theorem reinforced**: argmax-equivalence to the rule is a
  structural property of this problem that no variant within the
  "cell-partition + soft-prob target" family escapes. The only
  thing that matters for breaking above 0.97421 OOF is **a model
  whose errors are orthogonal to greedy's** (our current-best
  `xgb_nonrule` is the working example) — not a model that more
  cleverly reconstructs the rule's decision surface.
- **Files** (all on branch `claude/target-encode-xgb-residuals-F0z0S`):
  - `scripts/te_targets_subcell.py`, `scripts/flip_correction_blend.py`
  - `scripts/te_xgb_regression.py` + `scripts/blend_te_reg.py` now
    take `TE_VARIANT ∈ {orig, oof, subcell}` env var
  - artefacts `oof_pflip.npy`, `oof_flip_correction.npy`,
    `oof_xgb_te_reg_subcell.npy` + test counterparts + JSONs
- LB budget unchanged at 1/10 used today.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.

### 2026-04-22 — Session A: format check + threshold re-fit + monotone OvR (3 nulls, all informative)

- Goal: run the three cheap Session-A checks proposed in the
  brainstorm — (Step 0) confirm submission format has no silent
  bug, (Step 1) re-fit the 4 DGP rule thresholds on synthetic
  (not just on the 10k original), (Step 2) test monotone
  constraints on XGB-dist via one-vs-rest per-class binary heads.
- Changed: `scripts/refit_thresholds_synthetic.py` (precomputed
  feature vectors + int-label fast bal_acc via confusion-matrix
  diagonal — ~5ms per eval, 1715-config joint grid in ~30 s);
  `scripts/xgb_dist_monotone.py` (3 binary XGBs per fold with
  per-class monotone_constraints tuples: Low head has 18
  decreasing/increasing features, High head has 18 mirror,
  Medium head fully unconstrained because the class is
  non-monotonic in score); `scripts/blend_monotone.py`
  (fixed-bias log-blend sweep onto greedy and onto LB-best
  greedy+nonrule). Artefacts:
  `refit_thresholds_synthetic.json`, `oof_xgb_dist_monotone.npy`,
  `test_xgb_dist_monotone.npy`, `xgb_dist_monotone_results.json`,
  `blend_monotone_results.json`.

- **Step 0 — submission format sanity (clean, rules out silent bug)**:
  - `submission_greedy_nonrule_blend.csv` vs `data/sample_submission.csv`:
    270 000 rows match, `id` ordering identical, column names
    `[id, Irrigation_Need]` match, label casing `Low/Medium/High`
    matches. Label distribution on our best sub: 159 814 Low /
    99 085 Medium / 11 101 High (59.2 / 36.7 / 4.1 %), close to
    train prior (58.7 / 37.9 / 3.3 %) with the expected log-bias
    push on High. No silent bug to chase. This was the cheapest
    and most important sanity check — confirmed before any
    modeling work.

- **Step 1 — threshold re-fit (confirms rule thresholds are correct)**:
  - Baseline (25, 300, 30, 10): bal_acc 0.960973, raw 0.983644.
  - 1D per-feature sweeps:
      `t_soil`: best 25.00, Δ bal = −0.000003 (baseline optimal)
      `t_rain`: best 299.0, Δ bal = +0.000000 (baseline optimal)
      `t_temp`: best 29.9, Δ bal = +0.000338 (small drift)
      `t_wind`: best 10.0, Δ bal = +0.000051 (baseline optimal)
  - All 1D winners combined (25, 299, 29.9, 10): bal 0.961360
    (Δ = +0.000386), raw 0.983187 (Δ = **−0.000457**).
  - 1715-config joint grid around 1D winners confirms the 1D
    solution — no multi-axis interaction found.
  - **Verdict: thresholds are correct, Δ is a log-bias artifact.**
    If the true generator threshold were 29.9 instead of 30.0,
    raw accuracy would INCREASE (the rule would classify more
    rows correctly). Instead raw acc DROPS by 0.000457. The
    bal_acc gain comes from the small threshold shift
    redistributing ~200 boundary rows from Medium to High, which
    the log-bias tune is already doing post-hoc. Rule: **when
    a threshold drift improves bal_acc but hurts raw accuracy,
    it's finding a recall trade not a rule correction**. Don't
    update the DGP rule; keep `Soil<25, Rain<300, Temp>30,
    Wind>10`.
  - Δ bal = +0.000386 is below LB-probe gate (+0.0005) and below
    fold noise σ (~0.00088). No rebuild of downstream models
    needed.

- **Step 2 — monotone OvR XGB (architectural null + Jaccard insight)**:
  - 3 binary XGBs per fold, `binary:logistic` with per-class
    `monotone_constraints` tuples on 43-feature dist set. Same
    5-fold (seed=42) split. Constrained features: Low head 18
    (signs consistent with "wetter/cooler/calmer => more Low"),
    High head 18 (mirror), Medium head fully unconstrained.
  - Per-class best_iters show monotone constraints reduce
    effective capacity: Low (517-720) and High (553-689) heads
    terminate earlier than Medium (694-873) — unconstrained
    Medium needs more rounds to fit the non-monotonic shape.
  - Standalone OOF: argmax 0.96346, tuned 0.97323 (vs vanilla
    XGB-dist 0.97304, Δ = +0.00019 — within fold noise).
  - Error magnitude: 12 188 monotone errs vs 11 862 greedy errs
    / 11 830 LB-best errs. Monotone has ~3 % MORE errors.
  - **Jaccard vs greedy (post log-bias) = 0.8099**, vs LB-best
    = 0.8057. Decent orthogonality — below 0.85 "warn" threshold
    and well below the 0.95 "redundant" band.
  - Blend sweep (fixed baseline bias, log-space):
    ```
    vs greedy (0.97375):
      peak α=0.30  tuned=0.97383  Δ=+0.00009   (α > 0.5 strictly negative)
    vs LB-best (0.97421):
      peak α=0.05  tuned=0.97430  Δ=+0.00006   (monotone-neg past α=0.05)
    ```
  - **Null at blend level.** Both peaks well below LB-probe gate
    (+0.0005) and fold noise (~0.00088). Same failure mode as
    FT-Transformer and TabPFN: decent Jaccard orthogonality but
    the extra-error magnitude (monotone has 326 more errs than
    greedy) dominates — any positive blend weight drags errors
    into the mix faster than it helps correct the orthogonal
    ones. The +0.00019 standalone lift confirms monotone is a
    real alternative optimum (different per-tree structure), but
    it plateaus at the same ceiling as unconstrained trees on
    this feature set. Fourth "same-ceiling" experiment from the
    tree-ensemble family (vanilla XGB, routed XGB, spec-678,
    monotone OvR all land in 0.973–0.974 tuned OOF).

- Meta-read: Session A's three nulls are each valuable confirmations:
  1. Our LB-best submission format is verifiably correct.
  2. The DGP rule thresholds we've been using are the true
     generator thresholds, not a near-miss approximation.
  3. Monotone constraints, a structurally-different tree
     parameterization, still hit the 0.974 family ceiling and
     fail to add blend signal — reinforcing the architectural-
     ceiling diagnosis from the TabPFN/FT-Transformer nulls.
  Each of these closes an uncertainty the user's "look again for
  own-pipeline levers" prompt was chasing.

- LB budget unchanged at 1/10 used today (3 local experiments,
  no LB spend). No submission warranted from any of steps 0/1/2.
- Current best unchanged: `submission_greedy_nonrule_blend.csv`
  OOF 0.97421 / LB 0.97352.
- Next (per the Session plan): Session B (multi-seed fold
  bagging — the variance-estimation check that recalibrates the
  entire OOF→LB ladder), then Session C (flip-signal as
  training-time denoiser — the most architecturally novel
  remaining lever).

### 2026-04-22 — Session B: multi-seed fold bagging (FIRST lift in many sessions — OOF 0.97461, awaiting LB probe)

- Goal: answer whether the LB-validated single-seed OOF of 0.97421 is
  a number we can trust, or partly a lucky `StratifiedKFold(seed=42)`
  split. Every OOF on disk shares that one split — we've never had
  an independent variance estimate. Plan: retrain the full
  greedy+nonrule stack at 2 new fold seeds (7, 123), compute
  cross-seed spread, bag test probs in log-space, gate the bag on
  stability (σ < 0.0005) and emit an LB-candidate.
- Changed: `scripts/session_b_pipeline.py` (full 3-component pipeline
  parameterised via `FOLD_SEED` env var: trains `xgb_dist_routed_v3`,
  `xgb_specialist_678`, `xgb_nonrule`, then builds `hybrid_v3`,
  `greedy`, `lb_best = 0.85*greedy + 0.15*nonrule` log-blends; XGB
  training `seed=42` held constant across all runs to isolate fold
  variance from model variance). `scripts/session_b_analyze.py`
  (loads per-seed OOF/test, reports cross-seed spread, builds
  log-avg bag + prob-avg bag, tunes log-bias on bag OOF, emits
  `submission_lb_best_multi_seed_bag.csv` if gate passes). Seed=42
  uses the historical artefacts (`oof_greedy_blend.npy` +
  `oof_xgb_nonrule.npy`) as-is — that's the LB-validated submission.

- **Cross-seed OOF table (tuned bal_acc)**:
  ```
                          seed=42    seed=7    seed=123   mean      std       spread
  routed_v3               0.97332    0.97316   0.97341    0.97330   0.00013   0.00025
  spec_678 (in-domain)    0.95198*   0.65044*  0.65082*   (per-domain metrics)
  nonrule (alone, tuned)  0.56966    0.57071   0.57016    0.57018   0.00052   0.00104
  hybrid_v3               0.97352    0.97301   0.97336    0.97330   0.00026   0.00051
  greedy                  0.97375    0.97333   0.97355    0.97354   0.00021   0.00042
  LB-best (greedy+nr)     0.97421    0.97388   0.97401    0.97404   0.00018   0.00036
  ```
  *spec_678 argmax on spec domain; seed=42 legacy metric is different
   because it used a different domain restriction. Full-OOF metric
   on spec rows is the 0.65 number shown for 7/123.

- **Key finding — seed=42 is +0.00018 "lucky" but within normal range**:
  All 5 per-component spreads are < 0.001 (max 0.00104 for nonrule,
  which is tuned as argmax-post-bias on a near-useless 13-feature
  subset — tiny differences dominate percentage-wise). LB-best mean
  = 0.97404, which is 0.00018 below seed=42's 0.97421. The "true"
  single-seed LB-best OOF is about 0.974, not 0.97421. **This
  recalibrates every prior conclusion about what counts as "above
  noise"**: the fold-seed-variance floor is ~0.0002-0.0003 (not
  the 0.00088 "fold-std noise within one seed" we'd been using).
  Any prior OOF lift below 0.0003 should be treated as "could be
  split luck" from here on.

- **Multi-seed bag result — NEW CURRENT-BEST OOF at 0.97461**:
  - Log-avg of 3 LB-best OOFs → **tuned 0.97461** (+0.00040 vs
    historical seed=42 0.97421, +0.00057 vs cross-seed mean).
  - Prob-avg bag: **tuned 0.97464** (essentially identical —
    geometric vs arithmetic mean converges on low-variance inputs).
  - Stability gate PASSED: std 0.00018 < 0.0005 threshold.
  - Bag confusion matrix:
    ```
             Low  Medium   High     per-class recall
    Low     368330   1581      6    99.572%
    Medium    5069 229683   4322    96.079%
    High         0    685  20324    96.739%
    ```
    Δ vs seed=42 LB-best: Low recall flat, Medium +0.0003,
    High +0.0001. Essentially preserved per-class balance with
    tighter overall errors.
  - Submission: `submissions/submission_lb_best_multi_seed_bag.csv`.

- **Mechanism read-out**: bagging across fold-splits IS a real
  architectural lever, not a hyperparameter artefact. The 2026-04-22
  seed-bag-greedy experiment tested `fold_seed=42 held fixed, XGB_SEED
  varied (42, 7)` — that was LB-negative (LB 0.97284 vs 0.97296
  single-seed) because XGB at our HPs is near-deterministic across
  model seeds. This experiment holds XGB_SEED fixed and varies
  FOLD_SEED. Different variance source, different behavior —
  cross-split test-prob averaging produces distinct learned
  decision surfaces that compound productively. **Prior rule
  "below-1-fold-std OOF lift from near-deterministic bags =
  non-signal on LB" applies only to model-seed bagging; fold-seed
  bagging is a different beast.**

- **Expected LB for the bag**: at our historical OOF→LB gap
  (0.00069 for LB-best single seed), conservative LB ≈ 0.97392.
  But bagging removes the largest component of that gap (fold-
  split variance), so a tighter gap is plausible. Best-case LB ~
  0.97420. Both estimates exceed current LB-best 0.97352 by
  +0.00040 to +0.00070. This is the first genuinely lift-
  candidate submission since the 2026-04-21 greedy+nonrule
  discovery.

- Artefacts (committed via `!scripts/artifacts/*session_b*` and per-seed
  OOF/test under `!scripts/artifacts/oof_lb_best_fs*.npy`):
  ```
  scripts/artifacts/oof_{routed_v3,spec_678,nonrule,greedy,lb_best}_fs7.npy
  scripts/artifacts/oof_{routed_v3,spec_678,nonrule,greedy,lb_best}_fs123.npy
  scripts/artifacts/test_*_fs7.npy, test_*_fs123.npy
  scripts/artifacts/session_b_fs{7,123}.json
  scripts/artifacts/session_b_multi_seed_summary.json
  submissions/submission_lb_best_multi_seed_bag.csv
  ```

- **LB budget**: unchanged at 1/10 used today. The bag submission
  is PENDING user approval per the top-of-file submission rule.
  Candidate is `submissions/submission_lb_best_multi_seed_bag.csv`
  (OOF 0.97461, expected LB 0.97390-0.97420).
- Current best (LB-validated) unchanged:
  `submission_greedy_nonrule_blend.csv` at LB 0.97352.

- Next (if user approves LB probe and it lands above 0.97352):
  (a) extend bag to 5 seeds; (b) Session C (flip-signal denoiser)
  on top of multi-seed bag. If LB <= 0.97352: variance reduction
  gain didn't transfer; revisit gap calibration.

### 2026-04-22 — Session B LB result: bag LB 0.97297 = −0.00055 REGRESSION (OOF-log-bias overfit via bagging)

- Submitted at 21:16 with user approval. OOF 0.97461 → **LB 0.97297**.
  OOF→LB gap = **0.00164**, much wider than the seed=42 baseline's
  0.00069. **LB −0.00055 vs prior LB-best (0.97352)**.
- Read-out: the bag's OOF lift was NOT new signal — it was
  log-bias coord-ascent exploiting the reduced cross-seed variance
  to pick a sharper decision-rule operating point that doesn't
  transfer to the hidden LB split. Same failure mode as the
  2026-04-22 HP-tuning null: "any OOF lift that comes from better
  decision-rule fit, not better model predictions, risks blowing
  up the OOF→LB gap."
- Paradox reconciled: per-seed LB-best OOFs were 0.97421 / 0.97388 /
  0.97401 (mean 0.97404). The bag OOF of 0.97461 is +0.00057 ABOVE
  the mean. That's the red flag in hindsight — the bag "ensembled"
  three components that were already tuned on the SAME 630k rows.
  Classic stacking-on-OOF overfit mechanism.
- **Important new rule**: **fold-seed bagging creates OOF lift but
  not necessarily LB lift.** Every seed's OOF and bag's OOF are
  measured on the same 630k rows with different train/val
  partitions; averaging their holdout predictions smooths OOF error
  distribution, which lets log-bias find a sharper operating point.
  The LB test set is a different distribution. Test-prob averaging
  across fold-seed bags would only help LB if the model actually
  captured new signal per-seed, which near-deterministic XGB does
  not.
- Budget: 2/10 used today, 8 remaining.
- Current LB best unchanged ... EXCEPT — check next entry.

### 2026-04-22 — OTHER BRANCH: digit-extraction discovered, NEW LB BEST 0.97468 (not our work)

- While Session B was running, `claude/ensemble-model-pipelines-cvHRz`
  discovered **digit-extraction features** as a novel own-pipeline
  lever. Result: `submission_xgb_dist_digits_tuned.csv` at LB
  **0.97468**, +0.00116 over our prior 0.97352.
- Mechanism: for each numeric column, extract digits at decimal
  positions −3..+3 via `floor(v * 10^(-d)) % 10`. 11 numerics × 7
  digits = 77 cols, 31 dropped as constant, 46 surviving. Added to
  the 43-feature dist set → 89 features total.
- Hypothesis: the host's label-generator NN produces synthetic
  numeric values with non-uniform digit distributions (quantization
  / latent-variable footprints) that axis-aligned float splits
  cannot see. Per-digit features expose the pattern directly.
- XGB on digit-enriched features: tuned OOF 0.97449, LB 0.97468,
  gap **−0.00019** (first negative gap in competition log — LB
  BETTER than OOF). Error count 8,846 vs our LB-best's 12,372
  (28 % fewer). Jaccard vs LB-best: 0.57, lowest orthogonality
  seen — first time a new model has BOTH lower Jaccard AND lower
  error count than the baseline.
- Their branch also ruled out LGBM-digits (Jaccard 0.96 with
  XGB-digits, no diversity) and preemptively killed CatBoost.
  Tree-family diversity is exhausted on digit-enriched features.
- Implication for our work: the Session A + B + earlier negative
  diagnoses are still correct — within the original feature set,
  the tree ensemble ceiling IS ~0.974 and no NN / bagging / HP /
  monotone lever rescues it. But digit-extraction is a
  fundamentally DIFFERENT feature representation that bypasses
  the ceiling. The lever we missed wasn't an architecture — it was
  a feature engineering reframe.
- Current LB best: **0.97468** via
  `submission_xgb_dist_digits_tuned.csv`. Gap to pack (0.98114):
  still +0.00646.
- Branch state: the digits scripts and artefacts live on
  `origin/claude/ensemble-model-pipelines-cvHRz` and are not yet
  on our branch. Their next-bet list (per their CLAUDE.md):
  (a) seed-bag digit-XGB (cheap variance reduction), (b) lower-α
  blend digit-XGB × greedy at α ≤ 0.15, (c) OTE / ordered target
  encoding — the notebook claims "digit + OTE" is the recipe for
  the 0.98 ceiling.
- **Decision point for user**: do we (a) adopt their digit-XGB as
  our new baseline and continue Session C (flip-signal denoiser)
  on top of it, (b) redo digit-extraction on our branch
  independently for implementation validation, or (c) pivot
  entirely to OTE which is the claimed 0.98 lever?

### 2026-04-22 — digit-extraction: NEW LB BEST 0.97468 + tree-family diversity exhausted

- Goal: implement digit-extraction FE from the public-notebook pipeline
  description (digits −3..+3 on numeric features). Hypothesis: synthetic
  features may carry NN-generator quantisation artefacts that
  axis-aligned float splits can't see; per-digit features expose them
  directly. Branch: `claude/ensemble-model-pipelines-cvHRz`.
- Changed: `scripts/digit_features.py` — pure-function digit extractor,
  floor(v × 10^(-d)) % 10 with ε to defend against float rounding
  (e.g. 0.3*10=2.999...), plus zero-variance filter.
  `scripts/xgb_dist_digits.py` — XGB on the 43-feature dist set + 46
  surviving digit cols (11 numerics × 7 digits = 77, 31 dropped as
  constant-zero). Same XGB HPs as xgb_nonrule / xgb_dist. Wall ~4 min.
  `scripts/blend_digits.py` — fixed-bias α sweep vs greedy (0.97375)
  and greedy+nonrule (0.97421, prior LB-best).
- Standalone results (OOF bal_acc, 5-fold, seed=42):
  - Prior single-model best (xgb_hybrid_v3): 0.97352
  - XGB-dist + digits **argmax 0.96485, tuned 0.97449**
    → +0.00097 above the best prior STANDALONE model and +0.00028
    above the LB-best BLEND (greedy+nonrule 0.97421).
- Error diagnostics vs LB-best greedy+nonrule:
  - digit-XGB errors: 8,846; LB-best errors: 12,372 (28 % FEWER)
  - Error Jaccard = 0.57 (lowest orthogonality seen; nonrule was 0.80;
    NN/FT-T/TabPFN all 0.62-0.85 but with +16-42 % more errors)
  - **First orthogonal-model attempt where the new model has FEWER
    errors than the baseline** — every prior NN-family failure was
    "Jaccard looks OK but error-magnitude mismatch defeats blend".
- Fixed-bias log-blend sweep (greedy's fitted log-bias reused as-is):
  ```
  target                          peak α    OOF       Δ vs baseline
  vs greedy (0.97375)              0.65     0.97462   +0.00087
  vs greedy+nonrule (0.97421)      0.50     0.97491   +0.00070
  ```
  Both curves unimodal and clean; α=0.50 vs LB-best is a plateau
  from 0.4-0.5, less selection risk than prior OOF-tuned experiments.

- **LB results** (both submissions at 20:49 UTC, user-approved):
  ```
  prior LB best (greedy+nonrule)  OOF 0.97421  LB 0.97352  gap +0.00069
  digit-XGB standalone            OOF 0.97449  LB 0.97468  gap -0.00019
  digit-XGB × LB-best @α=0.50     OOF 0.97491  LB 0.97433  gap +0.00058
  ```
- **NEW LB BEST: 0.97468** via `submission_xgb_dist_digits_tuned.csv`
  (standalone, tuned log-bias). +0.00116 LB over prior best.
- Two surprises:
  1. **Standalone beat the blend on LB despite lower OOF**. α=0.50 was
     too much weight on the model that transfers best; blend averaged
     in components that generalize worse. Opposite of the usual
     selection-overfit pattern.
  2. **Negative OOF→LB gap (−0.00019) on the standalone** — LB is
     BETTER than CV. First time in the competition log. Suggests
     the digit features help the model generalize across the
     train/test split beyond what 5-fold CV measures. Possibly test-
     set digit distributions are less adversarial than a fold split.
- Gap to the pack: 0.98114 − 0.97468 = **+0.00646** (from +0.00762).
  Leader 0.98219 − 0.97468 = +0.00751.

- **LGBM-digits follow-up** (null, tree-family diversity exhausted):
  - `scripts/lgbm_dist_digits.py` — same 89-feature set, LGBM HPs
    mirrored from `benchmark_dist.py` (num_leaves=127,
    min_data_in_leaf=200, lr=0.05, feature/bagging_fraction=0.9).
    ~8 min on 5 folds.
  - Standalone tuned OOF **0.97350** (−0.00099 vs XGB-digits 0.97449).
  - **Jaccard(LGBM-dig, XGB-dig) = 0.9591** with near-identical error
    counts (8,874 vs 8,846). Effectively the same predictor.
  - Blend vs XGB-digits: monotone-negative from w_lgbm=0 (peak at 0).
  - Blend vs prior greedy+nonrule LB-best: peak w=0.15 OOF 0.97457
    (+0.00036). But strictly below the new LB-best (0.97468 LB), so
    this lever only helps the OLD path, not the CURRENT one. Dead.
  - Read-out: **trees leaf-wise (LGBM) vs level-wise (XGB) converge
    to near-identical predictions when features are highly
    informative** — the digit + distance features leave no room for
    architectural differences to surface. Same lesson as the
    Jaccard-too-high blend-null pattern: tree-family diversity on
    this feature set is structurally exhausted. CatBoost (next in
    queue) was killed before fold-1 completed based on the LGBM
    result, saving ~2.5h CPU. `scripts/cat_dist_digits.py` retained
    with a fold-1 Jaccard≥0.90 abort gate for a future GPU slot.
- LB budget: **5/10 used today**, 5 remaining.

- Candidate status:
  - **Primary (new LB best)**:
    `submissions/submission_xgb_dist_digits_tuned.csv` → LB 0.97468
  - Safe fallback (prior LB best):
    `submissions/submission_greedy_nonrule_blend.csv` → LB 0.97352

- Next bets (ROI order):
  1. **Seed-bag digit-XGB** (3-5 seeds, ~30 min) — variance reduction
     on new LB-best. Expected +0.0001-0.0003 LB. Cheap insurance.
  2. **Lower-α blend** digit-XGB × greedy at α ∈ {0.05,...,0.15} —
     α=0.50 lost 0.00035 LB; smaller α might preserve standalone's
     negative gap.
  3. **OTE (ordered target encoding w/ 4× shuffle)** — structurally
     different feature representation (cumulative LOO stats), not
     another model on the same features. 1-2h implementation. The
     notebook's "digit + OTE" two-pipeline stacker is the mechanism
     claimed to drive the 0.98 ceiling.

- **Lessons** (candidate adds to LEARNINGS.md):
  1. **Per-digit extraction is a fundamentally different signal path
     from raw floats** on synthetic-data problems. 46 digit cols
     lifted XGB-dist tuned OOF from 0.97304 to 0.97449 (+0.00145) —
     larger than any single FE lever tested before. Motivation:
     synthetic/NN generators often produce values with non-uniform
     digit distributions; axis-aligned splits on the float can't see
     this, per-digit features expose it directly. Worth trying on
     every synthetic tabular comp.
  2. **Negative OOF→LB gap is a signal to trust, not a measurement
     error**. 5-fold CV produces adversarial splits that can
     under-estimate OOD generalization on test sets drawn from a
     similar but distinct distribution. When it happens, don't
     chase OOF further — standalone is likely your final answer.
  3. **Blend α≥0.5 is suspicious when new model has sharply better
     standalone**. Blend LB (0.97433) < standalone LB (0.97468)
     confirms that averaging in weaker components pulls
     generalization down. If standalone LB > blend LB, revisit α
     much lower or skip the blend.
  4. **Tree-family diversity dies when features are highly
     informative**. Jaccard 0.96 between LGBM and XGB on the same
     digit-enriched feature set means architectural differences
     don't matter. For orthogonal blend signal: different feature
     representation (OTE, cell-partition specialists), or a model
     family that uses features differently (cumulative TE,
     attention tokens), not another tree variant.
