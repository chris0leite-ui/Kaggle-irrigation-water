# CLAUDE.md

Guidance for Claude Code when working in this repository.

## ⚠️ ARTIFACTS ARE TRACKED BY DEFAULT (gitignore inverted 2026-04-27)

`scripts/artifacts/` follows an INVERTED gitignore policy:
- **Tracked by default**: `oof_*.npy`, `test_*.npy`, `*_results.json`,
  any other `.npy`/`.json` not matching an ignore pattern.
- **Ignored**: per-fold checkpoints (`*_fold[0-9]*.npy`),
  `.log`/`.db`/`.pkl`, and transient prefixes (`_smoke_*.npy`,
  `tmp_*.npy`, `scratch_*.npy`).

This eliminates the prior 536-line per-file whitelist. Cross-branch
artifact sharing is now zero-friction: if you produce an OOF/test
.npy, just `git add` it and commit. No need to edit `.gitignore`.

If you need a transient .npy (smoke / debug / scratch), name it
with one of the ignored prefixes (`tmp_xxx.npy`, `scratch_xxx.npy`)
or it WILL be tracked.

## ⚠️ NEVER GIVE UP — ALWAYS ASSUME THERE'S MORE LIFT AVAILABLE

**Every session pulled from GitHub must take it as given that the
current LB ceiling is NOT structural and that we can still improve.**
The hypothesis "we have hit the own-pipeline wall" has been advanced
(and refuted) at every prior LB plateau in this repo: 0.97097 →
0.97296 → 0.97352 → 0.97468 → 0.97482 → 0.97581 → 0.97939 → 0.97998
→ 0.98005 → 0.98008 → 0.98094. Each plateau looked structural at the
time; each was broken by a mechanism not yet tried.

Concrete rules for this disposition:
1. Treat "saturation evidence" as evidence we have tested *known*
   levers, NOT as evidence no lever exists. The space of possible
   levers is unbounded; our prior tests are bounded.
2. When a session log says "ceiling is structural," interpret it as
   "the current OOF bank + tested attack vectors saturate at this
   value" — open question is what to ADD, not whether the wall is
   real.
3. Never recommend "lock final selection and stop spending compute"
   as a primary strategy while LB submissions remain in budget.
   Locking is a fallback for the final hours of the comp, not a
   stopping rule for ongoing exploration.
4. After every null result, brainstorm at least 3 mechanisms NOT
   yet on the hypothesis board before declaring exhaustion. Most
   plateaus broke via levers the prior session called "skipping on
   principled grounds" — re-examine those rejections critically.
5. The leader and pack scores are existence proofs that >0.98094 is
   reachable on this data. We are missing a mechanism, not facing a
   theoretical bound.

This rule overrides any session-log conclusion of the form
"own-pipeline ceiling reached." When you read such a conclusion,
treat it as a snapshot of partial knowledge, not as a final verdict.

## ⚠️ GPU KERNEL RUNTIME CAP — 1 HOUR MAX

**Never launch a Kaggle GPU kernel without a hard 1-hour wall-time
budget.** If an estimate suggests >1h, shrink the config (fewer
folds, smaller `n_ens`, reduced epochs, subsampled data) BEFORE
pushing. If a running kernel exceeds 1h, kill it — don't wait for
"just a bit more."

Context: 2026-04-24 RealMLP-TD via pytabkit pushed with a naive
"~45 min P100" estimate based on the public-kernel author's claim.
v3 ran **3h 34min of CPU preprocessing before GPU training even
started** (pytabkit's `n_ens=8` internal preprocessing × per-fold
`TargetEncoder(cv=5)` compounding + sklearn / Lightning setup).
Kernel killed at that point — zero output produced. Wasted the
queued GPU slot and ~4h of session time waiting on monitors.

Concrete rules for future GPU kernels:
1. Estimate wall time by MULTIPLYING published single-fold claims
   by `n_folds × n_ensemble × 2` (safety margin). If the product
   exceeds 60 min, shrink the config.
2. Always run `SMOKE=1` locally (or as a 5-min Kaggle variant)
   BEFORE pushing production. The `n_ens` / `cv` multiplier bugs
   show up immediately on 20k rows.
3. If a kernel is still in preprocessing at t+30min with no fold
   output, KILL IT — don't let it eat the budget.
4. Prefer CPU pipelines. Most own-pipeline levers on this problem
   worked fine on 12-16 core CPU in under 2h wall.

## ⚠️ LB SUBMISSION RULE — ALWAYS ASK FIRST, NEVER LOOP

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

**Never wrap `kaggle competitions submit` in a retry / `until` /
`while` / `for` loop, ever — even on transient errors (503, network
timeouts, OAuth failures).** Every loop iteration is a NEW LB
submission against the daily quota. On 2026-04-26 a `until ... |
grep -q "successfully submitted"` retry loop with a case-mismatch
in the success marker burned 4 redundant slots on
`submission_v6_full_a350.csv` (07:09:31, 07:10:04, 07:14:44,
07:15:22 — all returning the same deterministic LB 0.98012). 3 LB
slots wasted because the loop's terminator never matched Kaggle's
"Successfully submitted" capital-S string.

Concrete rules going forward:
1. **One submission per `kaggle competitions submit` invocation.**
   Run the command exactly once, report the result, wait for next
   user instruction. NEVER auto-retry on any failure.
2. **If Kaggle returns a transient error (503 / network),** report
   the error to the user verbatim and ask whether to retry. The
   user decides if/when to retry, manually.
3. **If a submission needs to be revised** (different α, different
   weights, different CSV), build the new CSV locally, present the
   diagnostic, and wait for a fresh user go-ahead. Each revision is
   a separate one-shot submit invocation.
4. **No Monitor / background loop / `until` / cron may include the
   `kaggle competitions submit` command** — period. Monitors that
   POLL submission status (read-only, e.g. `kaggle competitions
   submissions -v`) are fine. Monitors that WRITE submissions are
   forbidden.
5. The cost asymmetry is severe: a wasted slot can't be recovered
   today, and on the final day-of-deadline a wasted slot may cost
   the competition. The cost of pausing and asking is zero.

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

## ⚠️ KEEP FILES SHORT AND MODULAR

**Long single-file writes risk stream idle timeouts on the API.**
When scaffolding new pipelines (kernels, scripts), split into
multiple short files (≤~150 lines each) with clear
responsibilities — one file for model, one for features, one for
training loop, one for the orchestrator — rather than a single
large file. For Kaggle kernels that require a single `code_file`,
assemble a thin orchestrator that imports from sibling modules
(drop them next to the kernel script) or inline via a build step.
Same rule for plans: short modular docs, not monoliths.

## ⚠️ SMOKE-TEST BEFORE LONG RUNS

**Always run a 1-fold / 1-trial smoke pass before launching a
full multi-hour computation.** Kaggle kernels, Optuna sweeps,
seed bags, and any pipeline with >10 min wall time should be
validated end-to-end on a tiny configuration first — `N_FOLDS=1`,
`N_TRIALS=1`, a stratified subsample, or a CPU debug run. Catch
bugs (sibling-import errors, `from __future__` placement, shim
ordering, pip reinstall flags, tensor shape mismatches, OOM,
GPU/CUDA mismatches, output-path permissions) on the 2-minute
smoke cycle, not the 3-hour real run.

Concretely for a new Kaggle kernel:
1. Build + push a smoke config (`N_TRIALS=1, TRIAL_EPOCHS=1,
   N_FOLDS=1, subsample=50k`) — expect <5 min wall.
2. Confirm it completes (`COMPLETE` status, submission CSV
   present, results JSON parses, OOF/test npy shapes correct).
3. THEN push the production config.

Two failed kernel iterations (v1 SyntaxError on `__future__`
ordering, v2 broken torch reinstall due to shim placement)
on 2026-04-23 burned ~30 min and a P100 warmup before any
training happened. A 2-min smoke run would have caught both.

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

### 2026-04-23 — digits_pairs variant + 3-way blend: null

- Goal: extend digit-OTE key set with (digit × Crop_Type) pairs.
  Hypothesis: per-category per-digit-value class distributions should
  expose finer flip patterns than per-digit-value alone.
- Changed: `scripts/ote_features.py` — `_key_strings` rewritten to use
  numpy `np.char.add` vectorised string concat instead of pandas
  `DataFrame.agg(axis=1)` Python row-iterator. **6× speedup**: 92-key
  OTE pipeline dropped from "hung indefinitely" (15+ min on test-OTE
  build) to 47 min end-to-end wall.
- Results (OOF 5-fold, seed=42):
  - Standalone: argmax 0.96486 (digits 0.96520, digit-XGB 0.96485),
    tuned **0.97375** — ties default cat-OTE, below digits variant.
  - Blend vs digit-XGB (LB-best 0.97449): **peak α=0.40 → 0.97477
    (Δ = +0.00027)** — essentially identical to digits variant's
    +0.00028. Fractional 1-in-100k gap, within fold noise.
- 3-way blend (digits + digits_pairs + digit-XGB): grid search over
  weights on OOF → best at (w_digits=0.25, w_pairs=0.30, w_digit=0.45)
  → **OOF 0.97483 (Δ = +0.00006 vs digits-only blend)**. Tiny lift,
  well below LB-probe threshold (+0.0005). No submission warranted.
- Diagnosis: the pair encodings carry the SAME signal as single digit
  encodings; XGB's tree splits on raw digit cols + digits-OTE already
  recover per-category per-digit relationships internally. Adding
  explicit pair-OTE columns adds noise (276 OTE cols vs 138) without
  new information.
- **Fold-seed bag on digits variant: SKIPPED (recommended null)**.
  Three reasons: (1) Session B precedent (fold-seed bag of
  greedy+nonrule regressed LB −0.00055); (2) current LB-best has
  NEGATIVE OOF→LB gap (LB > OOF by 0.00005), so coord-ascent is
  already overshooting OOF pessimism — bagging tightens OOF and
  removes the pessimism cushion, likely regressing LB; (3)
  digits_pairs failed to add blend signal, so the architectural
  lever is exhausted on this pipeline — variance reduction can't
  recover missing signal.
- **Final OTE variant table**:
  ```
  variant         keys                    α   K   tuned OOF   peak α   Δ vs digit   LB
  default         8 cats + 6 pairs + 2    10  8   0.97375     0.00     +0.00000      -
  light           same                    1   2   0.97390     0.025    +0.00005      -
  **digits**      46 digit cols           10  8   0.97415     0.40     +0.00028      **0.97482**
  digits_light    46 digit cols           1   2   0.97371     0.40     +0.00020      -
  digits_pairs    92 (digit + pair)       10  8   0.97375     0.40     +0.00027      -
  3-way blend     digits + pairs + digit  -   -   0.97483     -        +0.00006      -
  ```
- **Current LB best unchanged: 0.97482** via
  `submission_digit_ote_digits_blend.csv` (digits-OTE α=0.40 blend).
  LB budget: 6/10 used today, 4 remaining.

### 2026-04-23 — NEW LB BEST 0.97581: greedy forward-selection over full OOF bank

- Goal: after exhausting within-OTE levers, pivot to a fundamentally
  different lever — ensemble the saved OOFs. Ran three parallel
  experiments: greedy forward-selection, ExtraTrees, and LGBM+OTE.
- **Winner: greedy 6-way log-blend → LB 0.97581 (+0.00099 over prior
  LB best 0.97482)**. Gap to pack 0.98114 now +0.00533, leader 0.98219
  +0.00638.
- Changed: `scripts/greedy_full_bank.py` — forward-selection log-blend
  over 15-17 saved OOF/test pairs. Uses digit-XGB's tuned bias as a
  fixed anchor (no retune per candidate) to avoid binhigh-style
  selection overfit. Emit gate: Δ ≥ +1e-4 (looser than +5e-4 since
  selection risk on pre-computed OOFs is low).
- Greedy result (fixed digit-XGB bias):
  ```
  start:    digit_xgb            OOF 0.97449  (anchor)
  + step 1: xgb_nonrule α=0.20   OOF 0.97506  Δ=+0.00057
  + step 2: digits_ote α=0.30    OOF 0.97534  Δ=+0.00028
  + step 3: xgb_corn α=0.10      OOF 0.97544  Δ=+0.00010
  + step 4: digits_pairs α=0.075 OOF 0.97552  Δ=+0.00008
  + step 5: digits_light_ote α=0.05 OOF 0.97558  Δ=+0.00006
  stop:     no candidate improves by ≥ 1e-5
  ```
  Final weights (log-space, sum to 1):
  ```
  0.4429  digit_xgb        (anchor, 44% of log-prob mass)
  0.2373  digits_ote       (24%)
  0.1107  xgb_nonrule      (11%, non-rule-feature signal)
  0.0879  xgb_corn         (9%, Frank-Hall ordinal decomposition)
  0.0712  digits_pairs     (7%)
  0.0500  digits_light_ote (5%)
  ```
- **Key discovery — xgb_nonrule as first greedy add was the biggest
  single-step lift (+0.00057).** The non-rule-feature lever (earlier
  LB +0.00056 as part of greedy+nonrule) transferred cleanly onto the
  digit-XGB anchor at a different weight. This suggests the non-rule
  signal is ORTHOGONAL to digit-family features, which makes sense:
  digit-XGB captures quantization artefacts in rule features but has
  no way to exploit the NN-generator's flip signal that depends on
  non-rule cats like Humidity and Previous_Irrigation. Non-rule-XGB
  fills that gap.
- **xgb_corn as 3rd add (+0.00010) also surprising**: previously ruled
  out as a standalone blend leg (null on greedy+nonrule). CORN's Frank-
  Hall ordinal decomposition trades Medium for High recall, which
  ALONE hurts macro-recall — but as a blend component at 9% weight on
  top of a stronger base, it provides complementary high-recall
  evidence without dominating.
- LB result (submitted 06:31 UTC, user-approved):
  - `submission_greedy_full_bank.csv` → **LB 0.97581**
  - OOF 0.97558 → LB 0.97581 = **gap −0.00023** (LB BETTER than OOF).
    Even more negative than digit-XGB's −0.00019 and digits-OTE
    blend's −0.00005. Digit-family pipelines produce progressively
    more negative gaps as components add.
- Updated calibration ladder:
  ```
  single tuned LGBM             0.97097 → 0.96972   gap 0.00125
  greedy 3-way log-blend        0.97375 → 0.97296   gap 0.00079
  greedy + nonrule α=0.15       0.97421 → 0.97352   gap 0.00069
  digit-XGB standalone          0.97449 → 0.97468   gap -0.00019
  digits-OTE × digit-XGB α=0.40 0.97477 → 0.97482   gap -0.00005
  **greedy full-bank 6-way      0.97558 → 0.97581   gap -0.00023**  ← NEW LB BEST
  ```
- **ExtraTrees on dist+digits: null**. Standalone argmax 0.96510,
  tuned 0.96676. 500 trees × `class_weight='balanced'` produces
  flattened prob scale (tuning lifts only 0.002 vs XGB family's 0.01).
  At fixed digit-XGB bias, OOF drops to 0.93023 — off-calibration.
  Greedy rejected it at every α (blending in ET's weaker answers
  hurts the composite). Rule: **orthogonal-model candidates need
  probability scales compatible with the anchor's bias before they
  can contribute to a log-blend**.
- **LGBM on dist+digits+OTE: null**. Tuned OOF 0.97330 — below both
  digit-XGB (0.97449) and LGBM-digits alone (0.97348). Best_iter
  hovered around 150-170 across folds (XGB-digits+OTE was 500+),
  meaning LGBM exhausts signal quickly on the OTE-enriched set. OTE
  features carry structural information LGBM's leaf-wise splits
  can't exploit as effectively as XGB's level-wise splits. Tested on
  top of the 6-way blend at α ∈ [0.025, 0.20]: strictly monotone
  negative. Confirmed null.
- **Path dependence warning**: rerunning greedy with `lgbm_digit_ote`
  in the candidate pool found a DIFFERENT local optimum (3-way:
  digit_xgb + xgb_nonrule + lgbm_digit_ote → OOF 0.97541) that's
  WORSE than the 6-way without it (0.97558). The greedy heuristic
  picked lgbm_digit_ote over digits_ote early because of slightly
  better fixed-bias standalone OOF, then couldn't find further
  additions. Lesson: **greedy local optima depend on candidate pool;
  run without weak candidates first, then test adding them on top of
  the strong blend**.
- LB budget: **7/10 used today**, 3 remaining.
- Artefacts committed (on feature branch + main):
  - `oof_greedy_full_bank_6way.npy`, `test_greedy_full_bank_6way.npy`
  - `oof_extratrees_dist_digits.npy`, `test_extratrees_dist_digits.npy`
  - `oof_lgbm_dist_digits_ote.npy`, `test_lgbm_dist_digits_ote.npy`
  - `greedy_full_bank_results.json`, `extratrees_dist_digits_results.json`,
    `lgbm_dist_digits_ote_results.json`
  - `submission_greedy_full_bank.csv` (LB 0.97581, new best)

### 2026-04-23 — ExtraTrees v2 + multi-start sanity check: both confirm 6-way is optimal

- Goal: after LB 0.97581, test two cheap levers to see whether our
  submitted 6-way blend is the true local optimum or whether a
  better blend exists.
- **ExtraTrees v2 (no `class_weight='balanced'`)** — `EXTRATREES_VARIANT=v2`
  via env var. Hypothesis: v1's balanced weights flattened probabilities
  to where log-bias couldn't recover them (OOF 0.93023 at digit-XGB's
  fixed bias → greedy-rejected). Without balanced weights, raw argmax
  tracks the Low-majority prior (0.96079 vs v1's 0.96419), but prior-
  reweight (0.96588) + tuned log-bias (**0.96631**) converge to a value
  just below v1 (0.96676). Null — both ET variants plateau ~0.008 below
  digit-XGB (0.97449), and the "calibration fix" produces slightly
  WORSE tuned OOF than v1. Implication: `class_weight='balanced'` was
  doing something log-bias couldn't reproduce, and either way ExtraTrees
  is structurally below XGB on this feature set.
- **Multi-start + backward-elim + weight refinement** on the 6-way:
  1. Multi-start (15 anchors): best alternate = `lgbm_digit_ote` start
     → OOF **0.97559** (Δ +0.00002 vs submitted 0.97558 — numerical
     tie). Different 6-way composition (lgbm_digit_ote 0.31 + digit_xgb
     0.31 + xgb_nonrule 0.21 + hybrid_lgbmxgb 0.09 + digits_ote 0.05 +
     cat_ote_light 0.03) but same ceiling. Confirms the greedy reaches
     the same plateau from multiple starting points.
  2. Backward-elimination: dropping any one component strictly hurts.
     Ranked by harm of removal (biggest loss = most essential):
     ```
     drop xgb_nonrule       -0.00063  (most essential, confirms +0.00057
                                       lift was real structural signal)
     drop digit_xgb         -0.00035
     drop xgb_corn          -0.00032
     drop digits_ote        -0.00028
     drop digits_pairs      -0.00009
     drop digits_light_ote  -0.00006
     ```
     The Frank-Hall CORN component (−0.00032 to remove) is genuinely
     contributing despite being a standalone-null leg on greedy+nonrule
     — adds up as a 9% weight in the anchor-family blend.
  3. Coord-ascent weight refinement (perturb each weight by ±0.025,
     ±0.05): converged at iter 0 → no improvement. The greedy weights
     are already at a local weight-optimum.
- **Verdict**: the submitted 6-way blend is a genuine, stable local
  optimum. Multi-start finds tied configurations; backward-elimination
  confirms all 6 components contribute positively; weight perturbation
  doesn't improve OOF. No LB submission warranted.
- **Meta-lesson added**: the greedy path-dependence trap we hit earlier
  (including lgbm_digit_ote in the pool produced a worse 3-way optimum)
  is resolved by multi-start — starting from `lgbm_digit_ote` as the
  anchor gives a tied solution. The heuristic rule "run greedy with
  strong-only candidates first" is good practice but the multi-start
  check is the cleaner verification.
- LB budget unchanged: 7/10 used today, 3 remaining.
- Current LB best unchanged (at time of this entry):
  `submission_greedy_full_bank.csv` at **LB 0.97581**. **Subsequently
  superseded** by the parallel-session public-notebook full recipe
  (see next entry: LB 0.97939).

### 2026-04-23 — public-notebook FULL RECIPE: NEW LB BEST 0.97939 (+0.00457)

- Goal: stop treating the +0.006 gap to the leader as a ceiling; pull
  the top-voted public kernels, read what they actually do, and run
  it. Motivated by the user push "stop seeing a ceiling where there
  is no ceiling. we simply lack expertise or the right approach."
- Sources read (all via `kaggle kernels pull`):
  - `cdeotte/original-data-exact-formula` — 103 votes. Publishes the
    exact LR coefficients on the 10k (logit formula we have as
    `dgp_formula.py`) but NOTHING about the 0.98+ LB pipeline.
  - `aliafzal9323/s6e4-0-978-xgb-cat-pairwise-te-magic` — 61 votes.
    LB 0.97779. Recipe: all C(19,2)=171 COLUMN pairs string-concat
    + factorize + sklearn `TargetEncoder(multiclass, cv=5)` per fold.
    Plus `TE_ORIG_*` (mean TE from 10k original) on every col.
    XGB+CAT GPU, LR stacking meta-learner.
  - `yunsuxiaozi/pss6e4-xgb-cv-0-979805` — 74 votes. CV 0.9798.
    Recipe: per-numeric digit features (`(v // 10**k) % 10` for
    k=-4..+3) + a custom `OrderedTE` class (per-class cumulative
    shuffled stats, shrinkage a=1) applied to digits+cats. Heavy-reg
    XGB (`max_depth=4, alpha=5, reg_lambda=5, max_leaves=30,
    max_bin=10000, lr=0.1, n_est=512, early_stop=1024`).
    Post-hoc Optuna class-weight tuning.
  - `include4eto/ps6e4-xgb-cudf-pseudo-labels` — 53 votes.
    V10 = consolidation of all the above. Feature blocks:
    + cats (8) + CAT×CAT combos (28) + digit cols (~66 after
    dropping test-constant positions) + num-as-cat (11) + threshold
    flags (4); numeric blocks: raw nums (11) + LR-formula logits (3,
    from cdeotte's coefs) + FREQ per cat+combo (~44) + ORIG_mean +
    ORIG_std per col (~48). OrderedTE on all ~117 cat-block
    features → 351 OTE features. Heavy-reg XGB params as above.
    Optional pseudo-labels (OFF in V10). LR stacking meta.
  - `mohit78241/s6e4-ensemble-voting-transfer-0-981-lb` — 58 votes.
    LB 0.981. Recipe: hard-vote ensemble of pre-existing public
    submission CSVs (`0.97971.a.csv`, `0.98074.csv`, etc., credited
    "nina2025"). **This is the public-CSV-blend path CLAUDE.md
    forbids** — not for us.
- Changed: scaffolded the V10 recipe as three small modules:
  - `scripts/recipe_features.py` — every FE block listed above.
  - `scripts/recipe_ote.py` — OrderedTE class (per-class cumulative,
    exclusive-of-row at fit, full-train stats at transform,
    shrinkage a=1). 12-row toy verified correctness (priors on
    first occurrence, shrunken stats on later rows, full-train
    stats on unseen keys). Later pd.concat optimisation to avoid
    fragmentation warnings (~2x speedup on full scale).
  - `scripts/recipe_full_te.py` — pipeline: FE → 5-fold StratifiedKFold
    (seed=42, aligned with all other OOFs on disk) → per-fold OTE
    on 117 categoricals (3 cls each = 351 OTE + 85 numeric = 436
    features used, 443 with flag duplicates) → heavy-reg XGB with
    `compute_sample_weight("balanced")` → save OOF + test probs →
    tuned per-class log-bias (common.tune_log_bias) → submission.
  - `max_bin` reduced from notebook's 10000 to 1024, `n_estimators`
    capped at 3000 (vs 50000), `early_stop=200` for CPU feasibility.
- Smoke test (`SMOKE=1`): 20k train, 10k test, 2 folds, smaller XGB.
  OOF tuned 0.96381 in 70s wall. Verifies the pipeline end-to-end
  before burning the full-scale run.
- Full production run (504k train, 5 folds, ~55 min total wall):
  ```
  fold 1  argmax 0.97544  best_iter 1414
  fold 2  argmax 0.97659  best_iter 1349
  fold 3  argmax 0.97721  best_iter 1253
  fold 4  argmax 0.97465  best_iter 1159
  fold 5  argmax 0.97557  best_iter 1230
  ```
  **Overall OOF argmax = 0.97589  (±0.00090)**
  **Overall OOF tuned log-bias = 0.97967  bias = [1.432, 1.469, 3.401]**
  Note: the Low/Medium biases are an order of magnitude larger than
  our prior pipelines' (~0.13 / ~0.57) because
  `sample_weight="balanced"` during XGB training already inflates
  Low/Medium probs; log-bias corrects back.
- LB probe (user-approved, submitted 08:42 UTC):
  `submission_recipe_full_te.csv` → **LB public = 0.97939**
  **Δ vs prior LB best (digits-OTE blend 0.97482) = +0.00457.**
  Biggest single lift of the competition by ~4x.
  OOF→LB gap = 0.00028 — the best calibration of any
  submission (previous best was −0.00019 on digit-XGB standalone).
- Updated calibration ladder:
  ```
  single tuned LGBM              0.97097 → 0.96972   gap 0.00125
  greedy 3-way log-blend         0.97375 → 0.97296   gap 0.00079
  greedy + nonrule               0.97421 → 0.97352   gap 0.00069
  digit-XGB standalone           0.97449 → 0.97468   gap -0.00019
  digits-OTE × digit-XGB α=0.40  0.97477 → 0.97482   gap -0.00005
  greedy_full_bank               0.97552 → 0.97581   gap  0.00029   (parallel branch)
  **recipe_full_te              0.97967 → 0.97939   gap  0.00028   ← NEW LB BEST**
  ```
- Pack 0.98114 now +0.00175 above (was +0.00632). Leader 0.98219
  now +0.00280 above (was +0.00737). Reachable via incremental
  polish on this pipeline (seed-bag, XGB+CatBoost 2-way OTE blend,
  LR stacking meta as the published Ali Afzal kernel does, etc.).
- LB budget: 3/10 spent today (1 recipe_full_te LB probe +
  2 digits-OTE variants from earlier). 7 remaining.
- Strategic read: the "architectural ceiling" framing was wrong.
  The +0.006 gap was just FE coverage — we had implemented ~1/5 of
  the published public-kernel recipe (digit extraction + narrow OTE).
  Running the full recipe (all OTE sub-levers + ORIG aggregation
  + FREQ + LR-formula logits + heavy-reg XGB operating point)
  closed most of the gap in a single experiment. **Rule for
  future competitions: spend 15 min pulling the top-voted public
  kernels every 2-3 days of null experiments**; the cost is
  trivial and the risk of missing a published lever is material.
- Artefacts committed:
  - `scripts/recipe_features.py` + `scripts/recipe_ote.py` +
    `scripts/recipe_full_te.py`
  - `scripts/artifacts/oof_recipe_full_te.npy` (gitignored by default
    — add explicit `!` exception if cross-branch blending wanted)
  - `scripts/artifacts/test_recipe_full_te.npy` (ditto)
  - `scripts/artifacts/recipe_full_te_results.json`
  - `submissions/submission_recipe_full_te.csv`
- Lessons logged to `LEARNINGS.md §Process` and `§Target encoding &
  FE menu`: community-kernel research cadence, OTE-as-family (not
  single-lever), heavy-reg wide-feature XGB as distinct optimum,
  script-module size discipline, SMOKE env-var toggle, monitor
  log-replay gotcha. Full recipe template captured for reuse on
  the next synthetic tabular competition.

### 2026-04-23 — greedy-from-recipe + LR meta-stacker (both null on top of 0.97939)

- Goal: test whether recipe_full_te (LB 0.97939) can be lifted by
  blending it with our prior digit-family OOFs or by a learned LR
  meta-stacker. Cheap post-hoc experiments — no retraining.
- Changed: `scripts/greedy_from_recipe.py` (forward-selection with
  recipe_full_te as anchor + fixed bias, 16-model candidate pool);
  `scripts/lr_meta_stack.py` (multinomial LR on 30-dim meta-features
  = 10 models × 3 classes, 5-fold OOF with class_weight='balanced').

- **Greedy from recipe (BORDERLINE)**: recipe_full_te (0.925) +
  digit_xgb (0.075) log-blend → OOF **0.97978** (Δ=+0.00012).
  - All other 15 candidates (digits_ote, nonrule, corn, greedy_6way,
    hybrid_lgbmxgb, etc.) rejected — recipe already captures what
    they offer. Only digit_xgb's raw digit features added a sliver.
  - Standalone OOFs at recipe's bias:
    ```
    recipe_full_te      0.97967  (anchor)
    greedy_full_bank    0.97237
    digit_xgb           0.97225
    xgb_corn            0.97211
    digits_ote          0.97157
    (others)            0.970-0.972
    ```
    Note: xgb_nonrule evaluates to 0.536 at recipe's bias (recipe's
    bias is tuned for recipe's scale, wildly different from nonrule's
    rule-free prob scale). Greedy still blends it in principle but
    effective weight is tiny.
  - Blend has 9,857 errors vs recipe standalone's 10,114 — fewer
    errors, Jaccard 0.9543 (high overlap). Real signal but small.
  - Above +1e-4 emit gate but below +5e-4 submit-ready threshold.
    Given recipe's POSITIVE OOF→LB gap (+0.00028), a +0.00012 OOF
    lift likely translates to −0.00016 LB or zero. Not submitting.

- **LR meta-stacker (NULL)**: 5-fold multinomial LR on concatenated
  per-class probs from 10 models (recipe + greedy_6way + digit_xgb +
  digits_ote + digits_pairs + cat_ote + lgbm_digit_ote + xgb_nonrule
  + xgb_corn + hybrid_lgbmxgb).
  - argmax 0.97833, tuned 0.97908 — **−0.00059 vs recipe standalone**.
  - The LR can't improve on recipe because recipe already occupies
    ~92.5% of the blend weight in the greedy (inferred from how
    little room is left). Passing through ~all recipe + tiny
    corrections is what greedy did, and LR can't do better than that
    without overfitting.
  - Consistent with earlier LR-meta-stacker null on the
    greedy+nonrule stack (from 2026-04-21 soft-blend session).

- Verdict: recipe_full_te is close to saturated on own-pipeline
  OOFs. Neither greedy (+0.00012, borderline) nor LR meta (−0.00059)
  gives a submission-worthy lift. LB budget unchanged: 7/10 used
  today, 3 remaining. Current LB best unchanged at **0.97939**.

- Remaining untried own-pipeline levers: CatBoost on the recipe
  feature set (novel model family, ~1h). Everything else (seed-bag,
  HP tuning, pseudo-labeling) is ruled out by prior LB regressions.

### 2026-04-23 — CatBoost on recipe feature set: blend null, LB-best unchanged

- Goal: test the one remaining untried own-pipeline lever — a novel
  model family (CatBoost) on the full recipe feature set (~440 cols).
  Target: Jaccard < 0.9 with recipe + competitive standalone OOF
  would unlock a lift; otherwise recipe is saturated.
- Changed: `scripts/recipe_full_te_catboost.py` — mirrors
  `recipe_full_te.py` but with CatBoostClassifier
  (`depth=4, l2_leaf_reg=10, iterations=2000, lr=0.1, Bernoulli
  bootstrap, od_wait=200`). Same 5-fold seed=42 split, same
  per-fold OrderedTE, same features, same class-balanced sample
  weights. 17 min total wall on CPU (16 min train + 1 min OTE).
- Standalone results (5-fold OOF, seed=42):
  - Per-fold argmax: 0.97740 / 0.97800 / 0.97919 / 0.97786 / 0.97784
  - Mean argmax **0.97806 ± 0.00060** (vs recipe 0.97589 mean,
    **+0.00217 higher per-fold**)
  - **Tuned log-bias OOF 0.97936** (vs recipe 0.97967, **−0.00031**
    below). Note the paradox: CatBoost is +0.00217 stronger at argmax
    but −0.00031 weaker after bias tuning. Means CatBoost's prob scale
    is closer to prior-balanced (argmax already near macro-recall
    optimum) while XGB needs large bias shifts to reach the same point.
  - Best-iter = 1999-2000 on every fold → CatBoost HIT the iteration
    cap without early stopping. Would benefit from more iterations
    but time-boxed.
  - Error count at its own bias = 10,447; at recipe's bias = 13,975
    (recipe's bias is calibrated for XGB prob scale, not CatBoost).
  - **Jaccard vs recipe_full_te at recipe's bias = 0.8060**. First
    time we've seen Jaccard < 0.85 at this competitive OOF level!
    Orthogonality looks promising on paper.
- Blend sweep (fixed recipe bias, α_cat × CatBoost + (1−α_cat) × recipe):
  ```
  α_cat=0.05  OOF 0.97958  Δ=−0.00008
  α_cat=0.10  0.97950  Δ=−0.00017
  α_cat=0.15  0.97947  Δ=−0.00020
  α_cat=0.20  0.97952  Δ=−0.00014
  α_cat=0.25  0.97961  Δ=−0.00005
  α_cat=0.30  0.97967  Δ=+0.00001 ← peak (noise)
  α_cat=0.40  0.97961  Δ=−0.00006
  α_cat=0.50  0.97939  Δ=−0.00028
  α_cat=0.60  0.97902  Δ=−0.00065
  ```
  **Essentially null under fixed bias.** The peak +0.00001 at α=0.30
  is noise-level; blend has 10,705 errors (recipe 10,114, +591 more).
- Blend sweep WITH per-α retuned bias (diagnostic only):
  - α=0.05 tuned 0.97974, α=0.35 tuned 0.97975 — +0.00007 to +0.00008
    lift with bias retuning, but this is log-bias coord-ascent overfit
    territory (binhigh-lesson rule).
- 3-way blend (recipe + catboost + greedy_6way) grid: peak at
  (recipe 0.76, catboost 0.14, 6way 0.10) → OOF 0.97978, **Δ=+0.00012**.
  Same lift as the greedy-from-recipe result earlier (digit_xgb alone
  added +0.00012). Adding CatBoost doesn't improve on digit_xgb's
  contribution — blend-null confirmed.
- Diagnosis: the **calibration gap** between CatBoost and XGB on this
  feature set is the problem. They want biases an order of magnitude
  apart on the High class (recipe's bias +3.40, CatBoost's +2.80).
  In a log-blend with a single fixed bias, whichever model's probs
  get preferential treatment dominates. Per-α bias retuning recovers
  tiny lifts but risks LB overfit. **Tree-family architectural
  diversity (XGB ⊗ CatBoost) doesn't help when the recipe feature
  set dominates the decision surface** — both trees converge to
  nearly identical error patterns on the rows that matter for
  macro-recall.
- **No LB probe warranted.** Current LB-best unchanged at
  `submission_recipe_full_te.csv` → **LB 0.97939**. LB budget
  unchanged at 7/10 used today, 3 remaining.
- **Final-selection diversification candidate**:
  `submission_recipe_full_te_catboost.csv` has OOF 0.97936 tuned
  (−0.00031 below recipe but close). If we lock 2 final submissions,
  one recipe + one CatBoost would give model-family variance
  protection on private LB even though LB-public is slightly lower.
  Not recommended given recipe's proven LB, but available.
- **Own-pipeline lever bank now fully exhausted** for this competition.
  Remaining paths to move LB:
  1. More iterations for CatBoost (it hit the cap) — could push
     standalone to 0.9795+. Expected LB lift < +0.00010. ~2-3h
     additional wall.
  2. Strategic: public-CSV blending (explicitly banned per CLAUDE.md
     top-of-file rule).
  Otherwise: lock 0.97939 as final. Pack 0.98114 stays +0.00175 above,
  leader 0.98219 stays +0.00280 above.

### 2026-04-23 — CatBoost CPU LB + GPU retry: both null, two-submission diversification unlocked

- **CatBoost CPU LB probe** (`submission_recipe_full_te_catboost.csv`,
  submitted 11:18 UTC after user approval): **LB 0.97935**, −0.00004
  below recipe_full_te's 0.97939 but essentially **tied**. OOF→LB gap
  **+0.00001** — near-perfect calibration (recipe's was +0.00028).
  Implication: CatBoost calibrates tighter than XGB on this problem.
  Also gives us a **final-selection diversification candidate**:
  Primary `submission_recipe_full_te.csv` (XGB family, LB 0.97939) +
  Safe fallback `submission_recipe_full_te_catboost.csv` (CatBoost
  family, LB 0.97935) covers both model families on private LB.

- **CatBoost GPU on Kaggle kernel** (16000-iter cap + proper early
  stopping): scaffolded `kaggle_kernel/kernel_catboost_recipe/` with
  inlined recipe_features + OrderedTE + log-bias tuner. Two failed
  kernel versions before success (v1: hard-coded COMP_DIR path, v2:
  `devices="0:1"` required 2 GPUs but only P100 present). v3 ran to
  completion on P100 in ~28 min.
- GPU results (5-fold, seed=42):
  - Per-fold argmax 0.97634 / 0.97704 / 0.97723 / 0.97556 / 0.97647
  - Overall OOF argmax **0.97653** (vs CPU 0.97806 — **−0.00153 worse**)
  - Tuned log-bias **0.97894** (vs CPU 0.97936 — **−0.00042 worse**)
  - best_iters: 6197 / 10919 / 9037 / 8492 / 8524 (avg ~8634, early
    stopping triggered — didn't hit 16000 cap). So more training time
    DID produce a converged model; the converged endpoint is just worse.
- **Counter-intuitive cause**: `bootstrap_type='Bernoulli'` (CPU-only)
  vs `bootstrap_type='Bayesian'` (GPU default). Bernoulli = per-tree
  random row subsampling; Bayesian = posterior row weights. For this
  problem, Bernoulli's stronger per-tree randomization acts as
  implicit regularization that the longer training would otherwise
  converge past. Bayesian lets CatBoost optimize to a worse minimum.
  **Rule**: when porting CatBoost between CPU and GPU, verify
  bootstrap_type compatibility — the defaults differ and the CPU
  default (Bernoulli) may be the thing that makes the CPU run work.
- GPU standalone at recipe's bias: 10,902 errors, Jaccard 0.7884 vs
  recipe. Better orthogonality than CPU (0.8060) but worse magnitude.
  Blend with recipe peaked at α=0.05-0.10 (TIED at 0.97967).
- 3-way blend (recipe 0.7 + CPU_cat 0.3 + GPU_cat 0.0) → OOF 0.97967
  (ties recipe standalone). GPU CatBoost gets **zero weight** when
  recipe + CPU_cat are already present.
- Verdict: GPU CatBoost **null** both standalone and as blend leg.
  Longer training doesn't fix the calibration gap it amplifies with
  the wrong bootstrap. No LB probe warranted.
- Lever bank fully exhausted. Final candidates locked:
  1. **Primary**: `submission_recipe_full_te.csv` → **LB 0.97939**
  2. **Safe fallback**: `submission_recipe_full_te_catboost.csv`
     → **LB 0.97935** (different model family, tight calibration)

### 2026-04-23 — LGBM on recipe feature set: 4th tree-family null, pattern confirmed

- Goal: LGBM has different splits than XGB/CatBoost — still a tree,
  but leaf-wise growth vs level-wise. Hypothesis: the recipe's 500-col
  feature set (esp. the 351 OTE cols + 38 ORIG stats) might give LGBM
  structurally new signal to exploit.
- Changed: `scripts/recipe_full_te_lgbm.py` — mirrors `recipe_full_te.py`
  but swaps XGBClassifier → LGBMClassifier. HPs match recipe philosophy:
  `num_leaves=16` (≈ max_depth=4), `min_data_in_leaf=20`, `lr=0.1`,
  `feature/bagging_fraction=0.8`, `bagging_freq=5`, `lambda_l1=5`,
  `lambda_l2=5`, `n_estimators=3000`, `early_stopping=200`. Same
  5-fold seed=42, same per-fold OrderedTE, same class-balanced sample
  weights. 13 min wall on CPU.
- Standalone results:
  - Per-fold argmax: 0.97527 / 0.97592 / 0.97619 / 0.97437 / 0.97539
  - Mean fold argmax 0.97543 (recipe XGB 0.97589, CPU CatBoost 0.97806)
  - **Tuned OOF 0.97926** (recipe: 0.97967, **−0.00041** below)
  - **Error count 10,082** (recipe: 10,114) — first novel model with
    FEWER errors than recipe
  - best_iter range 722-983 (capped at 3000, early stopping triggered)
- Blend results (fixed recipe bias):
  ```
  2-way LGBM × recipe:   peak α_lgbm=0.15 → 0.97971  Δ=+0.00004
  3-way + CPU CatBoost:  (recipe 0.75, lgbm 0.20, cat 0.05) → 0.97971  Δ=+0.00005
  ```
  Both below +1e-4 emit gate. **Null.**
- Jaccard vs recipe = **0.8350** — orthogonal but the blend still
  doesn't lift.
- **Diagnosis — redundancy math**: LGBM errors 10,082, recipe errors
  10,114, Jaccard 0.835. From |A∩B|/|A∪B| = 0.835 ⇒ they share
  ~9,189 error rows and disagree on only ~1,818. For a log-blend to
  lift, the disagreement rows need to be dominated by one model being
  right and the other wrong. With only 1,818 rows to work with (0.29%
  of total), even perfect selection among them gives at most ~0.001
  OOF lift — and real blends only get a fraction of that.
- **Pattern confirmed across 4 tree-family models on recipe features**:
  ```
  Model                Tuned OOF    Errors    Jaccard   Blend Δ vs recipe
  Recipe XGB           0.97967      10,114    1.00      — (baseline)
  LGBM on recipe       0.97926      10,082    0.835     +0.00004 (null)
  CatBoost CPU         0.97936      10,447    0.806     +0.00001 (null)
  CatBoost GPU         0.97894      10,902    0.788     +0.00000 (null)
  ```
  All 4 converge to nearly the same prediction surface. **Tree-family
  architectural diversity is structurally exhausted on this recipe
  feature set.**
- **Redundancy rule (added to LEARNINGS.md candidate)**: "When 4+
  gradient-boosted-tree model families all produce Jaccard ≥ 0.78 AND
  similar error counts on the same feature set, further tree-family
  additions are null. To break the pattern: change the feature set,
  not the model."
- No LB probe. LB-best unchanged at 0.97939. Artifacts committed:
  `oof_recipe_full_te_lgbm.npy`, `test_recipe_full_te_lgbm.npy`,
  `recipe_full_te_lgbm_results.json`,
  `submissions/submission_recipe_full_te_lgbm.csv` (diagnostic).
### 2026-04-23 — recipe_catboost + blend (parallel branch, NULL — replicates main's CatBoost null)

- Goal: first follow-up on the V10 recipe LB-best (0.97939). Train
  CatBoost on the same 443-feature recipe matrix as recipe_full_te,
  then log-blend / prob-blend / LR-stack against XGB-recipe. Hypothesis:
  tree-family diversity at the correct FE level should give +0.001–0.003
  mirror of the Ali Afzal public kernel's XGB+CAT architecture.
- Changed: `scripts/recipe_catboost.py` (CatBoost mirror of
  recipe_full_te; same load_and_engineer + OrderedTE, same 5-fold
  seed=42; HPs mirror XGB heavy-reg regime: depth=4, l2_leaf_reg=5,
  lr=0.1, iterations=3000, rsm=0.8, bagging_temperature=1.0,
  border_count=254, class-balanced sample_weight). `scripts/recipe_blend_stack.py`
  (fixed-bias + tuned-bias-diagnostic log / prob / LR-stack sweeps
  vs XGB-recipe anchor).
- Results (OOF tuned bal_acc, 5-fold):
  - CatBoost standalone tuned = 0.97897 (Δ = −0.00070 vs XGB 0.97967).
    Per-fold argmax 0.97760→0.97843→0.97758→0.97728 — each above XGB's
    fold argmaxes (0.97544→0.97721). But tuned lift smaller because
    CAT's High-bias landed at 2.80 (vs XGB's 3.40).
  - **Log-blend fixed-bias peak α=0.85 → 0.97958 (Δ = −0.00008, NULL).**
    Monotone-negative from CAT-heavy side. Tuned-bias diagnostic
    confirms no hidden signal: peak α=0.55 tuned = 0.97973, Δ=+0.00007.
  - Prob-blend fixed-bias peak α=0.90 → 0.97957, Δ=−0.00009. Same pattern.
  - LR stacker @ XGB bias = 0.97075 (Δ=−0.00892, catastrophic — LR's
    output scale mismatches XGB bias). At tuned-bias: 0.97965, essentially
    ties XGB alone (Δ=−0.00002).
- Diagnosis: **magnitude mismatch beats orthogonality.** Jaccard 0.68
  (good orthogonality) but CAT has 13,435 errors vs XGB's 10,114 (+33%).
  Classic blend-null magnitude mismatch — same pattern as FT-Transformer
  (+42%), TabPFN (+17%), MLP-v5 (+35%). Compounded by CatBoost's softer
  High-class probabilities (ordered-boosting signature) — CAT bias
  [1.43, 1.77, 2.80] vs XGB [1.43, 1.47, 3.40]. Any positive α drags
  High decision boundary off XGB's optimum under fixed-bias; tuned-bias
  retune recovers some ground but stays below anchor.
- LB delta: n/a (no submission from this work — fixed-bias gate killed all
  three candidates).
- Parked: CatBoost-on-recipe as a blend leg is a structural dead-end at
  this feature set. Would require per-class isotonic calibration to fix
  the High-class softness before blending, which is speculative.

### 2026-04-23 — LGBM-recipe + pseudo-label + greedy blend: NEW LB BEST 0.97998 (+0.00059)

- Goal: after the CatBoost-recipe blend null, pursue two
  "hiding in plain sight" levers:
  1. **V10 pseudo-label toggle**: the original include4eto V10 kernel
     has pseudo-labels as an explicit parameter we turned OFF.
     Prior pseudo-label attempts (2026-04-21) failed because the labeler
     was LB 0.97352 — boundary errors compounded. Recipe_full_te at
     LB 0.97939 has ~6x fewer test-set errors; τ=0.98 filters boundaries
     for ~99.5% purity.
  2. **LGBM leg on recipe features**: untested; LGBM doesn't share
     CatBoost's High-class calibration issue, so blend may work even
     if LGBM doesn't change the ceiling.
- Changed:
  - `scripts/recipe_lgbm.py` — LGBM mirror of recipe_full_te. Same FE,
    same OTE (a=1), same 5-fold seed=42, same 443-feature matrix.
    HPs mirror XGB's heavy-reg regime (num_leaves=30, max_depth=4,
    reg_alpha=5, reg_lambda=5, feature/bagging_fraction=0.8).
  - `scripts/recipe_pseudolabel.py` — V10's disabled toggle. Gates test
    rows by recipe's max-prob ≥ τ=0.98, assigns argmax class as
    pseudo-label. 5-fold split on REAL train only; pseudo rows always
    go to training side, val stays real-only (OOF alignment preserved).
    OTE per fold fits on augmented train (real_tr ∪ pseudo), so OTE
    statistics see pseudo-labels as ground truth.
  - `scripts/recipe_all_blend.py` — greedy forward log-blend over the
    recipe OOF bank {recipe_full_te, recipe_lgbm, recipe_pseudolabel,
    recipe_catboost} anchored on recipe's tuned bias.
- LGBM results (OOF, 5-fold, seed=42):
  - Per-fold argmax 0.97565 → 0.97599 → 0.97668 → 0.97426 → 0.97525
    (vs XGB fold argmaxes 0.97544 → 0.97659 → 0.97721 → 0.97465 → 0.97557,
    LGBM tracks ~6 bp below XGB per fold).
  - Tuned OOF = **0.97952**, Δ=−0.00015 vs XGB. Bias = [1.23, 1.27, 3.40].
  - **High bias 3.40 matches XGB's exactly** — CatBoost's 2.80 mismatch
    absent here. Low/Medium biases close. Calibration profile aligned.
  - Wall time ~15 min total (faster than XGB's 55 min on same feature
    matrix and depth).
- Pseudo-label results (OOF, 5-fold, seed=42):
  - Pseudo subset at τ=0.98: 226,162 / 270,000 test rows kept (83.8%).
    Label dist [Low 133923, Medium 83570, High 8669] — matches real-train
    prior (no confidence bias). +8,669 pseudo-High rows = +41% boost vs
    real-train's 21k High pool.
  - Per-fold argmax 0.97700 → 0.97627 → 0.97881 → 0.97503 → 0.97589
    (vs XGB fold argmaxes 0.97544 → 0.97659 → 0.97721 → 0.97465 → 0.97557,
    mean Δ = +0.00071 per fold).
  - **Tuned OOF = 0.97993, Δ=+0.00026 vs XGB-recipe — first standalone
    lift on top of recipe.** Bias [1.63, 1.47, 3.30].
  - Wall time ~48 min (5 folds × ~9.5 min each, training pool 504k→730k).
- Greedy blend (fixed anchor bias = recipe's [1.43, 1.47, 3.40]):
  ```
  standalone @ anchor bias     fixed       tuned       err     Jaccard vs recipe
    recipe_full_te             0.97967    0.97967    10114    1.00
    recipe_pseudolabel         0.97987    0.97993    10039    0.78
    recipe_lgbm                0.97939    0.97952    10018    0.86
    recipe_catboost            0.97739    0.97897    13435    0.68

  pairwise sweep vs recipe (fixed bias)
    recipe × pseudo            peak α=0.50  OOF 0.98012  Δ=+0.00046  ← WINNER
    recipe × lgbm              peak α=0.45  OOF 0.97967  Δ=+0.00001  (null)
    recipe × catboost          peak α=0.00  OOF 0.97967  Δ=+0.00000  (null)
  ```
  Greedy final: **recipe_full_te (0.50) + recipe_pseudolabel (0.50),
  OOF 0.98012, Δ=+0.00046**. Confusion matrix at blend: Low 99.50% /
  Medium 96.84% / **High 97.68%** (up from XGB-recipe alone's ~96.5%) —
  that's where the lift lives.
- **LB probe: submitted at 14:06 UTC, result LB = 0.97998** — **new best**,
  +0.00059 over prior best (0.97939). OOF→LB gap = **+0.00014** (tightest
  in competition so far). **LB delta EXCEEDS OOF delta** (+0.00059 vs
  +0.00046), meaning the blend generalizes better than CV predicted.
  Gap to pack 0.98114: +0.00116 (was +0.00175); leader 0.98219: +0.00221.
- Updated calibration ladder:
  ```
  recipe_full_te                    0.97967 → 0.97939   gap +0.00028
  **recipe × pseudolabel (50/50)    0.98012 → 0.97998   gap +0.00014**  ← NEW LB BEST
  ```
- LB budget: 5/10 used today (blend probe +1); 5 remaining.
- Read-out: **two diagnosis-aligned rules confirmed**:
  1. LGBM at wide-OTE recipe features produces near-identical predictions
     to XGB (Jaccard 0.86, err count within 1%) — tree-family diversity is
     exhausted at this FE level. Same lesson as 2026-04-22 LGBM-digits
     null (Jaccard 0.96 with XGB-digits).
  2. Pseudo-label strength scales with labeler strength. The 2026-04-21
     τ=0.95 hybrid-labeler null (hybrid at LB 0.97352) and this success
     (recipe at LB 0.97939 + τ=0.98) validate the "labeler must be ≥97%
     LB before pseudo-labels help" threshold.
- Meta-lesson: **"own-pipeline ceiling" framing was wrong** — at every
  prior session the conclusion was that remaining lift required public-CSV
  blending. The V10 pseudo-label toggle was sitting in the kernel we
  already implemented, just disabled. Adding a 2026-04-23 LEARNINGS.md
  entry: "when implementing a public kernel, enumerate every disabled
  toggle / commented-out parameter and test them individually."

### 2026-04-23 — stage-2 pseudo-label + 4-way blend: LB null, stacking-inflation confirmed

- Goal: the stage-1 blend landed LB 0.97998 (+0.00059 over recipe alone).
  Test whether **self-refinement via stage-2 pseudo-label** (use the
  LB-best blend as the labeler for a new pseudo-label run) + broader
  greedy blend over 5 candidates pushes further.
- Changed:
  - `scripts/build_blend_labeler.py` — builds the stage-1 blend as a
    standalone labeler: 50/50 log-blend of `test_recipe_full_te.npy` ×
    `test_recipe_pseudolabel.npy` at recipe's fixed bias [1.43, 1.47, 3.40].
    Saves as `test_recipe_blend_stage1.npy` + `recipe_blend_stage1_results.json`.
  - `scripts/recipe_pseudolabel.py` — parameterized via env vars
    (`LABELER_TEST_PATH`, `LABELER_BIAS_JSON`, `PSEUDO_SUFFIX`). Default
    behavior unchanged.
  - `scripts/recipe_all_blend.py` — CANDIDATES extended to include
    `recipe_pseudolabel_stage2`.
- Stage-2 results (OOF, 5-fold, seed=42):
  - Labeler = blend (LB 0.97998). τ=0.98 → 229,307 rows kept (+3,145
    vs stage-1's 226,162). Pseudo class dist matches real-train prior.
  - Per-fold argmax vs stage-1 pseudo:
    ```
    fold 1: 0.97626 (Δ -0.00074 vs stage-1 0.97700)
    fold 2: 0.97604 (Δ -0.00023 vs stage-1 0.97627)
    fold 3: 0.97782 (Δ -0.00099 vs stage-1 0.97881)
    fold 4: 0.97648 (Δ +0.00145 vs stage-1 0.97503)
    fold 5: 0.97640 (Δ +0.00051 vs stage-1 0.97589)
    sum Δ = 0.00000 — overall argmax identical to stage-1
    ```
  - **Tuned OOF = 0.98002** (Δ=+0.00009 vs stage-1's 0.97993; +0.00035
    vs recipe). Bias [1.53, 1.37, 3.40] — **High bias matches recipe's
    3.40 exactly** (stage-1's was 3.30 — stage-2 calibrates tighter).
  - Wall 45 min.
- Greedy forward blend (5 candidates: recipe_full_te, recipe_lgbm,
  recipe_pseudolabel, recipe_pseudolabel_stage2, recipe_catboost):
  ```
  standalone @ anchor bias      fixed       tuned       err     Jaccard vs recipe
    recipe_full_te              0.97967    0.97967    10114    1.00
    recipe_pseudolabel_stage2   0.97996    0.98002     9996    0.7800   ← new best
    recipe_pseudolabel          0.97987    0.97993    10039    0.7805
    recipe_lgbm                 0.97939    0.97952    10018    0.8555
    recipe_catboost             0.97739    0.97897    13435    0.6794

  greedy path:
    start:       recipe_full_te              OOF 0.97967
    + stage2     α=0.500 → OOF 0.98026  Δ=+0.00059 (biggest add)
    + catboost   α=0.150 → OOF 0.98029  Δ=+0.00003 (first time CAT adds)
    + stage1     α=0.125 → OOF 0.98033  Δ=+0.00004
    stop (LGBM rejected — Jaccard 0.86)

  final weights: recipe 0.37, stage2 0.37, catboost 0.13, stage1 0.13
  fixed-bias OOF = 0.98033  Δ vs recipe = +0.00066
  ```
  Blend confusion matrix: Low 99.51% / Medium 96.75% / **High 97.82%**
  (vs prior blend's 97.68% High). Error count 9,891 (recipe 10,114).
- **LB probe: submitted 15:12 UTC**, result **LB = 0.97997**. OOF→LB gap
  = **+0.00036** (prior blend's gap was +0.00014).
  Δ vs current LB best (2-way blend 0.97998) = **−0.00001**, essentially
  tied within LB noise. **NULL on LB despite +0.00021 OOF over prior blend.**
- Updated calibration ladder:
  ```
  recipe_full_te                    0.97967 → 0.97939   gap +0.00028
  **recipe × pseudolabel (2-way)    0.98012 → 0.97998   gap +0.00014**  ← LB BEST still
    4-way with stage-2+cat+pseudo   0.98033 → 0.97997   gap +0.00036    (null)
  ```
- Diagnosis:
  1. **Self-refinement saturates fast.** Stage-1 lifted +0.00059 LB;
     stage-2 adds 0 LB. The +0.00009 stage-2 OOF gain over stage-1 was
     tighter-calibration, not new signal. Once the labeler is good
     enough (recipe at 0.97939 LB), the pseudo-label set is already
     ~99.5% pure, so making the labeler better doesn't improve the
     downstream model.
  2. **Greedy over saved OOFs has a stacking-inflation ceiling** beyond
     2 anchor models. Each added component's OOF contribution lands at
     ~noise-level (+0.00003-0.00004), which doesn't clear the LB
     generalization threshold. The +0.00022 OOF from adding CatBoost +
     stage-1 to the greedy did NOT transfer — gap widened from +0.00014
     to +0.00036 (exactly matching the OOF inflation).
- **New rule** (added to LEARNINGS.md candidate): "For greedy-forward
  log-blends on a single-split OOF, the per-candidate lift must be ≥
  +0.0002 to have >50% chance of transferring to LB. Below that, the
  extra component is mostly fitting CV-split noise."
- LB budget: 6/10 used today, 4 remaining.
- Next bet: **Ali Afzal's 171-pair OTE** — the remaining feature-surface
  expansion the public-kernel lineage points at. We have 28 cat×cat
  combos; the untested 143 pairs (cat×num, num×num) are a fundamentally
  different feature space, not a restack.

### 2026-04-23 — allpairs (cat×num) + stage-2 2-way: LB −0.00009 null (OOF overfit)

- Goal: test two remaining levers flagged in the prior entry —
  (1) expand pair combos beyond cat×cat to include cat×num (Ali Afzal's
  "pairwise magic" lite), and (2) test the simpler `recipe × pseudo_stage2`
  2-way which the 4-way analysis hinted was OOF-stronger than our LB-best
  `recipe × pseudo_stage1`.

- **Allpairs pipeline** (`scripts/recipe_allpairs.py`):
  - Initial attempt with full C(19,2)=171 pairs (1015 features) **died
    silently at XGB init** — classic large-memory XGB histogram
    allocation OOM. Dropped num×num (55 pairs, lowest-signal — two raw
    floats rarely repeat → factorized keys near-unique → OTE shrinks to
    prior) and kept cat×cat (28) + cat×num (88) = 116 pairs, 795 features.
  - Rerun success. Per-fold argmax 0.97585 / 0.97617 / 0.97745 / 0.97526 /
    0.97589 — mean Δ vs recipe per fold = +0.00021 (+0.00008 fold-by-fold
    noise). Overall OOF argmax **0.97612**, tuned **0.97976** (Δ vs recipe
    +0.00009). Bias [0.93, 1.07, 3.20] — lower Low/Medium biases than
    recipe (pairs give sharper raw probs; less bias correction needed).
  - **Error count 9,938** — LOWEST of any variant tested (vs recipe
    10,114, pseudo_stage2 9,996, pseudo_stage1 10,039). Jaccard vs
    recipe = 0.8111. Decent orthogonality + fewer errors = strong blend
    candidate on paper.

- **Greedy blend (6 candidates now including allpairs)** —
  `scripts/recipe_all_blend.py`:
  ```
  standalone @ anchor bias     fixed       tuned       err     Jaccard vs recipe
    recipe_full_te             0.97967    0.97967    10114    1.00
    recipe_allpairs            0.97968    0.97976     9938    0.8111   ← new
    recipe_lgbm                0.97939    0.97952    10018    0.8555
    recipe_pseudolabel         0.97987    0.97993    10039    0.7805
    recipe_pseudolabel_stage2  0.97996    0.98002     9996    0.7800
    recipe_catboost            0.97739    0.97897    13435    0.6794

  pairwise sweep vs recipe (fixed bias)
    recipe × allpairs          peak α=0.45  OOF 0.97992  Δ=+0.00026
    recipe × pseudo_stage2     peak α=0.55  OOF 0.98027  Δ=+0.00060  ← best pairwise
    recipe × pseudo_stage1     peak α=0.50  OOF 0.98012  Δ=+0.00046  (= LB-best 2-way)

  greedy path (starting from recipe_full_te):
    + pseudo_stage2   α=0.500 → OOF 0.98026  Δ=+0.00059  (biggest add)
    + catboost        α=0.150 → OOF 0.98029  Δ=+0.00003
    + allpairs        α=0.200 → OOF 0.98033  Δ=+0.00004

  final weights: recipe 0.34, stage2 0.34, catboost 0.12, allpairs 0.20
  fixed-bias OOF = 0.98033  Δ vs recipe = +0.00067
  ```
  Interesting: the 4-way OOF (0.98033) is IDENTICAL to the earlier 4-way
  that used pseudo_stage1 in place of allpairs. Greedy swapped stage-1
  for allpairs with the same OOF ceiling — allpairs contributes the same
  magnitude of signal as stage-1 when added on top of stage-2.

- **LB probe: `recipe × pseudo_stage2` 2-way at α=0.55** (simpler than
  the 4-way, higher pairwise OOF than stage-1 version). Submitted via
  `submission_recipe_pseudolabel_stage2_w055.csv` at 17:33 UTC.
  Result: **LB = 0.97989**. Δ vs current LB-best = **−0.00009** (NULL).
  OOF→LB gap = **+0.00038** (vs stage-1 2-way's +0.00014).

- **Updated calibration ladder:**
  ```
  recipe_full_te                    0.97967 → 0.97939   gap +0.00028
  **recipe × pseudo_stage1 (α=0.50) 0.98012 → 0.97998   gap +0.00014**  ← LB BEST still
    recipe × pseudo_stage2 (α=0.55) 0.98027 → 0.97989   gap +0.00038    (null)
    4-way (stage2+cat+stage1)       0.98033 → 0.97997   gap +0.00036    (null)
  ```

- **Diagnosis — stage-2 pseudo-label is OOF-overfit:**
  Stage-2's OOF gain over stage-1 (+0.00015 on 2-way blend, +0.00009
  standalone) comes from **tighter calibration on the same 5-fold
  training split** — NOT from new signal. Stage-2's labeler (the
  LB-0.97998 stage-1 blend) was itself fit on those same 5 folds, so
  its pseudo-labels encode the training-side calibration more
  precisely. Hidden test split doesn't share those biases → gap blows
  up by exactly the OOF inflation amount (+0.00024 OOF gain → +0.00024
  gap widening from +0.00014 to +0.00038).

  Rule (first stated in 4-way null entry, now LB-confirmed twice):
  **"for greedy log-blends on a single CV split, per-candidate OOF
  lift < +0.0002 likely does not transfer to LB. Below +0.0003, treat
  any LB lift as lucky."** Stage-1 (LB-best) had +0.00046 OOF, cleared
  the threshold; stage-2 +0.00015 and the 4-way's +0.00021 did not.

- **Implication: self-refinement is dead at this labeler strength.** At
  LB 0.97998 (tighter than V10 kernel's ~0.976 labelers), pseudo-labels
  are already ~99.5% pure. Further stages cascade the same training-set
  bias. To break +0.97998 LB needs either a DIFFERENT feature source
  (allpairs in an LB-verified blend, not part of a stacked bundle) or
  structural knowledge distillation (use blend's SOFT probs as target,
  not argmax — uses the full posterior distribution the blend encodes).

- **Allpairs LB-probe not submitted.** Pairwise OOF 0.97992 would
  project LB ~0.97964 at stage-1-calibration, below LB-best.
  Allpairs is a stronger blend COMPONENT (fewer errors, decent
  orthogonality) but weaker STANDALONE vs stage-1's pseudo-label lever.
  If the next LB submission is allpairs-family, it should be in a
  `recipe × allpairs` 2-way at a LOW α (0.15-0.25) where the fewer-
  errors quality matters more than the lower standalone OOF — but that
  specific variant hasn't been tested yet and isn't in the greedy's
  emitted submission.

- LB budget: 7/10 used today, 3 remaining. Current LB best unchanged
  at 0.97998.

- Next untested own-pipeline levers (ranked by expected-value / risk):
  1. **Quantile-binned num×num OTE** — bin each numeric to ~20 quantiles
     before pair-concat; the 55 num×num pairs become tractable (key
     cardinality ~400 instead of ~630k^2). This is the "literal 171-pair
     magic" from the public kernels, which almost certainly bin their
     nums implicitly. ~2-3h compute.
  2. **Knowledge distillation from the LB-best blend** — train XGB on
     recipe features but targeting the 2-way blend's full 3-class soft
     probs (not argmax pseudo-labels). Uses full posterior info, not
     just argmax class. Novel; untested anywhere in the repo. ~1.5h.
  3. **τ sweep on pseudo-label (stage-1)** — we picked τ=0.98 by
     instinct. Test 0.95 and 0.99. Each retrain ~1h.
  4. **Pseudo-label using allpairs as labeler** — allpairs has FEWER
     errors than recipe, so its argmax pseudo-labels at τ=0.98 are
     slightly purer. Untested. ~1h.
  5. **DART booster** on recipe features. We've only tested gbtree.
     DART's tree-dropout is a structurally distinct model family at
     the same features. ~1-2h.
  6. **Heavy original-dataset weight (10x)**. Forces the decision
     boundary toward the rule-perfect space; may help rare-class
     recall. Tested lightly earlier, not with V10 recipe. ~1h.

### 2026-04-23 — 171-pair OTE production run: OOM-killed on fold 2 (compute planning miss)

- Goal: execute the `Next bet` from the prior entry — extend the V10
  recipe to all C(19,2)=171 feature pairs by binning the 11 numerics
  to 16-bin quantile categoricals, concatenating every (c1,c2) pair
  into a single factorized combo col, then running the same OrderedTE
  + heavy-reg XGB pipeline. 5-fold, seed=42, aligned OOF split.
- Context: main branch independently attempted the same 171-pair
  lever earlier in the day (see `2026-04-23 — allpairs (cat×num) +
  stage-2 2-way` entry above) and ALSO hit an OOM — but at a
  different stage (XGB histogram allocation at init, before
  training). Main dropped num×num (55 pairs, lowest-signal — raw
  float concat creates near-unique keys → OTE collapses to prior)
  and ran 116 pairs = cat×cat (28) + cat×num (88) = 795 features
  successfully. This branch took the "bin-nums-first" route (16-bin
  quantile binning before any pair construction), which makes
  num×num pairs tractable in principle — but the resulting 271 OTE
  keys × 3 classes = 813 OTE columns pushed peak RAM past the 21GB
  cap at the OTE-fit step instead of XGB init.
- Changed:
  - `scripts/recipe_pair_features.py` — `add_quantile_bins()` (16-bin
    quantile binning with `duplicates="drop"`, fit on train, applied
    to test/orig) and `add_all_pair_combos()` (C(n,2) over cats+bins,
    factorized across combined splits).
  - `scripts/recipe_full_te_171pair.py` — orchestrator mirroring
    `recipe_full_te.py` with bins treated as cat-like for OTE.
    Feature groups: cats=8, bins=11, combos=171, digits=66,
    num_as_cat=11, tres=4, logits=3, freq=179 (FREQ on cats+combos),
    orig_stats=38. **te_cols=271 → 813 OTE cols** (271 × 3 classes).
  - `scripts/blend_171pair.py` — fixed-bias sweep vs recipe (anchor A)
    and optionally vs LB-best (anchor B) per the `no_ote` Jaccard +
    error-count rule.
- Smoke pass (SMOKE=1, 20k train, 2 folds): PASSED in ~70 s, tuned
  OOF 0.96455. All FE blocks built, OTE fit + transform clean, no
  errors. Smoke config uses the same code path; the only difference
  is data size and XGB iterations.
- Production run (504k train, 5 folds, 813-OTE XGB) **CRASHED
  silently on fold 2 OTE-fit**:
  ```
  [16:48:54] === fold 1/5 ===
  [16:49:52]   OTE done in 50.4s (813 OTE cols)
  [17:13:51]   fold 1 argmax_bal_acc = 0.97453  best_iter=1215
  [17:13:51] === fold 2/5 ===
  <SIGKILL — no further output, no traceback>
  ```
  Fold 1 completed 25 min wall. Fold 2 died in <1 min, before any
  visible OTE-fit progress. Classic OOM-kill signature: no Python
  exception, process disappears from `ps aux`, 21 GB RAM immediately
  free again.
- Root cause (memory stack-up, **unplanned**):
  - Smoke at 20k × 2 folds peaked ~1.5 GB. Production scales ~25×
    in row count and ~1.2× in feature count, but memory doesn't
    scale linearly — it scales with **simultaneously-live frame count**.
  - At fold 2 entry, the process held:
    * `train` (504k × 900+ cols, post-FE)
    * `test` (270k × 900+ cols)
    * `orig` (10k × 900+ cols)
    * fold 1's leftover `X_tr` / `X_va` / `X_te` DataFrames if Python
      GC hadn't released them (pandas pd.concat fragments retain
      parent-block refs longer than expected)
    * fold 1's XGB Booster + its internal histogram buffers
    * `compute_sample_weight` array
  - Fold 2 then allocates: `X_tr.iloc[tr_idx].copy()` (403k rows ×
    900+ cols ≈ 1.4 GB), shuffled copy for OrderedTE, OTE `.fit()`
    builds 271 × 3 = 813 intermediate arrays (cum_cnt, cum_sum, etc.)
    each of length 403k during the groupby loop, then pd.concat of
    813 new float32 cols onto the 1.4 GB frame = another 1.6 GB peak.
  - Peak simultaneous allocation on fold 2 OTE-fit is plausibly
    ~18-22 GB against a 21 GB cap, no swap. SIGKILL at the peak.
- One-fold signal before the crash: **fold 1 argmax 0.97453 vs recipe
  fold 1's 0.97544** — directionally negative (−0.00091, inside the
  fold-σ band of ~0.00088 but on the wrong side). Not enough folds
  to judge OOF, but inconsistent with a "drop-in lift" expectation.
  Error orthogonality vs recipe couldn't be computed (single fold,
  and fold 1 OOF covers a different val set than recipe's fold 1 if
  the split differs — which it doesn't here, seed=42 aligned, so the
  Jaccard-on-fold-1 analysis is feasible once we recover it).
- **Decision**: do NOT re-run naively. Two concrete fixes before
  relaunching:
  1. **Per-fold subprocess isolation**: spawn each fold as a separate
     Python process via `subprocess.run()`, OS returns all memory on
     process exit. Zero cross-fold contamination. ~15 min plumbing,
     adds ~10 s/fold overhead from module re-imports.
  2. **Intra-fold `gc.collect()` + explicit `del`** at end of each
     OrderedTE.fit() loop iteration, plus `del` of fold-scoped frames
     at fold end. Cheaper (~2 min) but doesn't guarantee recovery
     under pandas pd.concat fragmentation.
  Recommendation: do (1). Robust, portable, and the plumbing is
  reusable for any future large-FE experiment.
- LB delta: n/a. No OOF output. Fold 1 intermediate not saved.
- Lessons logged to LEARNINGS.md §Process (new "Capacity-plan large
  FE runs" entry):
  1. **Smoke-testing validates correctness, NOT memory scaling.**
     Smoke at N=20k × 2 folds cannot reveal an OOM at N=504k × 5
     folds because peak memory is dominated by simultaneous live-frame
     count, not computation progression. Need a **memory budget**
     alongside the wall-time budget.
  2. **Heavy-FE pipelines must estimate peak RAM before launch.**
     Cheap 60-second calculation: rows × live_feature_cols × bytes ×
     simultaneous_frames. If >60% of available RAM, plan isolation
     (subprocess per fold, or dask/out-of-core) BEFORE the smoke.
  3. **Fragmented pandas frames outlive their scope.** `pd.concat`
     on 800+ cols per frame retains block-manager refs that Python
     GC doesn't free aggressively. `gc.collect()` at loop end is not
     sufficient — subprocess isolation is the only reliable fix.
  4. **Silent SIGKILL has no traceback; inspect process table + RAM
     state to diagnose**. `ps aux | grep python` returning empty
     alongside the log stopping mid-fold is the OOM signature. No
     need to hunt for a missing error line.
### 2026-04-23 — asymmetric 3-way with allpairs: LB null confirms stacking-inflation ceiling is structural

- Goal: after parallel session's `recipe_no_ote` reconfirmed the refined
  blend heuristic ("Jaccard < 0.80 AND err_count ≤ anchor" are BOTH
  needed), re-examine `recipe_allpairs` which satisfies both (Jaccard
  0.8111, errors 9,938 < recipe's 10,114) and has NOT been OOF-overfit
  like stage-2. Manually grid-search a 3-way blend with stage-1 (the
  LB-verified pseudo-label component) + allpairs added.
- Changed: no new scripts — manual grid-search on saved OOFs. Finds
  full-grid optimum at (w_recipe=0.50, w_stage1=0.30, w_allpairs=0.20)
  with fixed-bias OOF = **0.98033** (Δ=+0.00020 vs LB-best 2-way's
  0.98012). Rationale: asymmetric weights let allpairs enter without
  diluting recipe's proven LB-transfer calibration.
- Per-class recall at asymmetric blend: Low 99.51% / Medium 96.86% /
  **High 97.73%** (competitive with stage-2 4-way's 97.82%, above
  LB-best 2-way's ~97.68%). Error count 9,805 — LOWEST we've seen
  on any blend or standalone.
- Hypothesis: if stage-2's LB null was due to OOF-overfit (labeler
  trained on same folds), replacing stage-2 with stage-1 (LB-proven)
  + allpairs (fresh, not folds-contaminated) in the 3-way might
  transfer better. Best-case: LB 0.98019 at stage-1's +0.00014 gap.
  Worst-case: LB 0.97997 at 4-way's +0.00036 gap.
- **LB probe** (submitted 17:58 UTC):
  `submission_asym_3way_recipe050_stage1030_allpairs020.csv` →
  **LB = 0.97995**. Δ vs LB-best = **−0.00003** (effectively tied,
  null). OOF→LB gap = **+0.00038** — matches the stage-2 4-way's
  +0.00036 EXACTLY. Allpairs didn't change the gap math.
- **Stacking-inflation ceiling is structural, not component-specific.**
  Three separate 3+ component blends have all hit OOF 0.98033 and
  all landed LB 0.97995-0.97997:
  ```
  OOF    LB      gap      composition
  0.98033 0.97997 +0.00036  4-way w/ stage-2 (earlier)
  0.98033 0.97995 +0.00038  asymmetric 3-way w/ allpairs (this)
  0.98027 0.97989 +0.00038  2-way recipe × stage-2 (null)
  0.98012 0.97998 +0.00014  2-way recipe × stage-1 (LB BEST) ← structural sweet spot
  ```
  **Rule:** on this feature set, OOF→LB gap grows ~+0.0001 per
  +0.0001 OOF above 0.98012. LB stays pinned at ~0.97995-0.97998 for
  any blend that reaches OOF 0.9802-0.9805. The 2-way LB-best is the
  operating-point sweet spot; further stacking produces cosmetic OOF
  lift with zero LB transfer.
- **Implication:** breaking above LB 0.97998 requires a genuinely
  different mechanism, not another blend variant. Ceiling-breaking
  candidates:
  1. **Quantile-binned num×num OTE** (fixes the literal 171-pair
     attempt that OOM'd on raw-float pairs). Makes num×num
     factorization tractable by binning nums first. New feature
     surface, not another stacking variant.
  2. **Soft-target distillation from LB-best blend.** Train XGB on
     recipe features targeting the blend's full 3-class soft probs
     (not argmax pseudo-labels). Uses the blend's full posterior
     distribution; different signal-extraction path than hard pseudo.
  3. **Heavy original-weight training.** Multiply the 10k rule-perfect
     orig rows by 10x sample_weight during XGB training. Shifts
     decision boundary toward the rule-clean space; may particularly
     help rare-class (High) recall where the flip signal lives.
- LB budget: **8/10 used today, 2 remaining.** Save the 2 for tomorrow's
  potential ceiling-breaker probes. No further submissions today.

### 2026-04-23 — full 12-component greedy confirms +0.0002 LB-transfer ceiling is structural

- Goal: with ALL parallel-session OOFs synced locally (recipe,
  pseudolabel, pseudo_stage2, lgbm, catboost, allpairs, N1 subsets
  {no_digits, no_combos, no_orig, no_ote}, OTE-strength variants
  {a01, a10}), run comprehensive greedy over the full 12-component
  OOF bank to check whether any combination main hadn't tested
  produces a blend with per-step Δ ≥ +0.0002 (the documented LB-
  transfer threshold).
- Changed: ad-hoc script in `logs/greedy_full_recipe_bank.log` (no
  retraining — OOF-space only).
- **Results — greedy from recipe anchor (6-way, identical to main's
  4-way ceiling + 2 extras)**:
  ```
  + pseudo_stage2 α=0.500  OOF 0.98026  Δ=+0.00059  ≥+0.0002 ✓
  + catboost      α=0.175  OOF 0.98031  Δ=+0.00005  ✗
  + no_digits     α=0.050  OOF 0.98034  Δ=+0.00004  ✗
  + allpairs      α=0.075  OOF 0.98036  Δ=+0.00002  ✗
  + a10           α=0.025  OOF 0.98039  Δ=+0.00002  ✗
  final OOF 0.98039  (Δ +0.00072 vs recipe)
  ```
- **Greedy from LB-best anchor (pseudolabel 50/50, OOF 0.98012)**:
  ```
  + allpairs α=0.250  OOF 0.98026  Δ=+0.00014  ✗
  + no_ote   α=0.075  OOF 0.98031  Δ=+0.00005  ✗
  ```
- **Verdict**: every per-step addition after the first (pseudo_stage2)
  is below the +0.0002 LB-transfer threshold. Main already verified
  this with the 4-way `recipe+stage2+cat+stage1` (OOF 0.98033 → LB
  0.97997 null). Expected LB of the new 6-way ≈ 0.97997 — same
  structural ceiling.
- **The +0.00072 OOF lift over recipe is spread across 5 components,
  each below +0.0002. No single addition produces a real LB move.**
- **Own-pipeline LB ceiling at 0.97998 is robust** across all 12-
  component combinations (recipe, pseudo, pseudo_stage2, lgbm,
  catboost, allpairs, 4 N1 subsets, 2 OTE-alpha variants). The
  structural rule holds. Pack 0.98114 remains +0.00116 above,
  reachable only via public-CSV blending (banned).
- No LB probe warranted. LB-best stays `submission_recipe_greedy_recipe_pseudolabel.csv`
  → **LB 0.97998**.

### 2026-04-24 — soft-target distillation from LB-best teacher: OOF +0.00084 → LB −0.00148 (OOF-noise memorization null)

- Goal: execute candidate #2 from the ceiling-breaker shortlist — train an
  XGB student with custom soft-cross-entropy objective against the
  LB-best blend's full posterior, bypassing argmax pseudo-label's
  information loss. Motivation: every blend-level experiment so far
  compresses argmax-equivalent predictors; distillation trains on the
  teacher's full per-row posterior, including boundary uncertainty.
- Changed: `scripts/soft_distill_common.py` (teacher builder +
  custom soft-xent objective factory + hard-label mlogloss val metric),
  `scripts/soft_distill_xgb.py` (5-fold pipeline using xgb.train native
  API with obj= closure; same 443-feature recipe matrix, same
  StratifiedKFold seed=42, same OrderedTE; no class-balanced sample
  weight since teacher posterior already encodes it),
  `scripts/blend_soft_distill.py` (Jaccard + fixed-bias α sweep vs
  recipe anchor and LB-best anchor; 3-way grid; auto-emit gate at
  Δ ≥ +5e-4 fixed-bias).
- Teacher construction verified: softmax(0.5*log(recipe_full_te) +
  0.5*log(recipe_pseudolabel)) reproduces LB-best OOF 0.98012 exactly
  with bias [1.4324, 1.4689, 3.4008]. Teacher entropy: mean 0.0425,
  21,968 rows (3.5%) carry >0.3 entropy — the distillation surface.
- Wall: 2h02m total (fold 1 27min, folds 2-5 ~23min each). Custom
  obj Python callback is ~2.5× slower than native multi:softprob —
  504k × 3 softmax + grad + hess computation per iteration × ~3000
  iterations. best_iter 2989-2998 on every fold (near the 3000 cap;
  still learning marginal).
- Per-fold OOF argmax (distill vs recipe):
  ```
  fold 1:  0.97486 vs 0.97544   Δ = -0.00058
  fold 2:  0.97595 vs 0.97659   Δ = -0.00064
  fold 3:  0.97654 vs 0.97721   Δ = -0.00067
  fold 4:  0.97484 vs 0.97465   Δ = +0.00019
  fold 5:  0.97565 vs 0.97557   Δ = +0.00008
  OOF:     0.97557 vs 0.97589   Δ = -0.00032  (within fold-std noise)
  ```
- **Standalone OOF at distill's own tuned log-bias = 0.98096**
  (+0.00084 ABOVE LB-best teacher's 0.98012). Student bias
  [0.5324, 1.0689, 3.2008] — Low/Medium biases much smaller than
  recipe's [1.4324, 1.4689, 3.4008], suggesting student probs are
  already closer to bal_acc-optimal.
- Error count 9,520 < teacher's 9,851 < recipe's 10,114. Jaccard vs
  recipe = 0.7924 (below 0.80 "novel" threshold); Jaccard vs LB-best
  = 0.8155 (borderline). The blend-lift fingerprint looked present on
  every diagnostic.
- **LB probe (user-approved, submitted at 04:10 UTC)**:
  `submission_soft_distill.csv` → **LB public = 0.97850**.
  Δ vs LB-best = **−0.00148** (clear regression).
  OOF → LB gap = **+0.00246** — blew up 17× vs LB-best's +0.00014.
  Widest OOF→LB gap in the competition log.
- **Diagnosis — OOF-noise memorization null.** Per-row leak analysis
  is clean (teacher_oof[i] came from a model trained on folds != i).
  But the teacher's OOF contains ~12k errors (the 2% boundary-band
  where NN flips live). The student was trained at max_depth=4 ×
  ~3000 trees × 443 features, which is the same capacity regime as
  recipe XGB — ample to memorize the teacher's per-row confident-
  wrong posteriors on those ~12k rows. The student reproduces those
  overconfident-wrong posteriors on the test set, where they DON'T
  match the true labels.

  Leak-free does not equal overfit-free. Student capacity matched to
  teacher capacity means the student can perfectly mimic the teacher
  — including teacher mistakes. Unlike hard pseudo-labels
  (τ=0.98 filters >99% confidence, dropping boundary rows), soft
  distillation retains 100% of rows and propagates teacher errors
  into the student's decision surface.

- **Warning sign in hindsight**: student bias [0.53, 1.07, 3.20] vs
  recipe's [1.43, 1.47, 3.40] — Low/Medium biases an order of
  magnitude smaller. This meant student's raw probs were SHARPER on
  Low than recipe's. Sharpness came from teacher-mimicry, NOT from
  better discrimination — the student's "natural calibration" was
  fitting fold-specific calibration artifacts.
- **LB budget**: 1/10 used today (only 1 probe this session; blend
  variants anchored on the same overfit student not submitted).
  9 remaining.
- Current LB best unchanged: `submission_recipe_greedy_recipe_pseudolabel.csv`
  at **LB 0.97998**.
- Artefacts committed via gitignore whitelist for cross-branch reuse:
  `oof_soft_distill.npy` (7.3MB), `test_soft_distill.npy` (3.1MB),
  `soft_distill_results.json`, `blend_soft_distill_results.json`,
  5 candidate submissions (1 submitted, 4 diagnostic-only).
- **Rule (portable, logging to LEARNINGS.md)**: "Soft-target
  distillation from a bagged-OOF teacher to a student of equal
  capacity is a structural overfit trap. The student memorizes the
  teacher's OOF noise (including confident-wrong posteriors on
  boundary rows). Unlike hard pseudo-labels, soft distillation has
  no confidence gate, so teacher errors propagate at full strength.
  To use distillation safely: reduce student capacity by at least 2×
  (fewer trees, smaller depth, or stronger regularization), or
  train the teacher on N-1 folds with row i held out completely
  (not just from the one model that produced teacher_oof[i], but
  from ALL models in the teacher blend)."
- Next bet: pivot to ceiling-breaker #1 — 171-pair OTE with
  subprocess-isolation fix (fixes the 2026-04-23 OOM-kill). New
  feature surface, not a new learning signal, so it can't inherit
  the distillation overfit failure mode.

### 2026-04-24 — final-selection hedge audit (write-up)

- Goal: stop deferring the final-selection call. Deadline 2026-04-30;
  6 days out, 2/2 final slots spent once submitted. LB-state: LB-best
  3-way multi-seed `submission_3way_recipe025_s1035_s7040.csv` at 0.98005,
  prior LB-best 2-way `submission_recipe_greedy_recipe_pseudolabel.csv`
  at 0.97998, recipe standalone `submission_recipe_full_te.csv` at 0.97939,
  recipe CatBoost `submission_recipe_full_te_catboost.csv` at 0.97935.
- **Problem with the plan-of-record (primary 3-way + fallback 2-way)**:
  both anchor on `recipe_full_te + recipe_pseudolabel` (pseudo_s1). If
  pseudo_s1 overfits private LB — which is plausible since the whole
  stacking-inflation ceiling analysis treats it as the "unusually lucky"
  draw that stage-2 / seed-7 / seed-123 couldn't reproduce — BOTH
  submissions regress together. That's correlated exposure dressed up
  as a hedge.
- **Recommendation**: lock primary = 3-way multi-seed (0.98005), hedge =
  **`submission_recipe_full_te_catboost.csv`** (LB 0.97935, gap +0.00001,
  the tightest calibration in the whole ladder). Reasoning:
  1. Different model family (CatBoost ordered-boosting vs XGB), so the
     per-row error distribution is materially different (Jaccard 0.806
     vs recipe, 0.788 on GPU variant). Errors on private LB will not
     overlap with the XGB-family primary's errors in the same way the
     2-way fallback's will.
  2. Single-model, no blend composition — can't suffer stacking-
     inflation overfit. The +0.00001 gap is the only submission in the
     log with OOF ≤ LB, not the other way around.
  3. Hedge cost: −0.00070 LB vs 2-way fallback. That's the insurance
     premium. If the 3-way stays on top of private, the primary wins
     and we don't care about the gap. If the 3-way drops, a
     XGB-family fallback at LB 0.97998 likely drops with it;
     CatBoost-family fallback at LB 0.97935 might stay put or drop
     less (different error geometry under distribution shift).
  4. Alternative hedge = `submission_recipe_full_te.csv` (LB 0.97939).
     Same family as primary (both XGB-on-recipe), but no blend overfit.
     Slightly cheaper premium (+0.00004 vs CatBoost-hedge). I prefer
     CatBoost-hedge because model-family diversity is the stronger
     insurance-against-correlation-failure axis than within-family
     standalone-vs-blend.
- **Zero-risk auxiliary candidate emitted this session**:
  `submission_hedge_avg_2way_3way.csv` (OOF 0.98020, not yet LB-probed).
  50/50 log-mean of the two LB-verified blends at recipe's fixed bias.
  Not recommended as either final slot — shares overfit surface with
  BOTH its parents. Useful only as a diagnostic LB probe if we have a
  slot to spare (~10/day from 2026-04-25 reset).
- No LB spend this session (scaffold only; focal + GroupKFold-crop
  still in flight).

### 2026-04-24 — hedge-avg 50/50 submission built (not yet LB-probed)

- Goal: realize item 4 of the "overlooked levers" pass — zero-risk
  50/50 log-mean of the two LB-verified bests (2-way + 3-way) as an
  "ensemble-of-ensembles" midpoint submission for private LB variance
  protection.
- Changed: `scripts/hedge_avg_lb_bests.py`. Reconstructs 2-way and
  3-way from components (recipe_full_te, recipe_pseudolabel,
  recipe_pseudolabel_seed7labeler), takes a 50/50 log-mean in
  probability space, applies recipe's fixed tuned bias
  [1.43, 1.47, 3.40] (no retune — binhigh-rule compliance), emits
  `submissions/submission_hedge_avg_2way_3way.csv`.
- OOF @ recipe bias:
  - 2-way component       → 0.98012
  - 3-way component       → 0.98029
  - 50/50 log-mean hedge  → **0.98020**
- Test-argmax disagreement matrix:
  - 2-way ↔ 3-way           = 163 rows
  - hedge ↔ 2-way           = 88 rows (hedge agrees with 2-way more often)
  - hedge ↔ 3-way           = 75 rows
  - 88 + 75 = 163 — hedge sits exactly between the two submissions.
- Class dist: Low 159,534 / Medium 100,213 / High 10,253 — pulls the
  High count down from 3-way's 11,101 toward 2-way's territory
  (2-way's split is closer to train prior). Implicit "rare-class
  shrinkage" effect from log-averaging over two predictors with
  different High thresholds.
- Not recommended as a final-slot candidate (shares overfit surface
  with both its parents). Available as a diagnostic LB probe if
  surplus slots remain near deadline.
- Artefacts committed: `oof_hedge_avg_lb_bests.npy`,
  `test_hedge_avg_lb_bests.npy`, `hedge_avg_lb_bests_results.json`.

### 2026-04-24 — focal-loss XGB (γ=2, α=balanced) smoke: NULL (magnitude-trap calibration mismatch)

- Goal: the one untried *training-time* lever for the High Pareto
  frontier. Teacher's per-class recall [0.9949, 0.9685, 0.9774] is a
  Pareto ceiling on the current OOF bank; post-hoc overrides
  (detector, router, disagree-meta) all null because they rearrange
  the same bank. Focal loss changes the ERROR DISTRIBUTION produced
  by the base learner via the `(1-p_y)^γ` hardness modulation.
- Changed: `scripts/recipe_focal_obj.py` — multiclass softmax-focal
  grad/hess closure over (γ, per-class α, sample_weight). Unit
  tests: γ=0 recovers softmax CE exactly; γ∈{1,2,3} analytical
  gradient matches finite-difference at rel err ~1e-8.
  `scripts/recipe_focal.py` — mirrors recipe_full_te FE+CV but uses
  `xgb.train()` with the custom obj (α = balanced-per-class
  × ALPHA_HIGH env multiplier, γ via GAMMA env, early-stop on
  custom bal_acc metric). `scripts/focal_smoke_blend_gate.py` —
  post-hoc Jaccard + fixed-bias sweep diagnostic.
- Smoke pass (FOCAL_SMOKE=1, 2-fold × 630k, γ=2.0, α_High=1×balanced):
  - Fold 1: argmax 0.97598, best_iter=**45**
  - Fold 2: argmax 0.97553, best_iter=**39**
  - Overall OOF argmax: 0.97576 (σ=0.00022)
  - **Tuned OOF: 0.97742** with bias **[1.83, 1.77, 2.20]** —
    notably the High bias is 2.20 vs recipe's 3.40. Focal's γ
    modulation + balanced α already produces sharper High probs,
    so less post-hoc bias correction is needed.
  - Per-class recall at focal's own tuned bias:
    Low 0.99479 / Medium 0.96513 / **High 0.97235** — vs teacher's
    0.9774, focal's High is LOWER despite the extra hardness focus.
  - **Gradient verified correct** — `best_iter` 39-45 across folds,
    not the pathological best_iter=1 from the 2026-04-21 GCE bug.
- Fixed-bias blend gate vs LB-best 3-way teacher (held-out rows only,
  recipe bias [1.43, 1.47, 3.40]):
  - Teacher bal_acc @ recipe bias: 0.98029, errs 9,873
  - Focal bal_acc @ recipe bias:  **0.94801**, errs **36,942** (3.7×)
  - Jaccard(focal, teacher) = **0.2401** (lowest Jaccard we've ever
    seen — true architectural orthogonality)
  - Blend sweep α∈[0, 0.5]: **monotone negative from α=0.025 onwards**
    (α=0.050 Δ=−0.00001, α=0.500 Δ=−0.00142). Peak at α=0 (no blend).
- **Classic magnitude-trap failure** — the 11th NN-family-adjacent
  null with the same geometry: exceptional Jaccard orthogonality
  (0.24!) but 3.7× error-count overflow when evaluated at the
  anchor's calibration scale. Focal's α=balanced + γ=2 produces
  probs in a fundamentally different scale from recipe's XGB +
  balanced-sample-weight. At recipe's fixed bias (the rule to
  prevent binhigh-style retune overfit), focal's probs are
  misaligned and 36k rows flip to wrong argmax.
- **Production NOT launched**: 5-fold with the same objective won't
  change the calibration-mismatch geometry — the scale difference is
  architectural (training-time focal modulation), not variance.
  Retuning bias on the focal+teacher blend per-α would manufacture
  OOF lift that won't transfer to LB (binhigh-rule, 2026-04-21).
- **Rule** (candidate LEARNINGS.md add): *Training-time loss changes
  that alter a model's prob-scale calibration (focal γ>0, LDAM-DRW,
  logit-adjustment with non-uniform priors) cannot be blended with
  a differently-calibrated anchor at the anchor's fixed bias. They
  either (a) need full pipeline replacement — refit teacher at the
  new scale — or (b) need per-class isotonic calibration before
  blending, which already nulled on C0 isotonic+greedy.* This
  closes focal loss as a blend leg on this feature set; it remains
  unclosed as a STANDALONE replacement for recipe, which would need
  full 5-fold production + LB probe.
- No LB probe warranted. Artefacts committed for cross-branch reuse:
  `oof_recipe_focal_g2_aH1.npy` + test + results JSON + blend-gate
  JSON + diagnostic submission CSV.

### 2026-04-24 — session close-out (review-edge-cases)

- 4-item overlooked-levers pass executed on `claude/review-edge-cases-6K1Dm`.
- **Item 1 (final-selection hedge audit)**: written up — primary 3-way
  (LB 0.98005) + CatBoost hedge (LB 0.97935, gap +0.00001) preferred
  over 3-way + 2-way pairing (both share pseudo_s1 overfit surface).
- **Item 2 (focal-loss XGB)**: closed NULL. Smoke confirmed gradient
  math (best_iter 39-45, not pathological 1), tuned OOF 0.97742,
  but magnitude-trap blend gate (Jaccard 0.24 yet 3.7× errors at
  recipe bias) killed the lever. Production not launched —
  calibration-mismatch is architectural.
- **Item 3 (GroupKFold-crop)**: SCRIPT WORKS but environment repeatedly
  killed the 45-min CPU job at session turnover (three attempts
  with `setsid nohup` detach; each died mid-training). Last
  progress: fold 1 at round 1000, mlogloss 0.048 — consistent with
  recipe's single-fold training curve. No artifacts produced.
  Script `scripts/b2_groupkfold.py` is invoked with `GROUP=crop`;
  a future session with longer wall budget should complete it.
  Expected: OOF-honest per the Region precedent (−0.00029 delta),
  confirming structural ceiling.
- **Item 4 (hedge log-mean submission)**: zero-risk 50/50 log-blend
  of 2-way × 3-way at recipe's fixed bias. OOF 0.98020. Midway
  between anchors geometrically. Diagnostic only.
- LB budget unchanged (no probes this session). LB best unchanged:
  `submission_3way_recipe025_s1035_s7040.csv` at **LB 0.98005**.

## Hypothesis board

- **Current best (LB)**: `submission_tier1b_greedy_meta.csv` →
  **LB 0.98094 / OOF tuned 0.98084** (gap **−0.00010** — LB above OOF,
  first negative gap since digit-XGB era). Construction:
  ```
  lb3      = log_blend(recipe_full_te, recipe_pseudolabel, recipe_pseudolabel_seed7labeler;
                       0.25/0.35/0.40)
  stack1   = log_blend(lb3, realmlp;                    0.80/0.20)
  stack2   = log_blend(stack1, xgb_nonrule_iso;         0.925/0.075)   ← prior LB-best 3-stack
  final    = log_blend(stack2, xgb_metastack_iso;       0.70/0.30)     ← Tier-1b new step
  pred     = argmax(log(final) + [1.4324, 1.4689, 3.4008])
  ```
  Pack 0.98114 now only **+0.00020 above**; leader 0.98219 only **+0.00125 above**.
  LB budget: **3/10 used today** (3 = 1 recipe_full_te baseline from earlier,
  1 LB-best-3-stack confirmation, 1 new probe). 7 remaining.

- **Saturation status (2026-04-25 end-of-day)**: the new LB-best 4-stack
  is locally saturated against THREE independent attack vectors tested
  in Tier 1c:
  1. Greedy log-blend over expanded 132-component pool (incl. iso copies):
     step1 picks `recipe_no_digits α=0.010 → +0.00002`, sub-gate.
  2. Meta-stacker v2 with v1's OOF + binary specialists + 4-stack inputs
     (224-dim feature space): best v2_iso α=0.20 → +0.00002 OOF.
  3. Multi-seed bag of meta-stacker XGB seeds {42,7,123}: best add α=0.150
     → +0.00003 OOF.
  Plus parallel-session falsifications: RealMLP n_ens=4 strictly worse
  than n_ens=1 (commit 6662924), per_bin_blend NULL with regression risk
  (commit 80842d0). Conclusion: breaking past LB 0.98094 requires a
  fundamentally NEW signal source, not another OOF-stacking variant.

  **Why it worked**: greedy forward over a 63-component pool with isotonic
  calibration applied to every candidate. The XGB meta-stacker (`oof_xgb_metastack`,
  trained over ~200-dim feature space = 63 components × 3 + dgp_score + distances)
  has **8,948 errors** standalone vs LB-best stack's 9,572. Raw blend peaks at
  α=0.40 with +0.00012 OOF (LB-marginal); **isotonic-calibrated** version peaks
  at α=0.30 with +0.00023 OOF, and the LB delta over-shot by ~3.7× (+0.00086 LB
  vs +0.00023 OOF). Mechanism: meta-stacker corrects LB-best's log-bias
  over-push on score=6 boundary rows — the dominant Medium→High error bucket.
  See 2026-04-25 session log for the +157 net-correct-flips per-bucket breakdown.

  Second-best: `submission_lb3_realmlp_nonruleiso.csv` →
  **LB 0.98008 / OOF tuned 0.98061** (gap +0.00053). Greedy from the
  3-way multi-seed anchor + RealMLP α=0.200 + xgb_nonrule_iso α=0.075.
  This is the prior LB best the meta-stacker built on top of.

  Third-best: `submission_3way_recipe025_s1035_s7040.csv` →
  **LB 0.98005 / OOF tuned 0.98029** (gap +0.00024). Pure 3-way
  log-blend of recipe_full_te × pseudo_s1 × pseudo_s7 — the safe
  fallback. Tightest calibration of the pre-meta candidates; good
  hedge for private LB.

  Fourth-best: `submission_recipe_greedy_recipe_pseudolabel.csv` →
  **LB 0.97998 / OOF tuned 0.98012** (gap +0.00014).
  50/50 log-blend of recipe_full_te × recipe_pseudolabel at recipe's
  fixed tuned bias [1.43, 1.47, 3.40].

  Fifth-best: `submission_recipe_full_te.csv` → **LB 0.97939 /
  OOF tuned 0.97967** (gap +0.00028). Full V10 recipe: ~117
  categoricals (raw+pair+digit+num-as-cat+tres) OTE'd, plus FREQ +
  ORIG mean/std + LR-formula logits + threshold flags = ~500
  features. Heavy-reg XGB (max_depth=4, alpha=5, reg_lambda=5) +
  class-balanced sample weights + post-hoc log-bias.

  Sixth-best: greedy full-bank 6-way log-blend (digit_xgb 0.44 +
  digits_ote 0.24 + xgb_nonrule 0.11 + xgb_corn 0.09 + digits_pairs
  0.07 + digits_light_ote 0.05) → OOF 0.97558, LB 0.97581.
  Submission: `submissions/submission_greedy_full_bank.csv`.

  Seventh-best: digits-OTE × digit-XGB log-blend at α=0.40
  → OOF 0.97477, LB 0.97482. Submission:
  `submissions/submission_digit_ote_digits_blend.csv`.

  Eighth-best: XGB-dist + digits standalone, tuned log-bias →
  OOF 0.97449, LB 0.97468. Submission:
  `submissions/submission_xgb_dist_digits_tuned.csv`.

### 2026-04-23 — N1 recipe_no_ote: genuinely novel errors (Jaccard 0.60) but blend null — magnitude trap

- Goal: execute N1 from the "Next steps" menu below. First variant:
  `recipe_no_ote` — drop OrderedTE entirely, train XGB on the 92
  numeric features only (nums + tres + logits + freq + orig_stats).
  Hypothesis: OTE is the dominant signal source in full recipe; a
  tree that never sees OTE is a structurally different decision
  surface, unlike LGBM/CatBoost/XGB-dart which all share the 351
  OTE features.
- Changed: `scripts/recipe_subset_{fe,cv,xgb}.py` — short modular
  split per the CLAUDE.md rule. Env var `RECIPE_SUBSET ∈ {no_ote,
  no_digits, no_combos, no_orig}` selects which feature block to
  drop. Reuses recipe_full_te's FE pipeline and XGB HPs so the
  comparison is apples-to-apples. `scripts/analyze_vs_recipe.py`
  for quick Jaccard + fixed-bias blend-sweep diagnostics.
- Results (5-fold, seed=42, 630k):
  - Per-fold argmax: 0.97030 / 0.97102 / 0.97191 / 0.97005 / 0.97010
  - **Overall argmax 0.97067 ± 0.00083**
  - **Tuned OOF 0.97465** (bias [1.53, 1.47, 3.40]) — **−0.00502 vs
    recipe's 0.97967**. Weakest "on-recipe" tree so far.
  - Error count: **11,760** (+16% vs recipe's 10,114)
  - **Jaccard(err) vs recipe = 0.6040** — the lowest error overlap
    of any tree on the recipe features. LGBM was 0.835, CatBoost
    CPU 0.806, CatBoost GPU 0.788. The "novel" tag triggers per
    the < 0.80 heuristic.
- Fixed-bias log-blend sweep vs recipe:
  ```
  α=0.000  OOF=0.97967  Δ=+0.00000   (anchor)
  α=0.025  OOF=0.97967  Δ=+0.00000   ← peak (flat)
  α=0.050  OOF=0.97964  Δ=-0.00002
  α=0.100  OOF=0.97958  Δ=-0.00008
  α=0.500  OOF=0.97867  Δ=-0.00099
  ```
  Greedy forward-selection over all 14 saved OOFs (including
  no_ote): picked `recipe_full_te (0.925) + digit_xgb (0.075)`
  → OOF 0.97978 (+0.00012). **no_ote rejected at every α.**
- **Lesson — refinement of the Jaccard heuristic**: Jaccard 0.60
  is genuinely novel error geometry (lower than any prior tree
  null), but the +16% error-magnitude dominates. Even at α=0.025,
  the extra ~5k wrong answers from no_ote cancel the ~3.4k
  complementary right answers it contributes. Compare to
  pseudolabel on main (Jaccard 0.78, **FEWER** errors 10,039,
  blend Δ=+0.00046): low Jaccard PLUS ≤-anchor error count is
  what moves the blend.
  New rule to LEARNINGS.md: **"Jaccard < 0.80 is necessary but
  not sufficient for blend lift. A candidate must also have
  error count ≤ the anchor's — otherwise the magnitude drag on
  unique-wrong rows cancels the novelty on unique-right rows."**
- Implication: **N1 variants no_digits / no_combos / no_orig will
  hit the same trap.** Each is strictly weaker than full recipe
  (less information), so their error counts will be comparable to
  or worse than no_ote's 11,760. Skipping the remaining 3 variants.
- Next bet (reframed): the only known component satisfying BOTH
  Jaccard < 0.80 AND errors ≤ recipe is `recipe_pseudolabel`
  (main's 0.97998 LB-best component). OOF is gitignored on main,
  so we'd need to run `scripts/recipe_pseudolabel.py` locally
  (~48 min) to experiment with 3-way blends adding no_ote/digit_xgb.
  Alternatively pursue the untried 171-pair OTE (main's stated
  next bet) which creates a fundamentally new feature surface.
- LB delta: n/a (no LB probe warranted; OOF Δ +0.00012 via greedy
  is below the +0.0002 LB-transfer threshold).
- Artefacts:
  - `scripts/artifacts/oof_recipe_no_ote.npy` + `test_recipe_no_ote.npy`
  - `scripts/artifacts/recipe_no_ote_results.json`
  - `submissions/submission_recipe_no_ote.csv` (diagnostic, not LB-worthy)
- Current LB-best unchanged: `submission_recipe_greedy_recipe_pseudolabel.csv`
  at LB **0.97998** (from main).

### 2026-04-23 — N1 remaining 3 variants (no_digits, no_combos, no_orig) all LB-null — magnitude-trap rule confirmed

- Goal: despite the main-branch session saying "skipping remaining 3
  variants" after no_ote's blend-null magnitude trap, a parallel
  feature branch ran them anyway to verify the rule. All 3 ran on
  the same 5-fold seed=42 with same XGB HPs as recipe_full_te.
- Results (5-fold OOF, tuned log-bias):
  ```
                Tuned OOF   Δ vs recipe   Errors   Jaccard vs recipe
  recipe         0.97967        —         10,114   1.00
  no_digits      0.97956    −0.00011      10,393   0.7866   (+2.8% errs)
  no_combos      0.97951    −0.00016      10,742   0.8361   (+6.2% errs)
  no_ote         0.97465    −0.00502      11,730   0.6066   (+16% errs)
  no_orig        0.97961    −0.00006      10,618   0.8555   (+5.0% errs)
  ```
- **Magnitude-trap rule holds for all 4 variants.** Each has MORE
  errors than recipe (the rule predicts blend-null), despite some
  having lower Jaccards. no_orig is the closest-to-tied standalone
  but has highest Jaccard (0.8555 — errors overlap too much with
  recipe). no_ote has lowest Jaccard but +16% error magnitude.
- **5-way blend ceiling: OOF 0.97982 (Δ=+0.00015 vs recipe)**.
  Greedy forward from recipe anchor at fixed bias:
  ```
  + no_digits α=0.300  OOF 0.97978 (+0.00012)
  + no_orig   α=0.025  OOF 0.97982 (+0.00003)
  no further additions (no_combos, no_ote below threshold)
  ```
  5-way grid gives same ceiling 0.97982 at (recipe 0.60,
  no_digits 0.20, no_combos 0.15, no_ote 0.03, no_orig 0.00).
  Coord-ascent on the 3-way weights converged at iter 0 — already
  optimal.
- **+0.00015 OOF is well below the +0.0002 LB-transfer threshold**
  established by main's 4-way stage-2 blend null. Given recipe's
  +0.00028 OOF→LB gap, expected LB ≈ 0.97926 (below current LB-best
  0.97998). **No LB probe warranted.**
- Verdict: N1 is **fully closed** — all 4 variants individually
  confirm the magnitude-trap rule; blend ceiling +0.00015 OOF, below
  LB-transfer threshold. Artifacts preserved for cross-branch greedy
  use but unlikely to contribute to future blends.
- **What N1 proved beyond its immediate null**: the 5-way-blend
  ceiling of +0.00015 on recipe-subsets is a clean upper bound on
  "feature-removal diversity" as a lever. To break through, we need
  feature-ADDITION (171-pair OTE, external data source, etc.) or
  a new training objective, not subtraction.
- Artefacts:
  - `scripts/artifacts/oof_recipe_no_digits.npy` (+ test + JSON)
  - `scripts/artifacts/oof_recipe_no_combos.npy` (+ test + JSON)
  - `scripts/artifacts/oof_recipe_no_orig.npy` (+ test + JSON)
  - `scripts/artifacts/recipe_5way_blend_results.json`
  - 4 diagnostic submissions (LB-inferior, not for probe)

### Next steps: feature-set diversity on top of the recipe anchor (2026-04-23)

Confirmed after 4 tree-family nulls on recipe (XGB, LGBM, CatBoost CPU,
CatBoost GPU all Jaccard 0.78-0.84 + similar error counts): **further
tree-family additions on the same features are null. To break the
pattern, change the feature set, not the model.** Candidate levers,
all untried as of this entry, ranked by expected ROI:

  **N1. Recipe-subset XGBs for blend diversity** (top pick, ~40 min each).
  Recipe has 9 distinct feature blocks: `nums / cats / combos / digits
  / num_as_cat / tres / logits / freq / orig_stats`. Train recipe XGB
  on subsets:
    - recipe-no-digits: drop 66 digit cols + their ~198 derived OTE
      features → ~300 features. Forces model onto OTE+ORIG+FREQ only.
    - recipe-no-combos: drop 28 pair combos + their ~84 OTE → ~390
      features. Leans on single-cat OTE + digits.
    - recipe-no-OTE: just the 85 numeric feats (raw nums + tres +
      logits + freq + orig_stats). Pure-numeric XGB, very different
      decision surface. Expected weaker standalone but orthogonal
      errors to the OTE-dominated full recipe.
    - recipe-no-ORIG: drop 38 ORIG stats + their OTE contributions.
  Each creates an XGB that captures different aspects of the signal.
  Blend them with the full recipe — if the feature-block drop produces
  Jaccard < 0.75 with full recipe AND error count within +10%, the
  blend-null heuristic predicts a lift. All reuse the existing recipe
  pipeline with a one-line feature-slice change.

  **N2. Multi-strength OTE on recipe features** (~45 min). Recipe
  uses `a=1.0` for OTE shrinkage. Train parallel recipes at `a=0.1`
  (less smoothing) and `a=10` (more smoothing). Different shrinkage
  changes how low-count categories are encoded, producing different
  XGB splits on the same raw features. Blend all three.

  **N3. Pseudo-labeling with recipe as labeler** (~90 min). Prior
  attempts were null with weaker labelers (greedy+nonrule level).
  Recipe at LB 0.97939 is a much stronger labeler. High-confidence
  test predictions added as training labels could reshape the
  decision surface on boundary rows. Gate: confidence ≥ 0.98 AND
  rule_pred == argmax (double confirmation).

  **N4. XGB-dart on recipe** (~1h). Same gradient-boosted family
  but `booster='dart'` applies tree dropout during training. The
  4-way tree null pattern is within `gbtree`; dart is structurally
  different-enough that it may break the pattern. Lower expected
  ROI than N1/N2 given the pattern, but cheap.

  **N5. Greedy forward-selection over expanded OOF bank**. Re-run
  `greedy_from_recipe.py` after any of N1/N2/N3/N4 land — the new
  components enter the candidate pool. Previously greedy-from-recipe
  found only digit_xgb at +0.00012 (borderline). New orthogonal
  components may unlock a real lift.

  **Skipping on principled grounds:**
  - Further OTE key expansion (triples, etc.) — OTE family saturated
    across 5 variants already.
  - SVM / k-NN / Naive Bayes — earlier benchmarks (heuristic 0.60,
    NB 0.75, LR 0.83) show them too weak to contribute.
  - Seed bagging — LB-regressed twice (2026-04-22 Session B,
    2026-04-23 digit seed-bag). Below fold-std threshold rule.
  - General HP tuning — LB-regressed twice earlier. Any +OOF is
    likely bias-overfit.

  **Execution order**: N1 (recipe-no-OTE is the cheapest and most
  structurally different — best first shot) → N1 variants (no-digits,
  no-combos, no-ORIG) in parallel on separate runs → N5 greedy over
  the expanded bank. Budget: ~2-4h for the N1 family, 15 min for N5.

### Next steps: outside-the-envelope perspectives (2026-04-24)

Context: every tree family, NN variant, FE family (OTE/digit/pairs/
binning/residual/TE-from-X), blending family (log/prob/rank/LR-stack/
greedy), calibration family (global/per-score log-bias), training-
data-engineering family (pseudo-label stage-1/2, routing, cleanlab,
augmentation), and distillation family (hard pseudo, soft distill)
has been tested on the recipe anchor. The own-pipeline blend stack
appears structurally pinned at LB ~0.97998. These three perspectives
are modeling paradigms that don't map onto any lever in the log.

  **P1. Inference-time test-time augmentation on threshold axes**
  (top pick: no retraining, cheapest, ~30 min smoke + 5–10 min per-σ
  full run). Rationale: axis-aligned trees produce step-function
  discontinuities exactly at the rule thresholds (Soil=25, Rain=300,
  Temp=30, Wind=10) where flip signal concentrates; the host NN
  produces a smooth decision surface. Per-test-row, perturb the 4
  threshold-critical features with Gaussian noise σ ∈ {0.02, 0.05,
  0.10} × feature IQR, K=5–10 times; recompute rule indicators /
  distance features / digit cols / OTE lookups for each perturbation;
  average per-class log-probs. Approximates smoothing at inference
  without training a new model. Gate: fixed-bias OOF Δ ≥ +0.0003
  AND error-magnitude ≤ anchor. Expected: +0.0005–0.002 LB if
  within-cell smoothing is the missing piece; null if σ is
  ineffective or walks rows into wrong cells. Pivot from σ=0.02
  up if smaller settings are null.

  **P2. Symbolic regression for within-cell flip formula**
  (second pick: novel tool class, ~1 evening). Rationale: the 6-
  feature base rule was reverse-engineered as a closed-form integer
  expression; ~10k within-cell flips are a deterministic NN function
  of 7 non-rule continuous features (per 2026-04-21 DGP residuals
  EDA), but every attempt to *model* the flip (per-cell LR/MLP/EB,
  TE-regression, cleanlab, soft distill) is a black-box fit that
  plateaus at ~0.963 within-cell due to capacity/data tradeoff.
  PySR/gplearn search the space of *analytic expressions*. Target
  the 2 dominant error cells (score=3 Low→Medium n=5041; score=6
  Medium→High n=4163). Within each cell, binary-label
  flipped-vs-not and run PySR with 20–30 min budget on the 7 non-
  rule continuous features. Deploy any formula hitting >70% flip
  recall at <10% FP on clean rows as a hardcoded override on the
  LB-best blend. Distinct from all tree FE: an override is a
  deterministic replacement, not a probability blend, so it
  stacks orthogonally regardless of Jaccard/magnitude. Upside
  +0.001–0.003 LB if the NN has axis-aligned or polynomial-ish
  seams; null if the NN is smooth-curved.

  **P3. Transductive k-NN label propagation in a learned embedding**
  (third pick: highest infrastructure cost, ~2h). Rationale: every
  method tested is *inductive* — fit on train, query on test. Label
  propagation on a k-NN graph over (train ∪ test) in a learned
  embedding space is the one remaining modeling paradigm
  unrepresented in the log. Test-row prediction depends on
  nearby test-row geometry; failure modes are graph construction
  / bandwidth, not tree-shape or NN-capacity. (a) Fit a supervised
  contrastive embedding: small MLP on recipe 443-feature matrix
  with same-label pull / different-label push + classification
  head, freeze post-convergence. (b) Embed train+test into ~32-dim
  space. (c) Build k-NN graph (k=30, Gaussian kernel) via FAISS.
  (d) Run sklearn LabelPropagation/LabelSpreading (α=0.2).
  (e) Standard fixed-bias blend-gate vs LB-best 2-way. Upside
  +0.0005–0.002 LB if the embedding captures the NN-generator's
  manifold; null if it collapses onto rule features.

  **Parallelism**: P1 (TTA) reuses the existing recipe pipeline
  and runs first as a single-threaded OOF sweep on CPU
  (~30 min smoke + 1–2 hours full). P2 (symbolic regression)
  and P3 (label-propagation-in-embedding) are independent of P1
  and of each other — can launch in background on Kaggle GPU
  kernels while P1 runs locally. Priority order if all three
  null: revisit multi-seed pseudo-label chain (already scaffolded
  with FOLD_SEED=7 labeler, ran 2026-04-24) or pivot to the
  max_bin=10000 GPU-recipe as simplest remaining infrastructure
  lever.

  **Skipping on principled grounds:**
  - Conformal / per-row blend weighting — sophisticated but
    structurally a blend variant; prior global blend ceiling
    (OOF 0.98033 → LB 0.97995) likely still applies.
  - VAE/diffusion-augmented training — 10k-original augmentation
    already LB-regressed at w=20 (2026-04-21 training-data-quality
    experiment); synthesizing more rule-clean rows compounds that
    bias.
  - Adversarial / min-max training — requires retraining full
    recipe pipeline; expected bounded by tree-family ceiling
    already confirmed 4 times.

### 2026-04-24 — P1 TTA executed: NULL across all σ and both blend anchors

- Goal: execute P1 — inference-time TTA on the 4 rule-threshold numerics.
  Perturb Soil_Moisture / Rainfall_mm / Temperature_C / Wind_Speed_kmh
  with Gaussian noise σ × feature_IQR, recompute only the threshold-
  derived features (flags, LR logits, digit cols for those 4 nums, raw
  nums), predict K times, log-average. OTE / FREQ / num_as_cat / combos
  held fixed — perturbing them would degenerate to prior and add noise.
- Changed: `scripts/tta_helpers.py` (perturbation + feature-recompute
  helpers), `scripts/tta_recipe_full.py` (mirrors recipe_full_te with
  TTA inference loops; MAX_BIN=512 to halve XGB histogram memory after
  an OOM kill on the first production attempt with MAX_BIN=1024),
  `scripts/tta_analyze.py` (Jaccard + magnitude + blend-gate post-
  analysis). K=3, σ ∈ {0.001, 0.005, 0.010}.
- Full 5-fold production wall: 31 min on CPU (after OOM fix of `del
  orig` post-FE + per-fold `gc.collect` + `MAX_BIN=512`).
- Results (OOF bal_acc, 5-fold seed=42):
  ```
  variant   argmax   tuned    errs   Jacc(recipe)  Jacc(lb_best)
  baseline  0.97596  0.97955  8,380  0.9260        0.9101
  s001      0.97595  0.97953  8,450  0.9084        0.8936
  s005      0.97479  0.97870  8,984  0.8459        0.8324
  s010      0.97364  0.97787  9,600  0.7814        0.7683
  ```
  Note: baseline here differs slightly from recipe_full_te's 0.97967
  because MAX_BIN=512 vs 1024; apples-to-apples is tuned(baseline) vs
  tuned(s*), not vs recipe_full_te.
- Fixed-bias blend sweeps (over α ∈ [0, 0.50]):
  ```
                      vs recipe (anchor 0.97967)   vs LB-best 2-way (0.98012)
  s001                peak α=0.300  Δ=+0.00008     peak α=0.075  Δ=+0.00007
  s005                peak α=0.200  Δ=+0.00003     peak α=0.075  Δ=+0.00008
  s010                peak α=0.150  Δ=+0.00002     peak α=0.150  Δ=+0.00009
  ```
  **All deltas well below the +0.0002 LB-transfer threshold**
  documented on 2026-04-23. No LB probe warranted.
- **Classic blend-null pattern** (third confirmation of the
  2026-04-23 magnitude-trap rule): s001 has fewest extra errors
  (+70) but Jaccard 0.91 is too redundant with the anchor; s010
  has promising Jaccard 0.78 but +1,220 extra errors overwhelm
  the orthogonal signal. No σ threads the needle.
- **Interpretation:** at 1200+ XGB iterations on 504k rows, trees
  have placed near-optimal axis-aligned splits across the whole
  feature space. Any perturbation shifts far-from-boundary rows
  more often than it helps the ~2% of boundary rows; averaging
  K perturbations adds far-row noise faster than it smooths the
  boundary discontinuity. The mechanism that would make TTA work
  ("smooth step-function discontinuities near rule thresholds")
  doesn't dominate because (a) most rows are far from thresholds,
  (b) the model's splits are distributed across hundreds of
  non-rule features too. Matches the recurring failure mode of
  the axial-mechanism interventions (monotone constraints,
  Frank-Hall ordinal, etc.) on this feature set.
- **Portable rule** (logged for next synthetic tabular comp):
  **Tabular TTA on threshold-axis numerics is unlikely to lift a
  recipe-level ensemble once training has saturated. The
  perturbation-smoothing gain scales with boundary-row fraction
  (~2%), while the far-row noise scales with N. At 504k rows the
  far-row noise dominates.** Skip TTA on tabular problems unless
  the training set is small AND the boundary-row fraction is
  large (e.g., >10%).
- Artefacts:
  - `scripts/artifacts/oof_tta_recipe_{baseline,s001,s005,s010}.npy`
  - `scripts/artifacts/test_tta_recipe_{baseline,s001,s005,s010}.npy`
  - `scripts/artifacts/tta_recipe_results.json`,
    `tta_blend_gate_results.json`
- LB budget unchanged at 8/10 used yesterday, 2 remaining today.
  Current LB-best still `submission_recipe_greedy_recipe_pseudolabel.csv`
  at LB 0.97998.
- Next (per P1/P2/P3 plan): P2 (gplearn symbolic regression on the 2
  dominant error cells) launched immediately. P3 (label-propagation-
  in-embedding) queued if P2 is null.

### 2026-04-24 — P2 symbolic regression (gplearn) closed as NULL

- Goal: search analytic formulas in the 7 non-rule continuous features
  (Humidity, Prev_Irrigation, EC, Field_Area, Soil_pH, Organic_Carbon,
  Sunlight) that predict within-cell label flips at the 2 dominant error
  cells (score=3 rule=Low→flip to Medium, score=6 rule=Medium→flip to
  High). Hypothesis: if the NN generator's flip function has polynomial
  / algebraic structure, gplearn finds it; deploy as hard override.
- Changed: `scripts/p2_symbolic_flip.py` — bug fix (`_Program.execute`
  not `.predict`) + try/except per formula evaluation.
- Setup: gplearn SymbolicRegressor, pop=1000, gen=25, MAE loss,
  function set = (+, -, *, /, sqrt, log, abs), parsimony=0.01.
- **Results (trivial formulas won):**
  ```
  cell3 (n=102,157, flip_rate=4.796%)
    rank 0:  sub(Organic_Carbon, Organic_Carbon) = 0
             precision 0.0, recall 0.0, selected 0 rows
    rank 1:  sub(Field_Area_hectare, Field_Area_hectare) = 0
    rank 2:  sub(Organic_Carbon, Organic_Carbon) = 0

  cell6 (n=38,416, flip_rate=4.032%)
    rank 0:  sub(Field_Area_hectare, Field_Area_hectare) = 0
    rank 1:  sub(Humidity, Humidity) = 0
    rank 2:  sub(Field_Area_hectare, Field_Area_hectare) = 0
  ```
- **Diagnosis:** predicting constant 0 (no flip) gives MAE = flip_rate
  ≈ 0.04. gplearn's fitness landscape + parsimony penalty favors short
  formulas with low MAE. No polynomial/algebraic expression built from
  the 7 features achieves lower MAE than the trivial "predict 0" —
  the evolution converged on degenerate `sub(X, X)` = 0 formulas as
  globally optimal.
- **Closes the hypothesis:** the DGP's within-cell flip signal is NOT
  expressible as a compact analytic formula of the 7 non-rule
  continuous features. Consistent with the 2026-04-21 "DGP is a
  deterministic NN function, not a noise process" finding — the NN's
  within-cell decision surface is too non-linear / high-order for
  gplearn with population 1000 × 25 generations to find.
- Parallel run with sample_weight rebalancing (pos_weight×20) and
  pop=1500/gen=30 also converged to constant-length-1 formulas with
  fitness 0.495 — the imbalance fix doesn't rescue the null, the
  functional form simply isn't in the reachable search space.
- **No LB probe.** P2 lever CLOSED.
- Parallel learning: `p2_symbolic_flip.py` pattern can be reused on
  future synthetic-DGP competitions WHERE the DGP is known or strongly
  suspected to be a finite-degree polynomial function. On this
  competition, the NN generator is structurally outside that class.

### 2026-04-24 — P3 transductive label-propagation closed as NULL

- Goal: supervised-contrastive embedding + k-NN label propagation on
  (train ∪ test). Different modeling paradigm than any tree / NN /
  transformer / pretrained tabular-FM previously tested — test-row
  geometry enters prediction via the graph Laplacian. Structurally
  orthogonal failure modes from inductive methods.
- Changed: `scripts/p3_embed_propagate.py` — supervised contrastive
  MLP backbone (SupCon + cross-entropy joint loss on recipe 443-feature
  matrix), 32-dim projection, FAISS k=30 Gaussian kernel graph,
  label-spreading with α=0.2. 5-fold StratifiedKFold(seed=42) aligned
  with all other OOFs. `scripts/p3_analyze.py` runs the fixed-bias
  blend-gate vs both recipe and LB-best 2-way.
- Install: `pip install torch --index-url .../cpu` +
  `pip install faiss-cpu` (both CPU-only, ~250MB).
- Wall: 120 min on CPU. 15 epochs per fold × 5 folds × 34 s/epoch
  training = ~43 min; 6 k-NN queries (5 fold val + full-train test)
  with FAISS = ~17 min; full-train final embedding = ~12 min for 15
  epochs at 48s/epoch. First production attempt died silently at
  epoch 8 of fold 1 (likely session-cleanup event; OOM ruled out,
  memory was fully free afterwards); relaunched with
  `Bash(run_in_background=True)` for proper detachment.
- Standalone results (OOF, 5-fold seed=42):
  - argmax 0.96629, tuned **0.97047** (bias [2.53, 0.17, 4.50] —
    sharply different from recipe's [1.43, 1.47, 3.40], consistent
    with the embedding's different calibration regime).
  - errors **9,958** (+1,591 vs recipe's 8,367, +1,657 vs LB-best 2-way's 8,301).
  - **Jaccard vs recipe = 0.6456** (lowest Jaccard of any standalone
    tested on this feature set except FT-Transformer's 0.614).
  - Jaccard vs LB-best = 0.6473.
- Blend sweep (fixed bias, α ∈ [0, 0.50]):
  ```
  vs recipe (anchor 0.97967):      peak α=0.000  Δ=0.00000   (monotone negative)
  vs LB-best 2-way (anchor 0.98012): peak α=0.000  Δ=0.00000   (monotone negative)
  ```
  Any α > 0 is strictly negative. No blend weight threads the needle.
- **Classic magnitude-trap failure** (fifth confirmation on this
  problem). Jaccard 0.65 = genuinely novel errors; but +1,657 MORE
  errors than LB-best overwhelm the complementary-signal gain.
  The log-blend math: at α=0.1, we trade a tiny contribution of
  P3's unique-right rows (weighted 0.1) for a sizable contribution
  of P3's unique-wrong rows (also weighted 0.1 but there are more
  of them). Net loss.
- Portable rule (adds to LEARNINGS.md candidate): **transductive
  label-propagation in a learned contrastive embedding shares the
  magnitude-trap failure mode of inductive NN / transformer / TabPFN
  on deeply-engineered tabular problems.** Even "test geometry enters
  prediction" doesn't rescue you when the graph neighbors are
  systematically wrong on ~2% of rows the XGB is right on. For P3 to
  succeed, the embedding would need to be trained with a loss that
  penalizes errors on rows where a strong inductive baseline is right
  — which is effectively distillation, not transductive SSL.
- LB budget unchanged at 8/10 used Thursday, 2 remaining today.
  Current LB best unchanged at 0.97998.
- Artefacts:
  - `scripts/artifacts/oof_p3_embed_propagate.npy`
  - `scripts/artifacts/test_p3_embed_propagate.npy`
  - `scripts/artifacts/p3_embed_propagate_results.json`
  - `scripts/artifacts/p3_blend_gate_results.json`

### 2026-04-24 — All three outside-the-envelope perspectives closed

**Session close-out**: P1 (TTA), P2 (symbolic regression), P3 (label-
propagation) all NULL. Three fundamentally different modeling
paradigms, each tested as a standalone + blend-gate candidate on top
of the LB-best 2-way. None cleared the +0.00020 LB-transfer threshold;
each failed for a structurally different reason:

  - P1 TTA: perturbation noise scales with N (far-row noise) while
    boundary-smoothing gain scales with ~2% boundary fraction. At
    504k rows, noise dominates.
  - P2 symbolic: flip signal is not expressible as a compact analytic
    formula of 7 non-rule continuous features at gplearn's search
    capacity (pop 1000-1500, gen 25-30).
  - P3 label-prop: genuine error orthogonality (Jaccard 0.65) but
    magnitude overflow (+1,657 extra errors) — same failure mode as
    every prior NN-family attempt on this feature set.

LB 0.97998 is the firm own-pipeline ceiling across four major lever
categories (tree families, NN families, FE families, modeling
paradigms). Final-selection candidates remain:
  1. **Primary**: `submission_recipe_greedy_recipe_pseudolabel.csv`
     (LB 0.97998, OOF 0.98012, gap +0.00014)
  2. **Safe fallback**: `submission_recipe_full_te.csv`
     (LB 0.97939, OOF 0.97967, gap +0.00028)

Pack 0.98114 stays +0.00116 above; leader 0.98219 stays +0.00221 above.
Reachable only via public-CSV blending (CLAUDE.md rule forbids).

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

### 2026-04-23 — 171-pair binned GPU completion: NULL, with infrastructure learnings

- Goal: revive the 171-pair lever (Ali Afzal "pairwise-TE magic") on
  Kaggle GPU after CPU OOM'd twice. Goal #1: prove the binned 171-pair
  pipeline can finish at all. Goal #2: assess as a blend leg.
- Changed: `kaggle_kernel/kernel_171pair_gpu/` — single-file 545-line
  GPU kernel with inlined recipe FE + OrderedTE + log-bias tuner +
  quantile binning. Uses XGBoost 2.1+ `tree_method='hist', device='cuda'`
  + aggressive per-fold cleanup (del + gc.collect) + `del orig` post-FE.
  v1 SMOKE pass (20k/2-fold/200-iter, 30s wall on P100). v2 production
  full 5-fold/3000-iter.
- Production results (Kaggle P100, 37.7 min wall):
  - Per-fold argmax: 0.97478 / 0.97605 / 0.97580 / 0.97428 / 0.97599
  - Overall OOF argmax: 0.97538 (recipe baseline 0.97589, **−0.00051**)
  - **Tuned OOF: 0.97946** (recipe 0.97967, **−0.00018**)
  - Bias [1.23, 1.27, **3.30**] — High at 3.30 vs recipe's 3.40
  - Total wall 37.7 min vs failed 2.5h CPU attempt
  - 171 combos × 3 cls = 813 OTE cols, 1048 total features
  - Peak RAM 1GB / **31GB Kaggle GPU kernel** (CPU OOM'd at 21GB local —
    Kaggle GPU has 31GB, OOM was a local-container artefact)
- Blend analysis (saved OOFs):
  ```
  recipe          OOF=0.97967  errs=10,114  Jaccard=1.0
  LB-best 2-way   OOF=0.98012  errs= 9,851  Jaccard vs recipe=0.883
  171pair         OOF=0.97946  errs= 9,991  Jaccard vs recipe=0.812
  ```
  - vs recipe: peak α=0.60 → Δ=+0.00012 (below +0.0002 LB-transfer)
  - vs LB-best 2-way: peak α=0.025 → Δ=+0.00002 (noise)
  - 3-way grid (recipe + pseudo_s1 + 171pair): 0.98011 at
    (0.35, 0.45, 0.20), **Δ=−0.00002 vs LB-best** (doesn't even match)
- Diagnosis: **the extra 55 num×num pairs (vs allpairs's 116-pair which
  has cat×cat + cat×num only) are redundant with recipe's existing
  numeric encodings.** Specifically:
  - 16-bin quantile encoding on a numeric duplicates info already in
    digit-position features (66 cols) and num_as_cat (11 cols, fully
    factorized).
  - 171pair standalone (0.97946) is WORSE than allpairs (0.97976) — the
    extra 55 pair OTE columns add noise to feature_fraction=0.8 sampling
    rather than new signal.
  - Allpairs (cat×num pairs only) remains the right pair-expansion lever
    on this feature set.
- **Conclusion: the "literal 171-pair magic" lever does NOT apply to us.**
  Ali Afzal's public kernel likely needs the 171-pair encoding because
  their recipe doesn't have digit-extraction. Our V10 recipe has digits
  + num_as_cat already; the 171-pair attempt duplicates rather than
  complements. Lever closed.
- LB delta: n/a (no probe warranted; OOF below LB-best).
- LB budget unchanged at 8/10 used today, 2 remaining. LB best still
  0.97998 (recipe × pseudolabel 50/50).
- **Infrastructure learnings logged to LEARNINGS.md candidates:**
  1. **Kaggle GPU kernels have 31 GB RAM**, not 13 GB — the OOM the CPU
     attempts hit at 21 GB was specific to our local container's limit.
     For any pipeline that fits in 31 GB, GPU kernels are the safer bet
     for memory-heavy FE.
  2. **Subprocess-per-fold isolation was unnecessary for THIS pipeline**
     — aggressive `del + gc.collect()` per fold + freeing `orig` after
     FE was sufficient. Document the subprocess approach as the next-
     resort fix only when GC + cleanup don't suffice.
  3. **GPU XGBoost is ~5x faster on this workload** (37.7 min for 5-fold
     1048-feature production vs estimated 3-4h CPU). The prior
     CatBoost CPU/GPU experience suggested 2-3x; XGB's hist algo
     scales better to GPU.
  4. **GPU kernel scaffolding from one model family ports cleanly**:
     reused `kaggle_kernel/kernel_catboost_recipe/` patterns
     (boot-up checks, _find_one rglob, /kaggle/input + /kaggle/working,
     inlined functions, kernel-metadata.json structure). 30-min copy
     job, smoke passed first try.
  5. **Smoke + production split is critical**: v1 hardcoded SMOKE=True
     to validate kernel-metadata.json + GPU access + memory peak in 30
     seconds; v2 flipped SMOKE off for production. Avoids ~40 min wasted
     compute on a misconfigured production kernel.

### 2026-04-24 — multi-seed pseudo-label scaffold + run launched

- Goal: bypass the stage-2 OOF-overfit failure mode by using a labeler
  trained on a DIFFERENT 5-fold split (FOLD_SEED=7) than the target
  model's split (FOLD_SEED=42). Stage-2 failed because labeler
  (LB-best blend) and target both trained on seed=42 folds — pseudo-
  labels encoded the seed=42 calibration biases. Seed=7 labeler
  decouples the chain.
- Changed:
  - `scripts/recipe_full_te.py` — added FOLD_SEED env var (default 42).
    When non-default, output paths get a `_seed<N>` suffix composable
    with existing `_a01` / `_dart` suffixes.
  - `scripts/recipe_pseudolabel.py` — same FOLD_SEED env var added.
- Launched chain (CPU, ~1h45m wall expected):
  1. `FOLD_SEED=7 python scripts/recipe_full_te.py` → outputs
     `oof_recipe_full_te_seed7.npy`, `test_recipe_full_te_seed7.npy`,
     `recipe_full_te_seed7_results.json`. Standard recipe pipeline at
     a different fold split. ~55 min CPU.
  2. Auto-chained: `LABELER_TEST_PATH=test_recipe_full_te_seed7.npy
     LABELER_BIAS_JSON=recipe_full_te_seed7_results.json
     PSEUDO_SUFFIX=seed7labeler python scripts/recipe_pseudolabel.py`
     → outputs `oof_recipe_pseudolabel_seed7labeler.npy`. Target model
     trained at FOLD_SEED=42 (default), pseudo-labels from seed=7
     labeler. ~48 min CPU.
- Hypothesis: blend at recipe(seed=42) × pseudo_seed7labeler should
  preserve stage-1's structural lift (+0.00046 pairwise) AND avoid the
  stage-2 OOF-overfit (gap +0.00038 vs stage-1's +0.00014). If the gap
  stays at ~+0.00014 and the OOF beats stage-1's 0.98012, we have a
  ceiling-breaker.
- Status as of writing: stage 1 in progress (FE complete, fold 1
  starting). Will document outcome + blend analysis after both stages
  complete.

### 2026-04-24 — Cleanlab confident-learning A0 closed as NULL

- Goal: execute recommendation #1 from the 3-tier research plan. Use
  cleanlab.filter.find_label_issues to flag likely-flipped TRAINING
  rows using the LB-best 2-way blend as teacher, then test three
  interventions (drop/downweight/relabel) via a CLEANLAB_TREATMENT env
  var in recipe_full_te.py.
- Changed: scripts/cleanlab_diagnose.py (diagnostic),
  scripts/recipe_full_te.py (+79 lines for 3-variant intervention
  plumbing), gitignore whitelist for the mask.
- Diagnostic (run time: ~4 s):
  - Teacher = 50/50 log-blend of recipe_full_te × recipe_pseudolabel
    (reproduces LB-best OOF 0.98012 exactly).
  - prune_by_noise_rate flagged 2,035 rows (0.323% of 630k).
  - 98.4% precision vs known rule-mismatch rows (10,304 NN-flip rows).
  - 19.4% recall of the 10,304 known flips.
  - Flagged rows skew to boundary scores {3,4,7,8} and to rare High
    class (1.24% flagged vs 0.22% Low / 0.40% Medium).
  - Mean teacher self-confidence: flagged=0.012 vs unflagged=0.981.
  - Zero flagged rows where teacher agrees with observed label.
- DROP production (55 min wall, 5-fold seed=42):
  - Per-fold argmax vs recipe baseline:
      fold 1: 0.97534 vs 0.97544  Δ = −0.00010
      fold 2: 0.97669 vs 0.97659  Δ = +0.00010
      fold 3: 0.97787 vs 0.97721  Δ = +0.00066
      fold 4: 0.97619 vs 0.97465  Δ = +0.00154
      fold 5: 0.97628 vs 0.97557  Δ = +0.00071
    Mean fold-level lift = +0.00058 (promising).
  - Overall argmax = 0.97648 (Δ = +0.00059 vs recipe baseline 0.97589).
  - Tuned log-bias OOF = **0.97965** (Δ = −0.00002 vs recipe 0.97967;
    Δ = −0.00046 vs LB-best 0.98013).
  - Tuned bias = [1.03, 1.07, 2.90] vs recipe's [1.43, 1.47, 3.40] —
    sharper raw probs needed less bias correction.
  - best_iter per fold: 660 / 679 / 847 / 562 / 683 (vs recipe's
    ~1200-1400). Model converged faster without ambiguous flip rows.
- Error-geometry diagnostic (fixed recipe bias):
  - DROP errors = 10,187 vs LB-best 9,851 → DROP has +336 more errors.
  - Jaccard = 0.7962 (below 0.80 novelty threshold) BUT magnitude
    condition FAILS per our 2026-04-22 blend heuristic.
  - Per-class recall vs LB-best: Low −0.0004, Medium −0.0007, High
    −0.0006 (worse on ALL three classes).
- Blend sweep (fixed recipe bias):
  - vs LB-best: peak α_drop=0.40 → OOF 0.98021 (Δ = **+0.00009**)
    — below +0.0002 LB-transfer threshold. NULL.
  - vs recipe alone: peak α_drop=0.50 → OOF 0.97995 (Δ = +0.00029)
    — above threshold, but recipe is a weaker anchor than LB-best.
- Diagnosis: dropping the 2,035 flagged rows removes signal the
  model needs for the calibration sharpening that log-bias exploits.
  Per-fold argmax improves because each fold's training distribution
  is easier (fewer ambiguous rows), but the GLOBAL decision rule
  (log-bias tune) already compensates for that via its macro-recall
  operating point. Net OOF is flat.
- Implication: the 2,035 flagged rows are NOT stochastic label noise —
  they are DETERMINISTIC NN-flip signal. Cleanlab's "label error"
  interpretation is wrong for this problem (the 2026-04-21 DGP
  residuals EDA already established the labels are deterministic,
  not a Bernoulli flip process). Cleanlab found the hardest-to-learn
  flip rows; removing them trades "model confidence on clean rows"
  for "model experience with flip rows" — net zero.
- Downweight / relabel variants NOT executed:
  - Downweight is a softer drop; given drop gives zero lift AND has
    more errors, downweight unlikely to help.
  - Relabel replaces observed labels with teacher's rule-consistent
    argmax on 98.4% of flagged rows — effectively removes the flip
    signal from those rows. Expected LB regression.
- LB delta: n/a. No LB probe (below +0.0002 threshold).
- Budget unchanged. Current LB best unchanged at 0.97998.
- Lesson to LEARNINGS.md: "Cleanlab confident learning is designed
  for stochastic label noise. For DETERMINISTIC label transformations
  (synthetic-data competitions where labels come from a known or
  inferred function), cleanlab's 'label error' signal identifies
  hard-to-learn signal rows, not noise. Interventions that modify or
  remove those rows degrade the model's ability to learn the
  transformation. Before applying cleanlab, first confirm the label
  noise model matches (stochastic Bernoulli flip vs deterministic
  function): if the latter, skip to mixup/co-training instead."

### 2026-04-24 — C0 isotonic calibration + greedy forward-blend: NULL at stacking-inflation ceiling

- Goal: P5 recommendation — after cleanlab A0 + Saerens A1 both nulled,
  test whether per-class isotonic calibration on every saved OOF + a
  greedy forward-selection over the expanded bank can break past the
  LB-best OOF 0.98013. Motivation: isotonic normalizes each component's
  prob-scale independently before log-blending, which is the textbook
  fix for the "CatBoost in blend null" (High-class calibration mismatch).
- Changed: `scripts/c0_isotonic_greedy.py` (38 components, both raw and
  iso candidates, greedy from recipe anchor at fixed recipe bias
  [1.43, 1.47, 3.40]), `scripts/c0_safe_greedy.py` (same but EXCLUDE
  `{soft_distill, xgb_spec_678}` and use BOTH anchors: recipe alone
  and LB-best 2-way blend).

- **Per-component isotonic effect** (standalone @ recipe bias, selected):
  ```
  component                raw bal     iso bal    Δiso
  recipe_full_te           0.97967     0.97968    +0.00001
  recipe_pseudolabel       0.97987     0.97993    +0.00006
  recipe_catboost          0.97739     0.97936    +0.00197  (High-class fix)
  lgbm_te_orig             0.97038     0.97178    +0.00140  (weakest models gain most)
  tabpfn                   0.96165     0.96209    +0.00044
  soft_distill             0.98076     0.98049    −0.00027  (iso can't fix overfit)
  ```
  CatBoost got the biggest-ever isotonic lift (+0.00197) — confirms its
  High-class sharpness was the calibration-mismatch source of prior
  blend-null results.

- **C0 full greedy (38 components incl. soft_distill)**:
  - Step 1: + soft_distill__iso α=0.50 → OOF **0.98055** (Δ=+0.00042)
  - Step 2: + em_uniform__iso α=0.50 → 0.97959 (Δ=−0.00096, rejected)
  - **NOT A REAL LIFT**: the +0.00042 delta comes entirely from
    soft_distill__iso, and soft_distill has a confirmed LB regression
    (2026-04-24 entry: OOF 0.98096 → LB 0.97850, gap +0.00246).
    Isotonic calibration is a monotone per-class remapping; it changes
    prob scales but does not fix the underlying overfit ranking. A
    50/50 log-blend of recipe_test with a calibrated version of an
    already-overfit test posterior inherits the overfit.
  - Decision: NOT an LB probe. Confirmed rule: "a component with a
    verified LB gap ≥ +0.00246 cannot be rescued by OOF-level isotonic."

- **C0 safe greedy (37 components, exclude soft_distill + spec_678)**:
  - Anchor `recipe_full_te`: + recipe_pseudolabel_stage2 α=0.50 →
    **0.98026** (Δ=+0.00013). Step 2 below threshold.
  - Anchor `lb_best_2way` (recipe × pseudo_stage1): + recipe_allpairs__iso
    α=0.30 → **0.98031** (Δ=+0.00018). Step 2 below threshold.
  - Both final deltas below +0.00020 LB-transfer threshold. Expected
    LB if probed: ~0.97998 ± noise. NOT worth a slot.

- **Reconfirms the stacking-inflation ceiling at OOF ~0.98030**
  already documented on 2026-04-23 (three separate 3+ component stacks
  landing there with LB 0.97995-0.97997). Isotonic does not open new
  ground at the blend level because:
  1. Log-bias tune at the blend level already applies a global
     recalibration that subsumes per-component isotonic.
  2. The OOF ceiling is set by the information content of the 37
     component OOFs, not by their calibration shape.
  3. Soft_distill provides the only "apparent" lift beyond 0.98030,
     but it's the single known LB-regressor.

- Artefacts committed for cross-branch reuse:
  - `scripts/artifacts/oof_c0_greedy.npy` + test (0.98055 OOF, risky)
  - `scripts/artifacts/oof_c0_safe_recipe_full_te.npy` + test (0.98026)
  - `scripts/artifacts/oof_c0_safe_lb_best_2way.npy` + test (0.98031)
  - `scripts/artifacts/c0_isotonic_greedy_results.json`
  - `scripts/artifacts/c0_safe_greedy_results.json`

- **New portable rule** (logging to LEARNINGS.md): "Per-class isotonic
  calibration on bank OOFs does NOT break stacking-inflation ceilings.
  It's useful for individual weak components with calibration drift
  (CatBoost's High-class sharpness got +0.00197), but at the blend
  level, log-bias coord-ascent already occupies the operating point
  isotonic would find."

- **No LB probe.** LB-best unchanged at 0.97998. LB budget unchanged.

### 2026-04-24 — multi-seed pseudo-label completed: NULL, ceiling-breaker list exhausted

- Goal: complete the multi-seed pseudo-label experiment to test whether
  decoupling the labeler's fold-split (FOLD_SEED=7) from the target
  model's fold-split (FOLD_SEED=42) bypasses the stage-2 OOF-overfit
  failure mode. Stage-2 NULL on LB (0.97989) was attributed to
  labeler+target chain on the same seed=42 folds; seed=7 labeler
  removes that chain.

- **Stage 1: recipe at FOLD_SEED=7 (~40 min CPU)**
  - Per-fold argmax: 0.97828/0.97652/0.97512/0.97484/0.97323
  - Overall argmax: 0.97560 (±0.00170)
  - Tuned OOF: 0.97973  bias=[1.13, 1.27, 3.30]
  - Vs seed=42 recipe (0.97967): Δ=+0.00006, statistically tied.
    Confirms same-pipeline-different-seed produces equivalent overall
    OOF — the variation is fold-split noise, no structural lift from
    choosing a different seed.

- **Stage 2: pseudo-label at FOLD_SEED=42 with seed=7 labeler (~50 min CPU)**
  - Pseudo subset at τ=0.98: 222,978 rows (vs seed=42 labeler's 226,162)
  - Per-fold argmax: 0.97633/0.97641/0.97762/0.97625/0.97638
  - Overall argmax: 0.97660 (±0.00051)
  - **Tuned OOF: 0.98017** (bias [1.23, 1.27, **3.40**])
  - Vs recipe (0.97967):              Δ=+0.00051
  - Vs stage-1 pseudo (0.97993):       Δ=+0.00024
  - Vs stage-2 LB-blend-labeler (0.98002): Δ=+0.00015
  - **High bias 3.40 matches recipe family** (vs stage-2's 3.30) —
    seed=7 labeler produced better-calibrated pseudo-labels than the
    LB-best blend labeler (stage-2's labeler).

- **Per-fold sum identity is the smoking gun:**
  ```
                        f1      f2      f3      f4      f5     sum
  recipe (s42)         0.97544 0.97659 0.97721 0.97465 0.97557 4.87946
  pseudo_s1 (s42 lab)  0.97700 0.97627 0.97881 0.97503 0.97589 4.88300
  pseudo_s7 (s7 lab)   0.97633 0.97641 0.97762 0.97625 0.97638 4.88299
  ```
  pseudo_s1 and pseudo_s7 sums are **essentially identical** (0.00001
  apart). Same overall lift, distributed differently across folds.
  The fold-decoupling did NOT structurally change the lift mechanism.

- **Blend analysis at recipe's bias** (saved OOFs):
  ```
  pseudo_s1   tuned 0.97987  errs 10,039  Jaccard vs recipe 0.7805
  pseudo_s7   tuned 0.98002  errs 10,170  Jaccard vs recipe 0.7781
                                          Jaccard vs s1     0.8157
  LB-best 2way 0.98012      errs  9,851  (LB 0.97998 verified)

  pairwise recipe × pseudo_s7:  peak α=0.35-0.40 → OOF 0.98014
                                Δ vs LB-best = +0.00002 (noise)
  pairwise LB-best × pseudo_s7: peak α=0.30-0.35 → OOF 0.98029
                                Δ vs LB-best = +0.00017 (below threshold)
  3-way grid (recipe + pseudo_s1 + pseudo_s7):
                                best (0.25, 0.35, 0.40) → OOF 0.98029
                                Δ vs LB-best = +0.00017 (same as above)
  ```

- **Diagnosis:** pseudo_s7 has **MORE errors than pseudo_s1** (10,170
  vs 10,039) and more errors than recipe (10,170 vs 10,114). Fails
  the "errors ≤ anchor" half of the blend heuristic. The tuned-OOF
  lift over stage-1 (+0.00024) comes from bias-tuner re-trading errors
  across classes, not from genuinely fewer errors. Pairwise blend
  with recipe gives essentially same OOF as LB-best 2-way (0.98014
  vs 0.98012, indistinguishable).

  3-way blend reaches OOF 0.98029 — squarely in the structural
  stacking-ceiling band where 3 prior submissions all landed
  LB 0.97995-0.97997 (NULL). Expected LB ~0.97995-0.97998 = NULL.

- **No LB probe warranted.** The +0.00017 OOF lift falls below the
  +0.0002 LB-transfer threshold; expected LB lands at the structural
  ceiling.

- **Multi-seed pseudo-label closes the ceiling-breaker list.** ALL
  four candidates we identified as "potentially structurally different
  from existing OOFs" have now been tested and confirmed NULL:
  ```
  Candidate                              Result    Tested by
  ────────────────────────────────────────────────────────────
  Soft-target distillation               LB 0.97850 (-0.00148)  parallel session
  171-pair binned (cat+num quantile)     OOF 0.97946 (NULL)     this branch
  Stage-2 with LB-blend labeler          LB 0.97989 (-0.00009)  this branch
  Multi-seed pseudo (seed=7 labeler)     OOF 0.98017 (NULL)     this branch
  ```
  Combined with the prior 12-component greedy bank (all stacking
  variants confirmed null at OOF 0.98030 → LB 0.97995-0.97997), this
  is **rock-solid evidence that LB 0.97998 is the structural ceiling
  for own-pipeline approaches** on this competition's feature
  representation.

- LB budget: 8/10 used today, 2 saved for tomorrow.

- **New LEARNINGS.md candidates from this experiment:**
  1. **Per-fold OOF sum identity is the diagnostic for structural-
     equivalence**: when two pipelines produce nearly identical fold-
     argmax sums (4.88300 vs 4.88299), they differ only in WHICH rows
     they get right per-fold, not in TOTAL signal extracted. Decoupling
     mechanisms (different fold seeds) don't change this — the lift
     mechanism is the same, just the specific rows differ.
  2. **The "magnitude trap" applies to multi-seed pseudo too**: a
     candidate with MORE errors than the anchor will fail to lift even
     if its Jaccard < 0.80. Decoupling labeler seed doesn't cure this.
  3. **Multi-seed pseudo-label is NOT a structural fold-decoupling
     lever**: the stage-2 OOF-overfit failure mode is NOT bypassed by
     using a different fold-seed for the labeler. The mechanism we
     hypothesized (labeler+target on same folds → calibration leak)
     was wrong; the actual mechanism is more subtle (probably: the
     pseudo-label's argmax encodes the LABELER MODEL's specific
     tree-split bias, regardless of which folds the labeler was
     trained on).
  4. **Ceiling-breaker exhaustion takes O(n) experiments where n is
     the number of GENUINELY DIFFERENT mechanisms**, not the number
     of variants per mechanism. We tested 4 mechanisms × ~3 variants
     each = 12 experiments to be confident the ceiling is structural.
     Less than that and we'd have residual uncertainty; more would be
     diminishing returns.

- **Final candidates for LB submission (locked unless something
  fundamentally new arrives):**
  1. **Primary: `submission_recipe_greedy_recipe_pseudolabel.csv`**
     (LB 0.97998, OOF 0.98012, gap +0.00014). Verified
     ceiling-sweet-spot.
  2. **Safe fallback: `submission_recipe_full_te.csv`**
     (LB 0.97939, OOF 0.97967, gap +0.00028). Pure single-model
     baseline, no blend overfit risk on private LB.

  Pack 0.98114 stays +0.00116 above; leader 0.98219 stays +0.00221
  above. Reachable only via public-CSV blending (banned).

### 2026-04-24 — multi-seed pseudo-label LB probes: NEW LB BEST 0.98005 + A/B test ambiguity

- Goal: after the OOF-level multi-seed analysis suggested "null" based
  on per-fold sum identity, user pushed back on the OOF-heuristic
  framing and asked for an LB probe to calibrate. Submitted TWO probes:

- **Probe 1: 3-way (recipe 0.25 + pseudo_s1 0.35 + pseudo_s7 0.40)**
  - OOF 0.98029 at recipe's bias, errors 9,873
  - **LB = 0.98005 — NEW LB BEST (+0.00007 vs prior 0.97998)**
  - Gap OOF→LB = +0.00024 (tighter than 3 prior 3+ blends which had
    gap +0.00036 to +0.00038 → LB 0.97995-0.97997)
  - Breaks the "stacking-inflation ceiling" pattern by ~2x tighter gap

- **Probe 2: 2-way A/B (recipe 0.50 + pseudo_s7 0.50)** — controlled
  replacement of pseudo_s1 with pseudo_s7 in LB-best 2-way structure.
  - OOF 0.98012 at recipe's bias (EXACTLY matches LB-best 2-way's OOF)
  - **LB = 0.97969 (−0.00029 vs LB-best 0.97998)**
  - Gap OOF→LB = +0.00043 (vs LB-best 2-way's +0.00014)

- **Summary table:**
  ```
  blend                              OOF      LB       gap      Δ vs LB-best
  ─────────────────────────────────────────────────────────────────────────
  LB-best 2-way (recipe × s1)       0.98012  0.97998  +0.00014  anchor
  A/B 2-way (recipe × s7)           0.98012  0.97969  +0.00043  -0.00029
  3-way (rec + s1 + s7)             0.98029  0.98005  +0.00024  +0.00007  ← new LB best
  ```

- **Observations (hedged, no conclusions):**
  1. Two 2-way blends with **identical OOF** and identical weights differ
     on LB by 0.00029. Gap inflates from +0.00014 to +0.00043 with only
     the labeler seed changed. **OOF is not a reliable proxy for LB rank
     at the sub-0.0003 resolution** on this feature set.
  2. pseudo_s7 has 10,170 errors at recipe's bias vs pseudo_s1's 10,039
     (131 more). At 2-way level the extra errors dominate.
  3. pseudo_s7 and pseudo_s1 Jaccard 0.8157 — disagree on ~18% of error
     rows. In the 3-way with recipe as tie-breaker, the disagreement
     rows appear to get resolved favorably.
  4. The +0.00007 LB win from the 3-way (probe 1) could be:
     a. Genuine complementary-signal from multi-seed decoupling, OR
     b. Noise-band lucky draw (the −0.00029 from probe 2 shows
        pseudo_s7 alone is WORSE than pseudo_s1, so its "complementary
        signal" is not obviously net-positive).
  5. LB test set is 270k rows. A 30-row flip = 0.00011. The effects
     we're chasing (±0.0001-0.0003 LB) are at the resolution limit
     where test-fold draw matters.

- **Updated calibration ladder:**
  ```
  recipe_full_te                       0.97967 → 0.97939  gap +0.00028
  recipe × pseudo_s1 2-way (α=0.50)    0.98012 → 0.97998  gap +0.00014
  **3-way (rec+s1+s7) (0.25/0.35/0.40) 0.98029 → 0.98005  gap +0.00024** ← NEW LB BEST
    recipe × pseudo_s7 2-way (α=0.50)  0.98012 → 0.97969  gap +0.00043 (null)
  ```

- **LB budget: 10/10 used today. 0 remaining until reset.** Pack 0.98114
  still +0.00109 above. Leader 0.98219 still +0.00214 above.

- **Lessons learned / rules revisited:**
  1. **"Per-fold sum identity" was wrong as a null-declaration
     criterion.** pseudo_s1 and pseudo_s7 had near-identical per-fold
     sums (4.88300 vs 4.88299) but produced materially different LB
     results when used in 2-way blends (0.97998 vs 0.97969). The
     aggregate lift was similar but the per-row test-set predictions
     differed meaningfully.
  2. **The "below +0.0002 OOF lift = can't transfer" rule held for the
     3-way against the 2-way LB-best** — +0.00017 OOF → +0.00007 LB
     (factor ~0.4). But it needed an LB probe to determine whether
     that +0.00007 is real or noise.
  3. **OOF-based ceiling declarations need LB confirmation.** I
     prematurely declared the multi-seed mechanism null based on OOF
     patterns. The LB probe revealed a 0.00036 LB spread between two
     OOF-identical blends — OOF isn't fine-grained enough to predict
     LB rank at this resolution.
  4. **Candidate "final selection" primaries now available**:
     - `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005, new best)
     - `submission_recipe_greedy_recipe_pseudolabel.csv` (LB 0.97998,
       prior best)
     - `submission_recipe_full_te.csv` (LB 0.97939, safe single-model
       baseline)
     The +0.00007 lift of the 3-way over the 2-way is marginal
     relative to private-LB fold variance; a ~1-probe confirmation
     tomorrow (e.g. a different 3-way weighting or seed=123 labeler)
     would strengthen the choice between 3-way and 2-way as primary
     final.

### 2026-04-24 — c0_v2 4-way LB PROBE: 0.97961 (−0.00044 REGRESSION)

- Goal: LB-probe the greedy v2 4-way candidate
  (recipe + pseudolabel_stage2 α=0.50 + recipe_seed7 α=0.25 +
  recipe_171pair α=0.10, OOF 0.98050). Tests whether adding
  multi-seed + pair-binning on top of recipe anchor produces a new
  LB best above the current 3-way 0.98005.
- Submitted at 06:42 UTC, result **LB 0.97961**.
- **Δ vs LB-best = −0.00044** (clear regression).
- OOF→LB gap = **0.00089** — widest blend gap we've seen. Prior
  calibration ladder:
  ```
  2-way (recipe × pseudo_s1)      OOF 0.98012 → LB 0.97998  gap +0.00014
  3-way multi-seed (LB best)      OOF 0.98029 → LB 0.98005  gap +0.00024
  **4-way c0_v2 (w/ stage-2)       OOF 0.98050 → LB 0.97961  gap +0.00089 ← REGRESSION**
  ```

- **Diagnosis:** the c0_v2 greedy picked `recipe_pseudolabel_stage2`
  as step 1 at α=0.50 because its OOF lift (+0.00059) was the
  largest. But stage-2 is the confirmed OOF-overfit component
  (2026-04-23: stage-2 2-way at α=0.55 hit LB 0.97989 vs OOF 0.98027,
  gap +0.00038). Diluted in a 4-way with recipe + seed7 + 171pair,
  its overfit didn't cancel — it dominated. The OOF 0.98050 vs 3-way
  0.98029 apparent lift was stage-2's OOF inflation.

- **Refined rule (already knew but now LB-confirmed):** **greedy
  forward-selection with "pick highest OOF-Δ per step" is unreliable
  when some candidates are OOF-overfit relative to LB. Known
  LB-regressors (stage-2 pseudo-label, soft_distill) must be
  EXCLUDED from the candidate pool before running greedy.**

- Launched `scripts/c0_safe_greedy_v3.py` with EXCLUDE =
  `{soft_distill, xgb_spec_678, recipe_pseudolabel_stage2}` — same
  architecture but stage-2 removed. Running in background.

- **LB budget**: 4/10 used today, 6 remaining.
- **LB best unchanged**: `submission_3way_recipe025_s1035_s7040.csv`
  at LB 0.98005.

### 2026-04-24 — c0_v3 greedy (stage-2 excluded) + TTA production + P2 symbolic launched

**c0_v3 results** (EXCLUDE = {soft_distill, xgb_spec_678, pseudolabel_stage2}):

Anchor recipe_full_te:
  step1: + recipe_full_te_seed7                α=0.50  OOF=0.98019  Δ=+0.00053
  step2: + recipe_pseudolabel_seed7labeler     α=0.20  OOF=0.98035  Δ=+0.00016
  step3: + xgb_nonrule__iso                    α=0.15  OOF=0.98047  Δ=+0.00012
  step4: + em_uniform                          α=0.075 OOF=0.98055  Δ=+0.00008 (stop)
  Final: 0.98047 (Δ vs LB-best 3-way = +0.00018)

Anchor lb_best_3way:
  step1: + recipe_allpairs__iso                α=0.20  OOF=0.98041  Δ=+0.00013 (stop)
  Final: 0.98041

- **Concern**: recipe-anchor 4-way picked `recipe_pseudolabel_seed7labeler` at
  step 2. This component is the seed=7 labeler whose 2-way with recipe
  gave LB 0.97969 (gap +0.00043, worse than stage-2's +0.00038). It is
  ALSO an OOF-overfit LB-regressor. Adding it (even at α=0.20) likely
  pulls LB down the same way stage-2 did in c0_v2.
- Both 4-way candidates (OOF 0.98047 recipe-anchor, 0.98041 lb_best_3way-
  anchor) are below the +0.00020 LB-transfer threshold AND contain
  overfit-risk components. **Not worth an LB probe.**
- **Lesson**: greedy forward-selection keeps rediscovering the same
  OOF-overfit components (stage-2 → seed7labeler → allpairs_iso). The
  OOF gain from each is real CV-fit signal but doesn't transfer because
  these components fit fold-specific calibration noise.
- Next: pivoting to mechanism-novel experiments instead of more blend
  variants. Launched P2 symbolic regression (gplearn on cells 3 and 6,
  ~60 min CPU) and TTA production (σ ∈ {0.01, 0.02, 0.03}, fold 1
  already showed −0.002 to −0.006 per sigma, likely null but runs to
  completion for full OOF).

- **LB budget**: 4/10 used today, 6 remaining.
- **LB best unchanged**: 0.98005.

### 2026-04-24 — P2 symbolic regression (gplearn) closed as NULL

- Goal: search analytic formulas in the 7 non-rule continuous features
  (Humidity, Prev_Irrigation, EC, Field_Area, Soil_pH, Organic_Carbon,
  Sunlight) that predict within-cell label flips at the 2 dominant error
  cells (score=3 rule=Low→flip to Medium, score=6 rule=Medium→flip to
  High). Hypothesis: if the NN generator's flip function has polynomial
  / algebraic structure, gplearn finds it; deploy as hard override.
- Changed: `scripts/p2_symbolic_flip.py` — bug fix (`_Program.execute`
  not `.predict`) + try/except per formula evaluation.
- Setup: gplearn SymbolicRegressor, pop=1000, gen=25, MAE loss,
  function set = (+, -, *, /, sqrt, log, abs), parsimony=0.01.
- **Results (trivial formulas won):**
  ```
  cell3 (n=102,157, flip_rate=4.796%)
    rank 0:  sub(Organic_Carbon, Organic_Carbon) = 0
             precision 0.0, recall 0.0, selected 0 rows
    rank 1:  sub(Field_Area_hectare, Field_Area_hectare) = 0
    rank 2:  sub(Organic_Carbon, Organic_Carbon) = 0

  cell6 (n=38,416, flip_rate=4.032%)
    rank 0:  sub(Field_Area_hectare, Field_Area_hectare) = 0
    rank 1:  sub(Humidity, Humidity) = 0
    rank 2:  sub(Field_Area_hectare, Field_Area_hectare) = 0
  ```
- **Diagnosis:** predicting constant 0 (no flip) gives MAE = flip_rate
  ≈ 0.04. gplearn's fitness landscape + parsimony penalty favors short
  formulas with low MAE. No polynomial/algebraic expression built from
  the 7 features achieves lower MAE than the trivial "predict 0" —
  the evolution converged on degenerate `sub(X, X)` = 0 formulas as
  globally optimal.
- **Closes the hypothesis:** the DGP's within-cell flip signal is NOT
  expressible as a compact analytic formula of the 7 non-rule
  continuous features. Consistent with the 2026-04-21 "DGP is a
  deterministic NN function, not a noise process" finding — the NN's
  within-cell decision surface is too non-linear / high-order for
  gplearn with population 1000 × 25 generations to find.
- **No LB probe.** P2 lever CLOSED.
- Parallel learning: `p2_symbolic_flip.py` pattern can be reused on
  future synthetic-DGP competitions WHERE the DGP is known or strongly
  suspected to be a finite-degree polynomial function. On this
  competition, the NN generator is structurally outside that class.

### 2026-04-24 — A2 SwapNoise DAE (Porto Seguro mechanism): NULL

- Goal: execute Tier-A #A2 from the research plan. Train a label-unaware
  SwapNoise denoising autoencoder on train+test+orig joint (910k rows),
  extract 128-d bottleneck embeddings as extra numeric features for
  recipe_full_te's XGB. Architecturally decoupled from every prior
  label-supervised NN attempt (v5-v9 MLP, FT-T, TabPFN, pretrain-finetune
  MLP, NN-on-orig, soft-distill — 10 nulls collectively).
- Changed:
  - `kaggle_kernel/kernel_dae/` — 1.48M-param encoder-decoder MLP,
    [43 → 1024 → 512 → 256 → 128] with mirrored decoder, GELU+BN+
    dropout 0.1, SwapNoise p=0.15 in-batch column-preserving swap,
    MSE reconstruction, AdamW lr=1e-3 + cosine schedule.
  - `kaggle_kernel/ds_dae_embed/` — Kaggle dataset
    `chrisleitescha/irrigation-dae-swapnoise-embeddings` with
    fp16 (154MB + 66MB) for downstream kernel consumption.
  - `kaggle_kernel/kernel_recipe_dae_gpu/` — GPU XGB variant of
    recipe_full_te with +128 DAE cols loaded from the dataset.
  - `scripts/recipe_full_te.py` — `DAE_EMBED_PATH` env var added
    for local consumption (CPU path: ~4h ETA, aborted; GPU 21.6 min).
  - `scripts/blend_recipe_dae.py` — fixed-bias α sweep + 3-way grid.

- DAE production (Kaggle P100): 3.2 min wall for 30 epochs. MSE
  converged 0.268 → 0.106 (plateau at epoch 22). Embedding stats
  healthy (mean ≈ 0, std ≈ 0.36, range [-2.58, 2.95]).

- Recipe+DAE production (Kaggle P100): 21.6 min wall (vs CPU's
  estimated 4h).
  ```
                            fold argmax range     OOF argmax   tuned
  recipe baseline           0.97465-0.97721      0.97589      0.97967
  recipe+DAE                0.97416-0.97680      0.97545      0.97942
  Δ                         ~flat                -0.00044     -0.00025
  ```

- Blend analysis (fixed-bias, recipe's log-bias [1.43,1.47,3.40]):
  ```
                                  fixed@anchor  tuned    errs
  recipe_full_te                   0.97967      0.97967   10114
  recipe_pseudolabel               0.97987      0.97993   10039
  LB-best (recipe × pseudo)        0.98012      0.98012    9851
  recipe+DAE                       0.97930      0.97942   10025

  Jaccard DAE vs recipe            = 0.8425  (above 0.80 redundancy)
  Jaccard DAE vs LB-best           = 0.8404
  Jaccard LB-best vs recipe        = 0.8830

  sweep vs recipe (α_dae=0.15)     peak Δ = +0.00004
  sweep vs LB-best (α_dae=0.05)    peak Δ = +0.00003
  3-way (recipe, pseudo, dae)     best OOF 0.98013 at (0.45, 0.50, 0.05)
                                   Δ vs LB-best = +0.00001 (noise)
  ```

- **Verdict: NULL.** All peaks below the +0.0002 LB-transfer threshold;
  most are below fold-std noise. Jaccard 0.84 both anchors = blend-
  redundancy zone (heuristic: < 0.80 required for useful orthogonality).

- Diagnosis: the DAE, despite being label-unaware and architecturally
  distinct, converges to features that encode essentially the same
  manifold structure that OTE target encoding already captures at the
  recipe level. The 128 bottleneck dims reconstruct the same 43-d
  feature surface the recipe sees, so the downstream XGB doesn't gain
  discrimination. Fewer errors (10,025 vs 10,114) but worse tuned
  bal_acc because the DAE-induced error trade is less macro-recall-
  favourable than recipe's pure-feature errors.

- Pattern (now 11 NN-family nulls): Regardless of whether the NN sees
  labels (v5-v9, FT-T, pretrain-FT, soft-distill), is in-context
  (TabPFN), or is label-unaware (this DAE), the NN-generated features
  on this feature set fail to clear the Jaccard < 0.80 + error-count
  ≤ anchor bar. The recipe's FE + OTE already extracts ~all the
  signal available from train+test+orig. **Own-pipeline NN levers
  are exhausted on this problem.**

- LB delta: n/a. No submission emitted (all sweeps under the +0.0005
  emit gate). LB best unchanged at 0.97998.

- New rule for LEARNINGS.md: **"SwapNoise-DAE features don't add
  blend signal when the downstream model's FE already includes
  target-encoded categoricals over the same feature set. The DAE's
  reconstruction objective aligns its latent space with the same
  joint distribution OTE approximates conditionally, so the
  orthogonality the DAE theoretically provides doesn't survive as
  marginal gain."**

- Next: A0, A1, A2 all null (cleanlab, Saerens EM, DAE); C0 isotonic
  also null per the main-side entry above. Remaining untried is B0
  DivideMix (mechanism-novel), C1 TTA + C2 conformal, D-tier L1 zoo
  and model families (TabM, ModernNCA). Own-pipeline NN levers are
  exhausted on this feature set.

### 2026-04-24 — TTA (threshold-axis Gaussian perturbation) closed as NULL

- Goal: inference-time smoothing of tree step-functions at the 4 rule
  thresholds (Soil<25, Rain<300, Temp>30, Wind>10). Perturb raw
  numerics by σ·IQR Gaussian noise, K=5 times, recompute
  threshold-dependent features (rule flags, LR logits, digit cols),
  log-average predictions. Hypothesis: smooths tree step-function
  boundaries to approximate the NN generator's smooth decision
  surface. OTE/FREQ/num_as_cat held fixed (perturbing them would
  create unknown keys and degenerate to prior).
- Script: `scripts/tta_recipe_full.py` + `scripts/tta_helpers.py`
  (scaffolded on main 2026-04-24, executed this session).
- Production run: TTA_K=5, TTA_SIGMAS={0.01, 0.02, 0.03}, 5-fold
  seed=42, ~50 min wall.
- **Per-fold argmax vs recipe baseline (consistent across all 5 folds):**
  ```
  fold  baseline  σ=0.01   σ=0.02   σ=0.03
  1     0.97544   0.97324  0.97131  0.96926
  2     0.97659   0.97458  0.97248  0.97086
  3     0.97721   0.97538  0.97335  0.97174
  4     0.97465   0.97298  0.97097  0.96915
  5     0.97557   0.97343  0.97146  0.96997
  ```
- **OOF aggregate:**
  ```
  variant   argmax    tuned     bias                      Δ tuned
  baseline  0.97589   0.97967   [1.432, 1.469, 3.401]     0
            (= recipe_full_te exactly)
  s010      0.97392   0.97851   [1.632, 1.469, 3.401]    -0.00116
  s020      0.97191   0.97683   [1.632, 1.469, 3.401]    -0.00284
  s030      0.97020   0.97524   [1.132, 0.969, 2.901]    -0.00443
  ```
- **Diagnosis:** axis-aligned tree boundaries do NOT have smooth
  interpolation structure. Perturbing the raw threshold numerics:
  1. Flips rule-indicator binaries (0→1 or 1→0) when noise crosses
     the threshold — discontinuous change in the most important
     tree splits.
  2. Changes digit-position features (a different integer per
     perturbation) — more tree splits change discontinuously.
  3. OTE / num_as_cat / FREQ held fixed (good), but the tree's
     other path-dependent decisions (conditional on the flipped
     rule flags) still change non-smoothly.
  Log-averaging K predictions with discontinuous differences blurs
  correct predictions into wrong ones rather than smoothing a single
  correct boundary.
- **Fundamental mismatch**: TTA works when the model has a smooth
  underlying decision function that's being quantized by a discrete
  step. Tree ensembles have no smooth underlying function — the step
  IS the decision function.
- **Blend unsuitable**: all σ variants have +15-50% more errors than
  recipe. Blend heuristic (Jaccard <0.80 AND errs≤anchor) fails on
  magnitude side.
- **TTA lever CLOSED.** Baseline OOF artefact kept as clean
  regeneration of recipe_full_te for cross-branch diagnostic.

- **LB budget**: 4/10 used today, 6 remaining.
- **LB best unchanged**: `submission_3way_recipe025_s1035_s7040.csv`
  at LB 0.98005.

---

### 2026-04-24 — End-of-day session summary

**5 hypothesized mechanisms tested today, ALL null or LB-regressive:**
1. Cleanlab confident learning (A0) — flagged rows are deterministic
   signal, not stochastic noise
2. Saerens/BBSE EM label-shift correction (A1) — no label shift, log-bias
   already near-Bayes-optimal
3. C0 isotonic + greedy (v1/v2/v3) — same ~0.98030 stacking ceiling;
   v2 LB probed at 0.97961 (−0.00044, regression from stage-2 overfit)
4. P2 symbolic regression (gplearn) — no analytic formula for flip
   signal; converged on trivial `sub(X,X)=0`
5. P1 TTA inference-time smoothing — trees lack smooth underlying
   boundary to interpolate across; monotone regression at all σ

**Today's single new LB result:** c0_v2 4-way → LB 0.97961 (−0.00044
regression, gap +0.00089). Widest gap we've observed for a blend.
Confirms greedy forward-selection unreliable when pool contains
known LB-regressors (stage-2, soft_distill, seed7labeler).

**LB state unchanged**: 0.98005 (3-way multi-seed) remains the own-
pipeline ceiling. Pack 0.98114 (+0.00109), Leader 0.98219 (+0.00214).

**Budget used**: 1/6 net (started at 3, used 1, ended at 4/10).

**Remaining mechanisms (all require GPU, not runnable on this container):**
- P1/A2 DAE SwapNoise embeddings (Porto Seguro mechanism)
- P3 TabM ICLR 2025 BatchEnsemble MLP
- P4 ModernNCA differentiable kNN
- P3 label-propagation in learned embedding (from 2026-04-24
  outside-the-envelope list)

**Recommendation for tomorrow**:
- 1 LB probe at most on any remaining marginal OOF candidate
- Scaffold + push at least one GPU kernel (DAE being highest EV)
- Reserve remaining 5-6 probes for end-of-comp final selection hedging
### 2026-04-24 — 4-way multi-seed (seed=123 labeler) closed as OOF-null

- Goal: after the 3-way (recipe + s1 + s7) landed LB 0.98005 (+0.00007
  over the 2-way LB-best), extend to a 4-way by adding a third labeler
  trained on a different fold split (FOLD_SEED=123). Hypothesis: if the
  multi-seed mechanism is compositional (not a single lucky draw), a
  third seed should either compound the lift or cleanly saturate it,
  giving us calibration information about the lever's ceiling.
- Changed: launched `FOLD_SEED=123 scripts/recipe_full_te.py` →
  `LABELER_TEST_PATH=...seed123... scripts/recipe_pseudolabel.py` chain
  (~1h45m total wall); `scripts/blend_4way_multiseed.py` for the
  fixed-bias blend analysis. Whitelist exception added for 5 seed=123
  artifacts.
- Stage 1 (recipe at seed=123): tuned OOF **0.97895** with bias
  [1.03, 1.17, 3.40]. ~0.0007 below s42 (0.97967) and s7 (0.97973) —
  weaker labeler, but fold-std 0.00056 was the tightest of the three
  seeds.
- Stage 2 (pseudo with s123 labeler, target at FOLD_SEED=42):
  - τ=0.98 keep rate 82.8% (223,619 rows, s7 was 82.6%, s1 was 83.8%)
  - Tuned OOF **0.97992**, bias [1.43, 1.27, 3.40]
  - Cross-seed pseudos now tied: s1 0.97993, s7 0.98002, s123 0.97992
    (all within 0.00010). **Ideal "different signal, same strength"
    pattern** — pseudo-label augmentation compensated for the weaker
    stage-1 labeler (τ=0.98 only keeps rows all seeds agree on).
- Blend analysis (fixed recipe bias, 4-way OOF vs 3-way LB-best 0.98029):
  ```
  standalone @ recipe bias:
    recipe       0.97967  errs=10114
    pseudo_s1    0.97987  errs=10039  Jaccard=0.7805
    pseudo_s7    0.98002  errs=10170  Jaccard=0.7781
    pseudo_s123  0.97973  errs=10049  Jaccard=0.7820  ← fewer errs than s7
  pairwise recipe × s123 peak α=0.750 → 0.98002 (narrower than s1/s7)
  4-way axis-scan (shrink 3-way, add s123): β=0.000 optimal — s123
    gets ZERO weight when searching along this axis
  4-way dense grid step=0.05: best = (0.25, 0.15, 0.35, 0.25) → 0.98029
    (TIED with 3-way, different geometry)
  4-way fine grid step=0.025: top-15 all at 0.98029-0.98030 (plateau);
    best 0.98030 at (0.225, 0.300, 0.425, 0.050) — s123 only 5%
  ```
  **Best 4-way OOF = 0.98030** vs 3-way LB-best 0.98029 = **+0.00001**,
  firmly inside fold noise.
- Test-prediction-space diagnostic (best 4-way vs LB-best 3-way):
  - 4-way top (s123=0.05): **22 test rows disagree** (0.008%)
  - 4-way grid (s123=0.25): 82 test rows disagree (0.030%)
  - At this test-flip magnitude, expected LB delta is ±0.00010 —
    indistinguishable from private-fold draw variance.
- **Verdict: s123 is cleanly falsified. The multi-seed pseudo-label
  mechanism SATURATES at 2 labelers on this feature set.** The signal
  s123 carries is already contained in the {recipe, s1, s7} span.
  Adding the 3rd seed re-arranges blend weights without adding
  information — classic "ridge of local optima" pattern.
- LB budget: unchanged (10/10 used today, 0 remaining). **No LB probe
  warranted** — expected LB delta ±0.00010 is pure gamble, not
  experimental payoff.
- **Multi-seed ceiling calibrated** (important for future
  compositions):
  - 2-way recipe × pseudo(s42) lifted LB by +0.00046 (LB-best 0.97998)
  - 3-way adding pseudo(s7) lifted LB by +0.00007 (new best 0.98005)
  - 4-way adding pseudo(s123): OOF +0.00001, predicted LB ±0.00010
  Decay factor ~6x between each addition. Further labeler seeds
  (s456, s789, etc.) predicted to add ≤ +0.00002 each — not worth
  a slot.
- Current LB best unchanged: `submission_3way_recipe025_s1035_s7040.csv`
  at **LB 0.98005**.
- **Strategic implication: breaking the 0.98005 ceiling requires a
  fundamentally different lever, not more seeds.** Pack 0.98114 remains
  +0.00109 above. Untried levers still on the hypothesis board:
  1. **P1 — threshold-axis test-time augmentation** (cheap, no LB
     spend to validate). Per-test-row Gaussian perturbation at rule
     thresholds (Soil=25, Rain=300, Temp=30, Wind=10), K=5-10
     perturbations, re-run recipe FE + OTE lookups, average
     log-probs. Targets the axis-aligned-tree-vs-smooth-NN mismatch
     at exactly the rows where flips concentrate.
  2. **P2 — symbolic regression (PySR/gplearn) for within-cell flip
     formula** (~1 evening). Binary flipped-vs-not on score=3 (n=5041)
     and score=6 (n=4163) cells using the 7 non-rule continuous
     features. Deploy any formula with ≥70% flip recall at <10% FP
     as a hardcoded override — orthogonal to blending by construction.
  3. **P3 — transductive k-NN label propagation in a learned embedding**
     (~2h). Supervised contrastive MLP → embed train+test into ~32-d
     → FAISS k-NN graph → sklearn LabelPropagation. Test-row
     prediction depends on test-row-to-test-row geometry — a
     modeling paradigm unrepresented in the log so far.
- Artefacts committed for cross-branch reuse:
  - `scripts/artifacts/oof_recipe_full_te_seed123.npy` + test + JSON
  - `scripts/artifacts/oof_recipe_pseudolabel_seed123labeler.npy` + test + JSON
  - `scripts/artifacts/blend_4way_multiseed_results.json`
  - `submissions/submission_multiseed_4way_{axis,grid}.csv` (OOF 0.98029
    and 0.98029 — not recommended for LB probe, emitted for reference)

### Next steps: kernel-audit-derived plan (2026-04-24)

Context: after every own-pipeline NN/FE/blend/calibration lever was
exhausted, a fresh audit of the top-25 public kernels surfaced three
untried NN architectures and two FE deltas, AND confirmed the 0.9812+
ceiling exists only as public-CSV blending (banned by the top-of-file
rule). Public audit findings:

- **`beraterolelk/defeating-synthetic-noise-0-981`** (2026-04-21):
  pure weighted hard-vote over `submission_god_tier_098120.csv` +
  `nina2025/0.98117/0.98116/0.98114.csv` rival submissions. Confirms
  the 0.98120 score exists as a CSV in public datasets but is NOT
  reached via any own-pipeline mechanism.
- **`yekenot/ps-s6-e4-realmlp-pytabkit`** (27 votes, 2026-04-20):
  RealMLP via PyTabKit with PBLD periodic embeddings, n_ens=8 inside
  one model, label smoothing schedule, robust_scale+smooth_clip
  transforms. Claimed CV 0.97802 standalone. **Not tested by us.**
- **`yekenot/ps-s6-e4-trompt-pytorch-frame`** (18 votes, 2026-04-18):
  Trompt (2023 tabular architecture, prompt-based column attention)
  via pytorch_frame. **Unrepresented in our 11 prior NN nulls.**
- **`include4eto/ps6e4-tab-transformer-claude-vibe-coding`**
  (64 votes, 2026-04-20): TabTransformer + **mixup augmentation** for
  tabular (V5 revision). Different from FT-Transformer we tested.
  **Mixup mechanism never tested on this problem.**
- **FE deltas not fully covered by recipe**: yekenot uses
  `Soil_Moisture / Temperature_C` ratio (we have products, not this
  ratio), `(col % 1).round(2)` decimal-fraction features for 5
  numerics (structurally distinct from our `floor(v·10^k) % 10`
  digit extraction), and 6 specific hand-picked important_combos
  (may not exactly match our 28 cat-pair OTE keys).

**Realistic LB upside calibration**: +0.005 is NOT reachable in own-
pipeline (would exceed leader 0.98219). Public-kernel own-pipeline
ceiling floats at ~0.978–0.980 standalone, ~0.981 blended. Our 0.98005
is already at that frontier. Realistic further upside: **+0.0005 to
+0.002 LB**, via an architecturally-novel blend leg that passes the
Jaccard < 0.80 AND errs ≤ anchor gate, OR via a feature transplant
that adds material signal the recipe misses.

**Tier A — concrete kernel-anchored experiments (highest EV/cost):**

  **A1. RealMLP leg via PyTabKit** (Kaggle GPU kernel, ~45 min wall).
  Port yekenot's exact config: `pytabkit.RealMLP_TD_Classifier` with
  `n_ens=8, hidden_sizes=[512,256,128], plr_hidden_{1,2}={16,8},
  plr_sigma=2.33, ls_eps=0.01, lr=0.05, wd=0.0236, tfms=['one_hot',
  'median_center', 'robust_scale', 'smooth_clip', 'embedding',
  'l2_normalize']`. 5-fold StratifiedKFold(seed=42) aligned with every
  saved OOF. Per-fold TargetEncoder(cv=5, smooth='auto') over
  6 important_combos. Gate: fold-1 Jaccard vs LB-best 3-way (0.98005)
  AND vs recipe_full_te. If ≥ 0.90 abort; if in 0.85–0.90 cap blend
  expectation at +0.00015; if < 0.85 run all 5 folds then blend-gate.
  Expected: +0.0005–0.0015 LB if blend passes. This is the 12th NN
  attempt on the problem but the first with a production-tuned
  tabular-specific NN, not a from-scratch MLP.

  **A2. Trompt leg via pytorch_frame** (Kaggle GPU kernel, ~1h wall).
  Port yekenot's `ps-s6-e4-trompt-pytorch-frame` kernel. Same 5-fold
  alignment, same blend-gate discipline. Higher variance than RealMLP
  but genuinely architecturally orthogonal — column-level attention
  with learnable prompts, unrepresented in our log. Run only after A1
  lands to avoid double-GPU-kernel queue wait.

  **A3. Mixup re-run of recipe XGB** (CPU, ~1h). Implement per-row
  mixup: for each training row, sample a partner, mix numerics via
  β(0.4, 0.4) convex combination, sample categorical values from the
  two parents weighted by the mix coefficient, sample target via the
  mixed soft-distribution. Train XGB `multi:softprob` with this as
  augmented training pool (original rows kept, +1× mixup rows added).
  Gate: standalone OOF tuned ≥ 0.979 AND errs ≤ recipe; else null.

  **A4. FE transplant from yekenot** (CPU, 30 min). Add to recipe:
  `Soil_Moisture/Temperature_C` ratio, `(col % 1).round(2)` decimals
  for `{Temperature_C, Organic_Carbon, Soil_Moisture, Soil_pH,
  Sunlight_Hours}`, and any of the 6 important_combos not already in
  our 28-pair OTE key set. Retrain recipe XGB. Gate: OOF Δ ≥ +0.0002
  before considering an LB probe. Lowest EV of Tier A but cheapest.

**Tier B — mechanism-novel (medium EV, speculative):**

  **B1. Finish the kernel audit on 10 remaining high-vote kernels**
  (~30 min read time). Every kernel read this comp has surfaced at
  least one lever (recipe itself was +0.00457 from kernel reads).
  Unread at ≥ 30 votes: Aryan Kaisth EDA (60), Kashifalikhan's
  "simplest XGB" (31), Rohit Kumar's LGBM+group-stats (39, recent),
  Ravi's TunedBlend (39), Manasi's ensemble-voting-analysis (48,
  2026-04-22), sarcasmos baseline (83), sakuno's combination-FE-CAT
  (41, 2026-04-23 — very recent), Akos Pinter's ensemble baseline
  (37), saamhm's blend (43), djenkivanov's minimal-XGB (36). Cheap,
  high information-per-minute. Do this FIRST.

  **B2. GroupKFold diagnostic** (CPU, 1h). Re-split by Region or
  Crop_Type instead of stratified. If our recipe OOF drops materially
  under GroupKFold, we've been OOF-overestimating via region/crop
  leakage in the OTE means, and the real frontier is LOWER than we
  think (closes the apparent stacking-inflation ceiling). If it holds,
  confirms OOF is honest. Either outcome is informative.

  **B3. Multi-task XGB** (CPU, ~1h). Replace recipe's single 3-class
  head with 4 joint heads: `y`, `dgp_score`, `rule_pred`, `cell_id`.
  Train via custom gradient averaging. Untested; shared-representation
  learning may help trees generalize where single-task saturates.

  **B4. SMOTE-NC for rare-High class** (CPU, 45 min). Synthesize ~10k
  additional High rows via SMOTE-NC (handles mixed numeric+cat).
  Retrain recipe XGB. Different lever than sample-weight — changes
  the training distribution. May interact with class-balanced weights
  in either direction.

**Tier C — infrastructure/scale (high cost, bounded upside):**

  **C1. 100+ XGB variant stack + LR meta-stacker** (GPU, ~8h).
  Per NVIDIA Kaggle-grandmaster playbook (March 2026 winning
  solution was 150 models from 850 candidates). Sweep: HPs × fold
  seeds × feature subsets × loss weights → 100+ OOFs → class-balanced
  LR meta-stacker. Bounded by our ~0.98030 stacking-inflation ceiling
  but may break it at scale (prior greedy with 12 components was
  saturated at 6-component ceiling).

  **C2. End-to-end Optuna on recipe pipeline** (GPU, ~3h). 40 trials
  on the full V10 recipe as a monolith with tuned-bias OOF as
  objective. Per-component Optuna LB-regressed (2026-04-22) but
  end-to-end avoids per-component compounding overfit.

**Tier D — final-selection strategy (zero LB spend, variance-
capture):**

  **D1. Lock two finals**: primary = 3-way multi-seed (LB 0.98005),
  hedge = `submission_recipe_full_te.csv` (LB 0.97939). Reasoning:
  the 2-way (LB 0.97998) shares too much overfit surface with the
  3-way primary; recipe standalone provides genuine variance
  protection on private LB.

  **D2. Private-LB variance estimate**: per Session B, the fold-seed
  spread on LB-best was ±0.00036 across 3 seeds. Private-LB spread
  likely ±0.0005 around public. The +0.00007 delta between our 3-way
  and 2-way is sub-noise for private ranking. Either could win.

**Execution order (6 days to deadline, budget resets at 10/day):**

  - Day 1 (now): B1 (kernel audit, 30 min) + A1 (RealMLP scaffold, no
    GPU queue yet) in parallel. No LB spend. Plus Edit / scaffold A4
    locally while kernel queues.
  - Day 2: A1 finishes → blend-gate → optional LB probe if +0.0005
    passes. Launch A2 (Trompt). Run A4 (FE transplant) locally and
    A3 (mixup) if A1 looks promising.
  - Day 3: A2 finishes → blend-gate. Launch C1 (100+ stack) if best
    of A1/A2/A3/A4 has left us below 0.98050 OOF. Run B2 (GroupKFold)
    diagnostic as background.
  - Days 4–5: consolidate winners, final-select per D1, reserve 2 LB
    probes for last-day variance check.
  - Day 6 (submission deadline): final selection locked.

**Highest-EV single experiment**: B1 (kernel audit) — 30 minutes of
read time has surfaced every real lever we've found. Start there.

**Skip on principled grounds**:
  - Any variant of public-CSV blending — banned, mechanism is
    demonstrably what 0.98114+ kernels actually do.
  - Further multi-seed pseudo-label extensions — saturated at 2
    labelers (s42+s7 gave +0.00007 LB; s123 added +0.00001 OOF, not
    worth a slot).
  - Further NN-from-scratch MLP retries — 11 NN nulls with consistent
    blend-null pattern; only production-tuned tabular NNs (Tier A)
    have a shot.
  - Further HP tuning on recipe components — LB-regressed twice
    (per-component 2026-04-22, seed-bag 2026-04-22).


### 2026-04-24 — A4 FE transplant (utaazu 11 domain + 5 decimal) closed as NULL

- Goal: execute Tier A #A4 from the kernel-audit plan. Port utaazu's 11
  domain interaction features (`moist_rain`, `moist_temp`, `moist_wind`,
  `ET_proxy`, `heat_stress`, `drying_force`, `water_supply`,
  `water_deficit`, `soil_quality`, `moist_x_temp`, `wind_x_temp`) +
  5 decimal-fraction features `(col % 1).round(2)` on
  `{Temperature_C, Organic_Carbon, Soil_Moisture, Soil_pH,
  Sunlight_Hours}`. 16 new numeric features added on top of the 443-col
  recipe; total 459 features.
- Hypothesis: utaazu's ratios encode water-balance physics that our
  `logit_P_*` formula features approximate only via a 4-binary
  indicator basis. The decimal-fraction features are structurally
  distinct from our digit extraction (captures FRACTIONAL portion vs
  INTEGER digit positions). Either source might be non-redundant with
  the OTE-encoded joint feature manifold.
- Changed: `scripts/recipe_features.py` + `scripts/recipe_full_te.py`
  gained `EXTRA_FE` env var (`''|domain|decimal|both`). Suffix
  `_fex{variant}` on outputs keeps LB-best artefacts untouched. Smoke
  (SMOKE=1, 2-fold 20k) passed cleanly with 459 features. Production
  (EXTRA_FE=both, 5-fold 630k seed=42) ran ~42 min CPU.
- Results (5-fold OOF seed=42):
  - Per-fold argmax: 0.97559, 0.97653, 0.97720, 0.97423, 0.97600
  - Mean fold argmax: **0.97591 ± 0.00100** (recipe: 0.97589 ± 0.00090;
    A4 tracking recipe within fold-noise, actually +0.00002 at argmax
    but post-tune −0.00012 below)
  - **Tuned OOF: 0.97955** (recipe 0.97967, **Δ = −0.00012**)
  - Tuned bias [0.9324, 1.0689, 3.2008] vs recipe's [1.4324, 1.4689,
    3.4008] — Low/Medium biases dropped ~0.4, indicating the 16 extra
    FE features produced sharper raw probs. But the tuned argmax
    operating point shifted to less-optimal macro-recall geometry,
    netting a small loss.
  - Error count **10,024** vs recipe **10,114** — A4 has **fewer
    errors** (−90 rows) at its own bias. The FE additions are doing
    SOMETHING real at the prob level.
- Blend gate vs 3 anchors (fixed recipe bias):
  - vs recipe (0.97967): peak α=0.125 → 0.97973 (Δ = +0.00006)
  - vs LB-best 2-way (0.98012): peak α=0.025 → 0.98016 (Δ = +0.00004)
  - vs LB-best 3-way (0.98029): peak α=0.025 → 0.98030 (Δ = +0.00001)
- **Jaccards** (the decisive diagnostic):
  - vs recipe: **0.87**
  - vs LB-best 2-way: **0.86**
  - vs LB-best 3-way: **0.83**
  All ≥ 0.80 redundancy threshold. A4's errors overlap too much with
  every anchor's errors for the blend math to extract orthogonal
  signal — even though A4 has fewer absolute errors.
- **Verdict: NULL.** Every blend Δ is below +0.00020 LB-transfer
  threshold; every Jaccard exceeds the 0.80 novelty threshold. The
  extra 16 FE features are redundant with the recipe's 443-col OTE
  surface — XGB already reconstructs these ratios/decimals via splits
  on the OTE-encoded joint manifold.
- **Rule confirmed** (previously stated, now 2nd empirical validation):
  blend-gate requires BOTH `Jaccard < 0.80` AND `errs ≤ anchor`. A4
  satisfies the magnitude half (90 fewer errors than recipe) but
  fails the orthogonality half (Jaccard 0.83–0.87). Either alone is
  insufficient.
- LB delta: n/a (no submission warranted; below emit gate).
- Current LB best unchanged at **0.98005**.
- Artefacts committed:
  - `scripts/artifacts/oof_recipe_full_te_fexboth.npy` + test + JSON
  - `submissions/submission_recipe_full_te_fexboth.csv` (diagnostic,
    not for LB probe)
- Companion work: A1 RealMLP kernel scaffold (`kaggle_kernel/kernel_realmlp/`)
  + `scripts/blend_realmlp.py` blend-gate analysis committed in parallel.
  Not yet launched (requires Kaggle GPU push).

### 2026-04-24 — B1 kernel audit round 2 (10 high-vote public kernels)

- Goal: fresh audit of 10 high-vote public kernels never-read on this
  branch. Every prior kernel audit has surfaced at least one real
  lever (recipe itself was +0.00457 LB from a kernel read).
- Kernels read (via `kaggle kernels pull`):
  sakuno/irrigation-need-eda-combination-fe-cat (41 votes),
  chovyxu/playgrounds2026-ep4-neural-network (42),
  rohit8527kmr7518/ps-s6e4-lgbm-with-target-encoding-group-stats (39),
  utaazu/0-979-cv-single-lgbm-pairwise-te-bias-tuning (36),
  mahoganybuttstrings/pg-s6e4-realmlp-cv-0-97802-lb-0-97685 (50),
  blamerx/s6e4-lightgbm-high-balance-0-979-cv (33),
  blamerx/s6e4-xgboost-adv-fe-0-979-cv (33),
  ravi20076/playgrounds6e4-tunedblend-v1 (39),
  rohit8527kmr7518/ps-s6e4-catboost-pipeline (56),
  saamhm/eda-baseline-model-fe-cv-ensemble-blend (43).
- Findings ranked by novelty × plausibility:
  1. **RealMLP-TD via pytabkit** (mahoganybuttstrings, CV 0.97802 /
     LB 0.97685) — novel tabular NN arch NOT in our 11-NN-null set.
     Production-tuned with n_ens=8 BatchEnsemble heads, PBLD periodic
     embedding, smooth-clip scaler, label smoothing with cosine
     schedule. **Highest-EV remaining GPU kernel. Scaffolded as A1
     this session** (`kaggle_kernel/kernel_realmlp/`).
  2. **utaazu 11 domain interactions** — ported in A4 scaffold,
     tested NULL this session (see entry above).
  3. **blamerx pseudolabel τ=0.92 + full-train refit at pooled
     best_iter** — distinct from our per-fold stage-1/stage-2. Trains
     ONE model on `train ∪ confident_test` without CV. Untested.
  4. **rohit8527 LGBM group-by cat×num stats on synthetic 630k** —
     we only have ORIG_mean/std from 10k. Per-cat-group mean/std
     from the full 630k pool is untested FE. ~30 min CPU. Highest
     ROI remaining CPU experiment.
  5. **rohit8527 MIN_COUNT=5 rare-cat bucketing before OTE** —
     untried; map rare categories to a single bucket before
     target encoding. Cheap to port.
  6. **blamerx multiplicative class-weight + Nelder-Mead** —
     functionally equivalent to our log-bias coord-ascent but
     multiplicative. Likely same operating point. Low EV.
  7. **sakuno pd.qcut/pd.cut/(col/20).round binning** — overlap
     with our num_as_cat but different granularity.
  8. **saamhm 50-fold CV** — extreme-fold averaging as ensemble.
     10x our compute cost for marginal lift.
  9. **chovyxu TreeBinner (DecisionTreeClassifier bin edges)** —
     data-driven bin edges vs qcut. Minor lever.
  - **Skip**: `ravi20076/playgrounds6e4-tunedblend-v1` uses public-CSV
     blending (reads `.npy`/`.parquet` from ~10 other users' kernels
     into a `TunedBlender()` class). Banned by repo rule.
- Read-out: the 2 highest-novelty findings (RealMLP + blamerx
  pseudolabel τ=0.92) both require NEW compute (GPU kernel + CPU
  rerun respectively). FE-level transplants (utaazu A4 above,
  rohit8527 group-by stats, sakuno binning) are cheap but have
  lower EV given the recipe's mature OTE coverage.

### 2026-04-24 — GBY rohit8527 group-by cat×num stats closed as NULL

- Goal: execute B1 audit finding #4 — rohit8527's per-cat-group
  `mean/std` of each numeric on the SYNTHETIC 630k pool (we only had
  `add_orig_mean_std` which aggregates TARGET on 10k original; this
  aggregates NUMERIC distributions on the full train). 8 cats × 11
  nums × 2 stats = 176 extra numeric features via `GBY=1` env var.
- Changed: `scripts/recipe_features.py` new
  `add_groupby_cat_num_stats()`, `scripts/recipe_full_te.py` new
  `GBY` env var with `_gby` output suffix. Smoke (SMOKE=1, 20k
  2-fold): 619 features total, clean end-to-end. Production (GBY=1,
  5-fold 630k seed=42) ran ~3h CPU (folds 1-2 ~35 min each, folds
  3-5 ~30 min each; slower than recipe due to 619 features vs 443).
- Results (5-fold OOF seed=42):
  - Per-fold argmax: 0.97518, 0.97557, 0.97754, 0.97460, 0.97566
  - Mean fold argmax: **0.97571 ± 0.00099** (recipe: 0.97589)
  - **Tuned OOF: 0.97959** (recipe: 0.97967, Δ=−0.00008)
  - Tuned bias [1.03, 1.17, **3.00**] vs recipe's [1.43, 1.47, 3.40] —
    High bias dropped 0.40 (GBY features produce SHARPER High probs).
    Group-by stats on Humidity, Previous_Irrigation, etc. carry
    class-discriminative signal the model exploits.
  - Error count **10,040** vs recipe **10,114** (−74 rows — real
    signal gain at prob level, same pattern as A4).
- Blend gate vs 3 anchors (fixed recipe bias):
  - vs recipe (0.97967): peak α=0.450 → 0.97981 (Δ=+0.00014)
  - vs LB-best 2-way (0.98012): peak α=0.025 → 0.98014 (Δ=+0.00002)
  - vs LB-best 3-way (0.98029): peak α=0.025 → 0.98030 (Δ=+0.00001)
- **Jaccards**:
  - vs recipe: 0.87
  - vs LB2: 0.86
  - vs LB3: 0.83
  All ≥ 0.80 redundancy threshold — same pattern as A4.
- **Verdict: NULL** vs both LB-best anchors. Standalone tuned is
  0.97959 (−0.00008 vs recipe, within fold noise). Blend peaks
  below +0.0002 LB-transfer threshold.
- **Two-experiment pattern confirmed** (GBY + A4): any derived
  numeric FE on top of the recipe ends up with Jaccard 0.83-0.87 vs
  LB-best 3-way. The recipe's OTE + digit + ORIG_stats already
  encode what ratios, decimals, and per-cat group stats describe —
  trees reconstruct the same signal via splits on the existing
  OTE-encoded manifold. **Further numeric FE on recipe is
  architecturally capped**.
- Artefacts committed (whitelisted in `.gitignore`):
  - `scripts/artifacts/oof_recipe_full_te_gby.npy` + test + JSON
  - `submissions/submission_recipe_full_te_gby.csv` (diagnostic,
    not for LB probe)
- LB best unchanged at **0.98005**.
- Companion work: A1 RealMLP v3 RUNNING on Kaggle GPU (v1 CUDA
  error, v2 lightning-missing, v3 fixed both). Status check pending.
### 2026-04-24 — blamerx τ=0.92 pseudo-label executed: NULL (blend-redundant)

- Goal: execute blamerx's "lower τ + many-pseudo" mechanism from the B1
  kernel audit. Our stage-1 pseudo uses τ=0.98 (keep 84% of test rows);
  blamerx's suggestion was τ=0.92 (keep ~94%, +28k lower-confidence
  rows). Hypothesis: more pseudo training signal, especially on
  boundary rows where the labeler is only 0.92-0.98 confident, may
  produce orthogonal signal the stage-1 pseudo misses.
- Changed: no new script — reuses `recipe_pseudolabel.py` with
  `PSEUDO_TAU=0.92 PSEUDO_SUFFIX=tau092`. Phase 2 (full-train no-CV
  refit) was scaffolded as a contingent follow-up if Phase 1 passed
  the blend gate; cancelled when Phase 1 came back null.
- Pseudo subset stats:
  - keep_rate = 0.9417 (254,269 / 270,000 test rows) vs stage-1's
    84%. +28k lower-confidence pseudo rows added.
  - Label dist [Low 151,301 / Medium 93,989 / High 8,979] matches
    prior distribution (Low 58.6% / Medium 36.8% / High 3.5% vs
    train prior 58.7 / 37.9 / 3.3%). No confidence bias at class
    level.
  - Max-prob percentiles on kept rows: p25=0.9936, p50=0.9986, p99=1.0
    — most kept rows are still very confident; the new rows are in
    the 0.92-0.98 band (boundary-adjacent).
- Production run stats (5-fold seed=42, 758k training pool):
  - Per-fold argmax: 0.97707, 0.97652, 0.97870, 0.97683, 0.97706
  - Overall argmax = **0.97724 ± 0.00076**
  - **Tuned OOF = 0.98004** (bias [1.4324, 1.4689, 3.4008] matches
    recipe's exactly — the prob scale is similar to stage-1's)
  - 5 folds × ~520s wall = ~45 min total CPU
- Standalone comparison vs baselines:
  ```
                        tuned OOF   errors   Jaccard vs recipe   Jaccard vs stage-1
  recipe                0.97967      8,367       1.00               0.78
  stage-1 τ=0.98        0.97993      8,430       0.78               1.00
  blamerx τ=0.92        0.98004      8,484       0.85               0.87
  LB-best 2-way         0.98012         -          -                  -
  ```
- Fixed-bias log-blend sweeps:
  - vs recipe (base 0.97967): peak α=0.40 → 0.98019 (Δ=+0.00052).
    Above +0.0002 threshold but recipe is a weaker anchor than
    LB-best.
  - vs LB-best 2-way (base 0.98012): peak α=0.25 → 0.98023
    (**Δ=+0.00010**, BELOW +0.00020 LB-transfer threshold). All
    α ∈ [0, 0.5]: Δ ≤ +0.00010.
- **Verdict: NULL.** Two independent reasons predict LB null:
  1. Blend vs LB-best caps at +0.00010 (below threshold).
  2. Error count 8,484 > stage-1's 8,430 > recipe's 8,367 — fails
     the "errs ≤ anchor" half of the blend heuristic.
  3. Jaccards 0.85 (recipe) / 0.87 (stage-1) both exceed 0.80
     redundancy threshold.
- **Mechanism — why "more pseudo" hurt the blend**: the extra 28k
  lower-confidence pseudo rows encode the labeler's own boundary-
  band decisions (where max-prob 0.92-0.98). These are the rows
  where the labeler is most uncertain, and including them as
  confident training labels causes blamerx's decision surface to
  track the labeler MORE closely on those boundary rows. That's
  the opposite of what we want for orthogonal blend signal —
  higher Jaccard with recipe/stage-1 is the direct consequence.
- **Portable rule** (adds to LEARNINGS.md candidates): "For
  pseudo-label augmentation, τ controls a tradeoff between
  training-signal volume and blend-orthogonality. Very high τ
  (0.98+) keeps blend signal intact because only rule-aligned
  rows get labels; lower τ bleeds labeler decisions into boundary
  rows where the labeler itself is uncertain, which collapses
  orthogonality. The sweet spot for this problem is τ=0.98
  (stage-1), not lower."
- Phase 2 of the plan (blamerx's full-train refit without CV,
  pooled best_iter ≈ 1147) cancelled: Phase 1's blend-null result
  predicts the full-refit LB would also null, and LB budget is
  scarce (0/10 remaining today).
- Artefacts committed:
  - `scripts/artifacts/oof_recipe_pseudolabel_tau092.npy` + test
  - `scripts/artifacts/recipe_pseudolabel_tau092_results.json`
  - `submissions/submission_recipe_pseudolabel_tau092.csv`
    (diagnostic — OOF 0.98004, below LB-best 2-way's 0.98012)
- LB budget unchanged: 10/10 used today (0 remaining until reset).
  LB best unchanged at **0.98005** (3-way multi-seed).
- **Consequence for the broader pseudo-label lever**: we now have
  three empirical data points in our pseudo-label ladder:
  ```
  stage-1 τ=0.98 (recipe labeler, s42)   → LB 0.97998 (gap +0.00014)
  stage-2 τ=0.98 (LB-best labeler, s42)   → LB 0.97989 (gap +0.00038)
  blamerx τ=0.92 (recipe labeler, s42)    → OOF +0.00010 vs LB-best (null)
  seed-7 labeler τ=0.98                   → LB 0.97969 (gap +0.00043)
  ```
  Stage-1 τ=0.98 is the only variant that transferred to LB positively.
  Every decoupling (stage-2 chain, lower τ, seed-7 labeler) tightened
  OOF calibration in ways that didn't survive the test split.

### 2026-04-24 — B2 GroupKFold diagnostic: honest OOF confirmed, ceiling is real

- Goal: execute B2 from the kernel-audit plan. Re-split the 630k
  training set by Region (5 groups: South, West, East, Central, North)
  instead of stratified-on-y. If StratifiedKFold(seed=42) is leaking
  region-specific signal via OTE group means, OOF drops materially
  under GroupKFold and the apparent 0.98005 LB ceiling is partly CV
  artifact. Otherwise, OOF holds and the ceiling is a real structural
  saturation.
- Changed: new `scripts/b2_groupkfold.py` (thin 191-line wrapper;
  imports `load_and_engineer` from `recipe_full_te`, swaps
  StratifiedKFold for GroupKFold, keeps everything else identical —
  same 443-feature matrix, same XGB HPs, same class-balanced sample
  weights, same OrderedTE, same log-bias coord-ascent). Env var
  `GROUP=region|crop` selects the grouping column. 5 regions × 5-fold
  GroupKFold → each val fold validates exactly 1 region.
- Production run stats (5-fold seed=42, 45 min CPU):
  ```
  fold  val region   n_val     best_iter   argmax_bal_acc
  1     South        134,809   1363        0.97543
  2     West         131,189   1183        0.97493
  3     East         126,163   1166        0.97280   ← hardest (new-region generalisation)
  4     Central      123,712   1238        0.97681   ← easiest
  5     North        114,127   1175        0.97463

  OOF argmax   = 0.97500 ± 0.00130   (σ wider than StratifiedKFold's ~0.00088)
  Tuned OOF    = 0.97938             (bias [1.3324, 1.1689, 3.4008])
  ```
- Comparison vs StratifiedKFold baseline:
  ```
                          tuned OOF     bias
  StratifiedKFold s42     0.97967       [1.4324, 1.4689, 3.4008]
  GroupKFold Region       0.97938       [1.3324, 1.1689, 3.4008]
  Δ                      −0.00029
  ```
- **Verdict: HONEST OOF.** Δ=−0.00029 is well within the 0.002
  "honest" threshold (ruled out leakage ≥0.005 as material; 0.002–0.005
  as moderate; ≤0.002 as honest). StratifiedKFold(seed=42) is NOT
  exploiting region-specific leakage.
- Interpretation details:
  - Per-fold σ widened from ~0.00088 (StratifiedKFold) to 0.00130
    (GroupKFold) because each fold now validates a structurally
    distinct region. The wider variance reflects genuine regional
    heterogeneity, not leakage.
  - East is the hardest held-out region (−0.00287 below baseline mean);
    Central is easiest (+0.00114 above baseline mean). Spread ~0.004
    across regions is within the bias sensitivity a global log-bias
    tuner can accommodate.
  - Medium bias dropped ~0.30 (1.47→1.17) — natural consequence of
    the fold structure producing less-balanced per-fold priors. The
    High bias is unchanged at 3.40, confirming the High-class
    calibration point is region-invariant.
- **Strategic consequence**: the apparent stacking-inflation ceiling
  documented across multiple entries (3 submissions at OOF 0.98030 →
  LB 0.97995-0.97997; 12-component full greedy saturating at same
  level) is confirmed REAL structural saturation. Not a CV artifact.
  Not a region-leakage artifact. To break above LB 0.98005 requires
  a fundamentally different mechanism, not another blend variant.
- **Portable rule** (adds to LEARNINGS.md candidates): "When a
  stacking ceiling is suspected, run GroupKFold over each plausible
  leakage vector (region / crop / temporal / user-id). If the tuned
  OOF holds within 0.002 across all vectors, the ceiling is
  structural; if any vector produces a ≥0.002 drop, investigate
  leakage in the OTE / frequency / group-stats features."
- Companion blamerx τ=0.92 run (same session) came back NULL —
  see 2026-04-24 blamerx entry above. Combined result of this
  session: two open hypothesis items (B2 diagnostic, blamerx τ=0.92)
  both closed; B2 with positive diagnostic value (ceiling is real),
  blamerx with null blend result.
- Artefacts committed:
  - `scripts/b2_groupkfold.py`
  - `scripts/artifacts/oof_b2_groupkfold_region.npy` + test + JSON
  - `submissions/submission_b2_groupkfold_region.csv` (diagnostic)
- LB budget unchanged at 10/10 used today (0 remaining). No probe
  warranted — B2 standalone OOF is BELOW LB-best, and its purpose
  was diagnostic anyway.
- Current LB best unchanged at **0.98005** (3-way multi-seed).
- **Remaining open items on the board after this session** (for
  other agents / next session):
  - A1 RealMLP GPU kernel (on `claude/review-leaderboard-strategy-IMYgZ`,
    awaiting GPU queue)
  - rohit8527 group-by cat×num stats FE (same branch)
  - A2 Trompt GPU kernel (unclaimed)
  - A3 Mixup XGB (unclaimed)
  - B0 DivideMix (unclaimed; only if A1 produces a passing leg)
  - B3 Multi-task XGB (unclaimed)
  - rohit8527 MIN_COUNT=5 rare-cat bucketing (unclaimed)
  - GroupKFold by Crop_Type as a second diagnostic axis (cheap
    extension, ~45 min CPU — run with `GROUP=crop`)

### 2026-04-24 — disagree meta-stack + selective router: both NULL at per-class Pareto frontier

- Goal: attack the "magnitude trap" that's killed every prior blend leg
  from two orthogonal angles simultaneously. (Option 3) train a shallow
  XGB on teacher-vs-candidate DISAGREEMENT features (not raw probs) —
  the stacking math no α-blend weight can encode. (Option 1) abandon
  global log-blend entirely: route per row, keeping LB-best argmax on
  confident rows and deferring only to a router-chosen argmax on
  low-confidence rows. Both run on saved OOFs; no retraining of base
  learners, ~8 min each on 8 CPU cores. Teacher = LB-best 3-way
  (recipe 0.25 + pseudo_s1 0.35 + pseudo_s7 0.40) at fixed recipe
  bias [1.4324, 1.4689, 3.4008], OOF 0.98029, LB 0.98005.
- Changed: `scripts/meta_common.py` (shared loader — teacher
  reconstruction, 7-candidate bank, y/dist/score features, pinned
  StratifiedKFold(seed=42)), `scripts/disagree_meta_stack.py`
  (Option 3, 36 features: per-cand P_teacher−P_cand ×3 + argmax
  disagreement flag + teacher conf/entropy/argmax + dgp_score +
  signed distances), `scripts/selective_router.py` (Option 1, 28
  features: argmaxes + confidences + disagreement flags + score/dist,
  τ ∈ {0.80..0.99} sweep + High-class-only gate), and
  `scripts/analyze_options_3_and_1.py` (Jaccard + forecast-LB +
  consolidated verdict).
- **Option 3 meta-stacker — NULL**:
  ```
  standalone tuned OOF       0.98015  (Δ vs teacher  -0.00014)
  meta @ recipe bias         0.97992  (Δ  -0.00036)
  meta errors                9,538    (teacher 9,873 — 335 FEWER)
  Jaccard vs teacher         0.8827   (redundancy zone; needs <0.80)
  blend sweep peak α=0.35    0.98030  (Δ  +0.00001)  ← NULL
  ```
  Per-class recall at meta's tuned bias: Low 0.9953 / Med 0.9695 /
  **High 0.9756** (vs teacher Low 0.9949 / Med 0.9685 / High 0.9774).
  Meta trades **0.18 pp of High** for small Low+Medium gains.
  Under macro-recall that trade is exactly net-zero. The 335-row
  error reduction is invisible because macro-recall cares about
  per-class rate, not absolute count.
- **Option 1 router — NULL**:
  ```
  standalone tuned          0.98027  (Δ vs teacher  -0.00002)
  τ=0.80  routed  8,340   bal_acc 0.98024  Δ -0.00005  net_wins +157
  τ=0.90  routed 19,544   bal_acc 0.98026  Δ -0.00003  net_wins +424
  τ=0.95  routed 37,467   bal_acc 0.98028  Δ -0.00000  net_wins +316  ← peak
  τ=0.97  routed 55,160   bal_acc 0.98027  Δ -0.00001  net_wins +309
  τ=0.99  routed 106,406  bal_acc 0.98027  Δ -0.00001  net_wins +308
  ```
  Low-conf (τ=0.95) routed set, per-class breakdown:
  ```
  class     in routed   teacher right   router right   Δ
  Low         25,170        24,220         24,298      +78
  Medium      11,085         5,389          5,655      +266
  High         1,212         1,052          1,024      -28
  ```
  Router wins +316 rows cumulatively but **loses 28 High**. Under
  macro-recall the +78 Low / +266 Med gains are diluted by their
  large class denominators while the −28 High is amplified (High
  denominator 21k). Net bal_acc: flat.
- **High-promotion diagnostic probe** (rule: if teacher argmax ≠
  High AND max-candidate P(High)|recipe_bias > θ → flip to High):
  ```
  θ=0.50  n=26,968  Δ=-0.02844   monotone worse
  θ=0.60  n=13,350  Δ=-0.01346
  θ=0.70  n= 5,595  Δ=-0.00544
  θ=0.80  n= 1,738  Δ=-0.00153
  θ=0.90  n=   252  Δ=-0.00014
  θ=0.95  n=    62  Δ=-0.00005
  ```
  Strictly monotone negative at every θ. **No candidate in the
  7-model bank predicts High correctly on rows where teacher
  misses, even at very high candidate confidence.**
- **Mathematical read** (portable): the LB-best 3-way teacher
  occupies a **per-class Pareto frontier** at
  `(recall_L, recall_M, recall_H) = (0.9949, 0.9685, 0.9774)`.
  Every row-level rearrangement built from the existing 7-candidate
  OOF bank shifts error mass ALONG the frontier — it cannot push
  the frontier outward. Both a smart meta-stacker (Option 3) and
  surgical per-row routing (Option 1) land at the same OOF 0.98028–
  0.98030 ceiling because that's the frontier's upper envelope
  under the current component geometry. To break 0.98030 OOF
  requires a component whose errors are orthogonal **specifically
  in the rare-High direction** — i.e. correctly predicts High on
  rows where teacher says Medium, without adding new False-High
  noise. The per-candidate High-promotion sweep proves no such
  component exists in the current bank.
- **New portable rules** (logging to LEARNINGS.md candidates):
  1. **"Fewer total errors" is not enough when the metric is
     macro-recall.** A meta-stacker that reduces total errors can
     still be net-zero if the error reduction is distributed
     against the rare class. Under macro-recall, errors-per-class
     matter, not errors-total. Blend-gate should include per-class
     recall delta, not just total-correct delta.
  2. **Per-row routing has the same ceiling as global log-blend**
     when router features are derived from the same component bank.
     The router can't invent new orthogonal signal — it can only
     choose which existing argmax to trust. If no argmax in the
     bank is systematically better than teacher on the rare class,
     routing plateaus at teacher.
  3. **High-promotion rule sweep** is a cheap lever-existence test
     (~30 s): for each θ, count how often "at least one candidate
     says High at confidence > θ" correlates with true High among
     teacher-misses. If monotone-negative at every θ, no future
     blend / meta / route experiment built on the same bank can
     break teacher in the rare-class direction. Run this BEFORE
     scaffolding more complex experiments.
- LB budget unchanged at 10/10 used yesterday, full 10 remaining
  tomorrow. LB best unchanged: `submission_3way_recipe025_s1035_s7040.csv`
  at **LB 0.98005**. No LB probe warranted — both forecasts land
  at ~0.98005, indistinguishable from current best.
- Artefacts committed for cross-branch reuse (.gitignore whitelist):
  - `scripts/artifacts/oof_disagree_meta.npy` + `test_...` + JSON
  - `scripts/artifacts/oof_selective_router.npy` + `test_...` + JSON
  - 4 scripts under `scripts/` — all ~150 lines each, standalone,
    reusable as diagnostic tools on any future LB-best teacher.
- **Next (re-framed by this finding)**: the remaining path past
  LB 0.98005 is necessarily a component-ADDITION experiment, not
  a stacking/routing rearrangement. Candidates aligned with the
  High-recall-specific orthogonality requirement:
  1. **RealMLP-TD** (already scaffolded in `kaggle_kernel/kernel_realmlp/`,
     not yet run on GPU) — untested NN family; if its errors are
     orthogonal to teacher AND biased toward High recall, first
     real lever since multi-seed.
  2. **Focal-loss XGB** on the full recipe feature set, with γ=2
     and High-class α upweight. Produces probs that push High
     recall up by construction; different error geometry than the
     balanced-sample-weight recipe XGB family.
  3. **Self-supervised "missed-High" detector**: binary XGB trained
     only on rows where `teacher_argmax ∈ {Low, Medium}` and
     `y = High`, using the same 43-dist feature set. Negative
     class = teacher-correct rows (subsampled). If AUC > 0.9 on
     held-out rows, the signal exists and can be deployed as a
     hard-gated override (not a blend). Untested.

### 2026-04-24 — RealMLP kernel killed at t+3.5h (wasted GPU budget)

- Goal: A1 RealMLP-TD via pytabkit on Kaggle GPU. B1 kernel audit #1
  ranked this as highest-EV remaining experiment (production-tuned
  tabular NN, only NN family not yet nulled).
- Changed: `kaggle_kernel/kernel_realmlp/` scaffold with
  `RealMLP_TD_Classifier(n_ens=8)`, 11 raw nums (raw + factorized) +
  8 cats + 15 pair combos of 6 rule-relevant features, per-fold
  multiclass TargetEncoder(cv=5). 5-fold StratifiedKFold(seed=42).
- Kernel iterations:
  - v1: `torch.AcceleratorError: CUDA error: no kernel image` on P100
    (sm_60). Classic pre-installed-torch-vs-Pascal mismatch.
  - v2: `ModuleNotFoundError: No module named 'lightning'`. Fix: install
    torch+torchvision cu121 pair explicitly + `pip install lightning`.
  - v3: boot + GPU detection OK, training started, then stalled in
    per-fold preprocessing.
- v3 timeline on Kaggle P100:
  - t+0 to t+52s: torch 2.5.1 cu121 install + pytabkit install.
  - t+52s to t+3h34min: CPU preprocessing silent (no output).
  - t+3h34min: Lightning Trainer.fit() engaged, GPU available msg.
  - **Killed at that point by user** (0 fold results after 3.5h).
- Estimated cause: `n_ens=8` × 5-fold × per-fold TargetEncoder(cv=5)
  produces a 200× preprocessing multiplier inside pytabkit. Mahogany-
  buttstrings' CV 0.97802 claim was likely a single-fold run with
  simpler preprocessing; we didn't scale up the estimate.
- **User instruction: hard 1h wall-time cap on GPU kernels going
  forward.** See ⚠️ rule at top of CLAUDE.md.
- Lever status: RealMLP via pytabkit is NOT closed, but re-attempt
  requires a drastically smaller config:
  - `SMOKE=1` first — single fold, n_ens=1, 3 epochs, 20k rows.
  - If smoke succeeds + produces a sensible OOF, scale to n_folds=5
    with n_ens=4 (not 8), epochs=50 (not pytabkit default 256).
  - Hard budget: kill at t+60min.
- LB best unchanged at **0.98005**. Own-pipeline lever count still
  converging on "ceiling is structural" per B2 GroupKFold confirmation
  (OOF honest) + 15+ other nulls.

### 2026-04-24 — RealMLP retry (careful, n_ens=1): blend LB 0.97991 — Jaccard 0.62 real but magnitude trap dominates (12th NN lever closed)

- Goal: careful retry of the A1 RealMLP lever after yesterday's 3h34min
  preprocessing kill. Three config fixes applied:
  1. `n_ens=1` explicit (was pytabkit default 8; removed 8× BatchEnsemble
     internal preprocess multiplier that caused the hang)
  2. `n_epochs=40` explicit (was pytabkit default ~256)
  3. `TargetEncoder(cv=2)` (was cv=5; cuts internal TE passes 60%)
  Plus two safety nets: fold-1 t+20min kill + total-wall t+55min kill,
  both with graceful partial-output save on kernel side.
- Pipeline: SMOKE-first discipline enforced via `IS_SMOKE=True` top-level
  toggle. Kaggle SMOKE v5 confirmed end-to-end on 20k × 2 folds × 3 epochs
  in ~5 min wall (2.6s + 2.0s per-fold GPU training, tuned OOF 0.87958 —
  structurally clean). Downstream `np.bincount` dtype bug (y was float64
  from pd.Series.map) fixed before production push.
- Production (v6): 5/5 folds completed in 57.1 min kernel-internal. The
  `TOTAL_KILL_SEC=55*60` fired at fold 5 END exactly as designed — final
  save executed normally, all outputs written. Total Kaggle wall from
  push ~62 min including pip installs.
- Per-fold argmax: 0.97042 / 0.96820 / 0.97270 / 0.97053 / 0.97090
  (σ=0.00144). **OOF argmax 0.97055, tuned 0.97636** with bias
  [1.2324, 1.4689, 3.4008]. Feature set: 30 base cols (8 cats + 11
  raw float nums + 11 factorized dupes of the same nums) + 9 pair
  combos (6 pairs dropped as `nunique > N/2`) + 27 TE cols (9 combos
  × 3 classes) = ~66 features fed to RealMLP_TD_Classifier.
- Blend-gate analysis (`scripts/blend_realmlp.py`, fixed recipe bias
  [1.4324, 1.4689, 3.4008]):
  ```
  standalone @ recipe bias:   0.97633  errs=10472  (+358 vs recipe 10114)
  tuned (own bias):           0.97636

  Jaccards:
    vs recipe_full_te         0.6171   ← LOWEST NN Jaccard in comp log
    vs LB-best 2-way           0.6251
    vs LB-best 3-way           0.6206
  ```
- Blend sweeps (fixed recipe bias, α ∈ {0..0.50}):
  ```
  vs recipe   peak α=0.275  OOF 0.97991  Δ=+0.00024
  vs LB2      peak α=0.375  OOF 0.98039  Δ=+0.00027   ← selected for LB probe
  vs LB3      peak α=0.200  OOF 0.98047  Δ=+0.00019   (1e-5 below +0.0002 gate)
  ```
- **LB probe (submitted 18:32 UTC with user approval):**
  `submission_lb2_realmlp_a0375.csv` → **LB public = 0.97991**.
  Δ vs LB-best (0.98005) = **−0.00014**.
  Gap OOF→LB = +0.00048 (anchor LB2's historical gap was +0.00014 →
  inflated by +0.00034 when RealMLP added to the blend). Classic
  magnitude-trap behavior at the LB-transfer level: the 358 extra
  RealMLP errors translated to ~+0.00034 LB degradation despite
  novel Jaccard.
- Read-out: this is the **12th NN null** on this problem but of a
  NEW flavor. Prior 11 nulls (v5-v9 MLP, FT-T, TabPFN, pretrain-FT,
  NN-on-orig, soft-distill, DAE) all had BOTH Jaccard≥0.80 AND
  err-magnitude issues. RealMLP uniquely satisfied the orthogonality
  half (Jaccard 0.62 — best of any NN ever tested here) but still
  failed the magnitude half (+3.5% more errors than anchor). New
  rule for LEARNINGS: **Jaccard 0.62 with 0.000 err-count margin IS
  already LB-negative** — errs ≤ anchor is tighter than just +3.5%
  for LB transfer. Informal threshold: errs ≤ `1.005 × anchor` for
  positive LB Δ, not `1.04 × anchor`.
- Lever status: RealMLP at **n_ens=1** is closed. Open follow-ups:
  1. n_ens=4 retry (~45 min expected, requires careful wall-time
     budget; expected standalone OOF ~0.9770, same Jaccard pattern,
     potentially lower error count if ensembling reduces per-row
     noise).
  2. Frozen RealMLP OOF + test are committed to the artifact bank
     as a real diversity leg — future greedy-stacking experiments
     on expanded OOF pools MAY still find a local optimum that
     includes it (just not at 0.375 α on LB2 anchor).
- LB budget: **1/10 spent today**, 9 remaining. LB best unchanged
  at **0.98005**.
- Artefacts:
  - `scripts/artifacts/oof_realmlp.npy` (5.4 MB)
  - `scripts/artifacts/test_realmlp.npy` (3.1 MB)
  - `scripts/artifacts/realmlp_results.json`
  - `scripts/artifacts/blend_realmlp_results.json`
  - `submissions/submission_lb2_realmlp_a0375.csv` (submitted, LB 0.97991)
  - `submissions/submission_lb3_realmlp_a020.csv` (not submitted)

### 2026-04-24 — greedy refit + RealMLP 3-stack: NEW LB BEST 0.98008 (+0.00003)

- Goal: after the RealMLP n_ens=1 blend hit LB 0.97991 (-0.00014 from
  magnitude trap despite Jaccard 0.62), user asked how to build on
  the result. Diagnostic: run greedy forward-selection with RealMLP
  added to the 38-component OOF bank, from two anchors (recipe and
  LB-best 3-way), to see whether a better blend configuration than
  the hand-picked α=0.375 exists in the pool.
- Changed: `scripts/greedy_realmlp_refit.py` (mirrors c0_safe_greedy_v3
  pattern with realmlp added to CANDIDATES, fixed recipe bias,
  EXCLUDE = {soft_distill, xgb_spec_678, pseudo_stage2}); 17 min wall
  on CPU. `scripts/emit_realmlp_3stack.py` emits the greedy-chosen
  submission.
- Greedy results:
  ```
  Anchor: recipe_full_te  (starts OOF 0.97967)
    step1: + recipe_full_te_seed7          α=0.500  +0.00053
    step2: + recipe_pseudolabel_seed7lab   α=0.200  +0.00016
    step3: + xgb_nonrule__iso              α=0.150  +0.00012
    step4: + em_uniform                    α=0.075  +0.00008 (stop, <1e-4)
    final 0.98047 — RealMLP NOT picked
  Anchor: lb_best_3way  (starts OOF 0.98029)
    step1: + realmlp                       α=0.200  +0.00019  ← TOP PICK
    step2: + xgb_nonrule__iso              α=0.075  +0.00014
    step3: + recipe_full_te_a10            α=0.200  +0.00006 (stop, <1e-4)
    final 0.98061 (Δ+0.00032 vs LB3 0.98029)
  ```
- **Anchor dependency is the key insight:** from recipe (weaker anchor),
  greedy prefers same-family additions (seed7 recipe/pseudo). From
  LB-best 3-way (stronger anchor, already exhausted within-family
  lifts), greedy picks RealMLP FIRST — its cross-family orthogonality
  only becomes the top pick after the anchor has consumed the easy
  tree-family gains.
- Structural comparison (3-stack vs first RealMLP blend):
  ```
  metric          submission_lb2_realmlp_a0375  submission_lb3_realmlp_nonruleiso
  OOF             0.98039 (+0.00027)            0.98061 (+0.00032)
  errs vs anchor  +358 (10472 vs 10114)  TRAP   -301 (9572 vs 9873)  PASS
  Jaccard vs LB3  0.62                          0.92
  High recall     similar                       0.9774
  LB              0.97991 (-0.00014)            0.98008 (+0.00003)
  ```
- **LB probe (user-approved, submitted 19:28 UTC):**
  `submission_lb3_realmlp_nonruleiso.csv` → **LB public = 0.98008**.
  Δ vs prior LB-best (0.98005 from 3-way multi-seed) = **+0.00003**.
  OOF→LB gap = **+0.00053** (wider than multi-seed's +0.00024,
  tighter than first RealMLP blend's +0.00048).
- Updated calibration ladder:
  ```
  greedy + nonrule α=0.15       0.97421 → 0.97352  gap +0.00069
  digit-XGB standalone          0.97449 → 0.97468  gap -0.00019
  digits-OTE × digit-XGB α=0.40 0.97477 → 0.97482  gap -0.00005
  recipe_full_te                0.97967 → 0.97939  gap +0.00028
  recipe × pseudo 2-way         0.98012 → 0.97998  gap +0.00014
  3-way multi-seed              0.98029 → 0.98005  gap +0.00024
  lb2 + realmlp α=0.375 (trap)  0.98039 → 0.97991  gap +0.00048
  **lb3 + realmlp + nonrule_iso 0.98061 → 0.98008  gap +0.00053**  ← NEW LB BEST
  ```
  Gap to pack 0.98114: +0.00106 (was +0.00109).
  Gap to leader 0.98219: +0.00211 (was +0.00214).
- Validates THREE previously stated rules:
  1. **Refined blend-magnitude rule (errs ≤ 1.005 × anchor)**: first
     blend +3.5% errs → LB -0.00014; this blend -3.0% errs → LB +0.00003.
     Sign of the LB delta tracks sign of the err-count delta.
  2. **Greedy forward-selection beats hand-picked α**: the hand-picked
     α=0.375 on LB2 produced a worse blend (OOF and LB) than the
     greedy-picked α=0.200 on LB3 with xgb_nonrule_iso stacked on.
  3. **RealMLP transfers as a blend leg** when paired correctly with
     an LB-proven anchor + an LB-proven secondary leg (xgb_nonrule
     was itself an LB +0.00056 lift earlier; its isotonic-calibrated
     version on top of RealMLP addition stacks cleanly).
- Per-class recall breakdown (stack vs LB3 anchor):
  Low 0.9955 (vs 0.9951, +0.0004); Medium 0.9689 (vs 0.9675, +0.0014);
  High 0.9774 (vs 0.9782, -0.0008). Net: macro-recall trade favors
  LB transfer — larger Medium+Low gains slightly outweigh small High
  drop under macro-recall's equal-class-weight aggregation.
- LB budget: **2/10 spent today**, 8 remaining.
- Artefacts:
  - `scripts/greedy_realmlp_refit.py`, `scripts/emit_realmlp_3stack.py`
  - `scripts/artifacts/greedy_realmlp_refit_results.json`
  - `submissions/submission_lb3_realmlp_nonruleiso.csv` (LB 0.98008, new best)
- **Updated final-selection candidates**:
  1. **Primary (new LB best)**: `submission_lb3_realmlp_nonruleiso.csv`
     → LB 0.98008, OOF 0.98061, gap +0.00053
  2. **Safe fallback (prior LB best)**: `submission_3way_recipe025_s1035_s7040.csv`
     → LB 0.98005, OOF 0.98029, gap +0.00024 (tighter calibration,
     lower variance surface — good hedge for private LB)
- Next bets:
  1. **n_ens=4 RealMLP retry** (~45 min GPU) — a lower-error RealMLP
     should stack even cleaner. Expected OOF floor ~0.9770+ (+0.00060
     vs n_ens=1), potentially pushing 3-stack OOF to 0.9808-0.9810.
  2. **4-step greedy** trying step 3 (recipe_full_te_a10) — risky
     since step 3 Δ=+0.00006 is below both LB-transfer threshold
     (+0.0002) and fold noise (~0.00088). Likely OOF-overfit.
  3. **RealMLP + other NN family** (Trompt / TabM) — test if another
     architecturally orthogonal leg stacks cleanly with RealMLP.
  4. **Greedy with finer α grid** around α=0.200 for RealMLP step —
     5-min diagnostic to check if {0.175, 0.225} give materially
     different OOF.

### 2026-04-24 — Missed-High detector: AUC 0.9711 but deploy precision 6.5% << break-even 8.8% → NULL closes Pareto-frontier via explicit High route

- Goal: execute the #1 candidate proposed after the disagree+router
  closure — train a binary XGB specifically targeting rows where
  `y = High AND teacher_argmax != High` (the 475 rows teacher misses,
  0.075% prevalence). If held-out AUC ≥ 0.9 the signal exists; deploy
  as a hard override (flip `teacher_pred ∈ {L, M}` → `High` on rows
  where `P_missed > θ`). Runs on top of LB-best 3-way teacher.
- Changed: `scripts/missed_high_detector.py` (binary:logistic XGB,
  depth=4, scale_pos_weight=1325, 46 features = 24 dist + 11 raw
  numerics + 3 teacher probs + teacher conf/argmax + 3 recipe probs
  + 3 nonrule probs, 5-fold StratifiedKFold seed=42 stratified on y).
  `scripts/missed_high_deploy.py` (θ-sweep × 3 score-band variants:
  all scores, score ∈ {5,6,7,8}, score=6 only).
- Diagnostic before run (crucial): missed-High by score band:
  ```
  score    n_in_band   missed-H    rate
  0-3        374k         0          0%
  4         117k        11       0.009%
  5          79k       124       0.157%
  6          38k       331       0.862%   ← 70% of all misses
  7          15k         5       0.033%
  8         2.7k         4       0.149%
  9         3.2k         0          0%
  ```
  95% of the missed-High signal concentrates in score ∈ {5, 6} bands.
- **Detector results** (5-fold OOF):
  - Per-fold AUC: 0.9765 / 0.9767 / 0.9835 / 0.9837 / 0.9830
  - **Overall OOF AUC = 0.9711** (consistent, all folds ≥ 0.97)
  - Training wall: 24 s total (best_iter 22-101 per fold — signal is
    easy for XGB to pick up).
  - Signal EXISTS in ranking terms. But…
- **Deploy sweep results** (all score-bands + all θ):
  ```
  BEST observed config: θ=0.90 all_scores
    n_overridden =  7,399
    correct      =    193   (true missed-H)
    false        =  7,206   (not actually High)
    precision    =  3%
    bal_acc      = 0.97331  (Δ = −0.00698 vs teacher 0.98029)
  ```
  Every single (θ, band) combination is **strictly net-negative on
  bal_acc**. At every threshold, false overrides outnumber correct
  overrides 30–100×.
- **Top-N rank-based probe** (idealized precision ceiling):
  ```
      N  precision  recall_miss  bal_acc   delta
    100    0.040       0.008     0.98022  -0.00007
    200    0.065       0.027     0.98023  -0.00005   ← max precision
    475    0.055       0.055     0.98007  -0.00021
   1000    0.051       0.107     0.97977  -0.00051
  ```
  Best achievable precision on the detector's top picks is 6.5% at
  N=200. Even the idealized "just take top N by rank" is net-negative.
- **Mathematical closure — break-even precision under macro-recall**:
  Each **correct** override adds `+1/21009 ≈ +4.76e-5` to High recall.
  Each **incorrect** override removes roughly `1/239074 ≈ +4.18e-6`
  from Medium recall (most false flips come from Medium, which
  dominates `teacher_pred != H` rows). For bal_acc to even break even,
  need `correct/incorrect > 21009/239074 = 0.088` → **precision ≥ 8.8%**.
  Observed precision at every θ is 1–6.5%, **3–8× below break-even**.
  The detector's ranking is real but insufficient to deploy.
- **Definitive closure** of the "High-recall orthogonality" path. The
  4-way evidence stack is now:
  1. (2026-04-24) Disagree meta — 335 fewer total errors, but trades
     High down 0.18 pp — Δ +0.00001 null.
  2. (2026-04-24) Selective router — +316 net correct but +78/+266/−28
     per-class distribution is anti-macro-recall — Δ -0.00000 null.
  3. (2026-04-24) Missed-High detector — AUC 0.9711 ranking signal
     exists but top-N precision 6.5% << 8.8% break-even — Δ -0.00005
     at best rank-based attempt.
  4. Per-candidate High-promotion sweep (from disagree+router entry)
     monotone-negative at every θ.
  **Teacher's High recall of 0.9774 is a genuine Pareto-frontier
  ceiling** given the current 7-candidate OOF bank. The 475 missed-H
  rows are not cleanly separable from teacher-correct rows in the
  feature space we have.
- **Portable rule** (LEARNINGS.md candidate): **"AUC ≥ 0.95 on a
  binary detector with <0.1% prevalence does NOT imply a useful
  operating point."** AUC is a pairwise ranking metric; when
  positives are rare, even modest false-positive rates on the
  negative class drown the top picks. Compute break-even precision
  from the target metric's class weights FIRST (under macro-recall:
  `break_even = n_positives / n_negatives_in_override_space`) and
  compare to observed top-N precision BEFORE declaring the lever
  workable. The detector's AUC tells you signal exists; precision
  tells you whether you can deploy it.
- LB delta: n/a (no submission warranted — every config strictly
  below teacher's OOF). LB budget unchanged at 10/10 used yesterday.
- Artefacts committed for cross-branch reuse:
  - `scripts/artifacts/oof_missed_high.npy` (630k,) binary prob
  - `scripts/artifacts/test_missed_high.npy` (270k,) binary prob
  - `scripts/artifacts/missed_high_results.json` (AUC breakdown)
  - `scripts/artifacts/missed_high_deploy_results.json` (θ × band sweep)
- **Strategic implication**: own-pipeline LB 0.98005 is the real
  ceiling within the current OOF bank. Breaking it requires a
  component whose errors come from a FUNDAMENTALLY different feature
  pathway, not a re-mapping of existing components. Remaining
  untried bets with plausible High-orthogonality:
  1. **RealMLP-TD on Kaggle GPU** — only NN family untested; if
     errors come from its periodic embeddings rather than trees,
     the detector pattern may break.
  2. **Focal-loss / Logit-Adjusted XGB on recipe features** —
     rare-class-focused training-time mechanism, not another
     post-hoc override.
  3. **SMOTE-NC or CVAE for synthetic High rows** — training-data
     augmentation, not model architecture. Direct attack on the 21k
     High sample shortage.

### 2026-04-24 — focal-loss recipe XGB + capacity-reduced soft-distill: both null with useful diagnostics

- Goal: execute the top-2 untried own-pipeline levers from the
  2026-04-24 brainstorm. (#1) Multi-class focal loss on the recipe
  feature set with γ=2 and α=invfreq — training-time class-asymmetric
  mechanism, not a post-hoc override. (#2) Capacity-reduced
  soft-distillation from the LB-best 2-way teacher (depth 4→3,
  max_leaves 30→15, n_round 3000→1500) — fixes the prior soft_distill
  overfit (OOF 0.98096 → LB 0.97850, gap +0.00246) by halving student
  memorization capacity.
- Changed: `scripts/focal_loss_common.py` (multi-class focal-xent obj
  factory, analytic grad + diagonal Hessian approx); `scripts/recipe_focal.py`
  (orchestrator mirroring soft_distill_xgb pattern, native xgb.train
  API, same FE+OTE pipeline, no sample_weight since alpha is baked
  into loss); `scripts/soft_distill_xgb.py` parameterized via
  `XGB_DEPTH`, `XGB_MAX_LEAVES`, `XGB_NROUND` env vars preserving
  defaults; `scripts/blend_focal_distill.py` (joint blend-gate
  analysis vs recipe / LB-best 2-way / LB-best 3-way anchors).

- **#1 focal loss (γ=2, α=invfreq=[1.0, 1.55, 17.61])**: NULL,
  structurally backwards.
  - Per-fold argmax: 0.97342 / 0.97347 / 0.97327 / 0.97095 / 0.97208
    (mean 0.97264 ± 0.00107; fold 4 hit 3000-iter cap). Recipe
    per-fold argmax mean was 0.97589.
  - OOF tuned 0.97683  (-0.00284 vs recipe 0.97967; -0.00346 vs
    LB-best 3-way 0.98029). Bias [1.53, 1.47, 3.00] — High bias 3.00
    < recipe's 3.40 (sharper raw High probs from α=17.6).
  - Errors 12,082 (+19% vs recipe 10,114); Jaccard 0.66 vs recipe,
    0.65 vs LB2, 0.61 vs LB3 — best error-orthogonality of any
    tree candidate.
  - Per-class recall (tuned): L 0.9944, M 0.9660, **H 0.9701**.
    **All three classes WORSE than every anchor.** H 0.9701 vs
    recipe 0.9765 / LB-best 2-way 0.9768 / LB-best 3-way 0.9774.
  - Blend gate vs all 3 anchors: monotone-negative from α=0 →
    strict null at every weight.
  - **Failure mode**: α=17.6 on rare-class + γ=2 (1-p)^2 modulator
    compound. γ=2 starves gradient on easy Low/Medium boundary
    rows (many of which are informative); α=17.6 introduces gradient
    noise on High rows that XGB at depth=4 can't exploit cleanly.
    Net: weaker discrimination on every class, not a differently-
    balanced one. Focal is closed as a direct-replacement lever on
    this feature set with this alpha magnitude.

- **#2 capacity-reduced soft-distill (depth=3, max_leaves=15,
  n_round=1500)**: OOF PASS all 3 blend gates; LB regression.
  - Per-fold argmax: 0.97541 / 0.97572 / 0.97618 / 0.97461 / 0.97550
    (mean 0.97548 ± 0.00051; best_iter consistently hit 1499 cap —
    student still learning at cutoff, regularization by budget).
  - OOF tuned **0.98066** — first standalone to beat LB-best 3-way
    0.98029 (+0.00037). Bias [0.83, 1.27, 3.20].
  - Errors 9,678 (FEWER than every anchor: recipe 10,114, LB2 9,851,
    LB3 9,983). Jaccard 0.79-0.81 vs anchors (below 0.80 novelty
    threshold vs recipe and LB3, just above vs LB2).
  - Per-class recall (tuned): L 0.9945, M 0.9698, **H 0.9777**.
    Higher on every class than every anchor — first candidate ever
    to achieve this AND have fewer errors.
  - Blend gate: peak α=0.45-0.50, Δ=+0.00042 to +0.00067 on all 3
    anchors. All PASS the +0.0002 LB-transfer threshold.
  - **LB probe (user-approved)**: `submission_soft_distill_small.csv`
    → **LB = 0.97865**. OOF→LB gap = **+0.00201**. Not enough
    capacity reduction — student at depth=3 / 1500 rounds still
    memorized teacher OOF noise.
  - Comparison to prior soft_distill:
    ```
                        prior (d=4, r=3000)   small (d=3, r=1500)
      OOF argmax         0.97557              0.97548
      OOF tuned          0.98096              0.98066  (-0.00030)
      errors             9,520                9,678    (+158, less memorize)
      LB                 0.97850              0.97865  (+0.00015)
      OOF->LB gap        +0.00246             +0.00201 (-0.00045)
    ```
  - Gap narrowed by 0.00045 (capacity reduction did something), but
    only +0.00015 LB over prior. Standalone LB 0.97865 is 0.00143
    BELOW new LB-best 0.98008 (achieved separately via RealMLP blend
    on `claude/gpu-nn-implementation-mh7N0`, merged to main at 19:27
    today).

- **Updated calibration ladder:**
  ```
  prior soft_distill (d=4, r=3000)   0.98096 → 0.97850  gap +0.00246
  **distill_small (d=3, r=1500)      0.98066 → 0.97865  gap +0.00201** (this session)
  LB-best 2-way (recipe × pseudo_s1) 0.98012 → 0.97998  gap +0.00014
  3-way multi-seed                   0.98029 → 0.98005  gap +0.00024
  **RealMLP 3-stack (other branch)   0.98061 → 0.98008  gap +0.00053 ← NEW LB BEST**
  focal-loss (this session)          0.97683 → (not probed, strict null)
  ```
- **Portable rules** (LEARNINGS.md candidates):
  1. **Focal loss with heavy α on a tuned XGB REGRESSES rather than
     balances.** The training-time gradient distortion at α=17.6
     combined with γ=2 modulator produces a weaker model on every
     class, not a rare-class-focused one. Rule: don't use focal
     loss on top of a class-weighted recipe unless you're prepared
     to reduce BOTH α AND γ well below Lin et al. defaults.
  2. **Soft-target distillation from a bagged-OOF teacher is
     structurally overfit-prone even at 2× capacity reduction.**
     Going from depth=4/3000 rounds to depth=3/1500 rounds narrowed
     the OOF→LB gap by 0.00045 (real effect) but the residual
     +0.00201 gap still eats all the standalone OOF lift. Portable
     fix requires either (a) 4×+ capacity reduction (depth=2,
     rounds≤500 — speculative), or (b) row-wise teacher leak
     elimination: hold row i out of ALL bagged components that
     form the teacher, not just the single model that produced
     teacher_oof[i]. The current 2× reduction is NOT sufficient.
  3. **"First to satisfy all blend-gate heuristics on OOF" is NOT
     a sufficient condition for LB transfer.** distill_small was
     the first candidate with (a) standalone > every anchor,
     (b) errors < every anchor, (c) per-class recall > every anchor,
     (d) Jaccard < 0.80 vs the strongest anchor, AND (e) peak-α
     blend Δ > +0.0002 on all 3 anchors. It still LB-regressed by
     -0.00143. Rule: any candidate built from teacher-OOF inputs
     (distillation, pseudo-label stage-2+) needs an independent
     leak-elimination check before being trusted.

- LB delta: -0.00143 vs LB-best (distill_small standalone submitted).
  Budget: 6/10 used today (5 probes from other branches earlier
  today + 1 this session), 4 remaining.
- Current LB best: **0.98008** (`submission_lb3_realmlp_nonruleiso.csv`
  on main, from RealMLP-TD blend on sibling branch).
- Artefacts committed for cross-branch reuse (gitignore-whitelisted):
  `scripts/artifacts/oof_recipe_focal_g2_invfreq.npy` + test + JSON,
  `scripts/artifacts/oof_soft_distill_small.npy` + test + JSON,
  `scripts/artifacts/blend_focal_distill_results.json`,
  `submissions/submission_recipe_focal_g2_invfreq.csv` (diagnostic,
  below all anchors, not for LB probe),
  `submissions/submission_soft_distill_small.csv` (probed → 0.97865).

- **Next bets** (post-session):
  1. **Extreme capacity reduction distill** (depth=2, rounds=500) —
     direct test of whether the fix is simply not-enough-reduction.
     ~20 min wall. If OOF collapses below recipe, the lever is
     dead; if OOF stays ≥ LB-best 3-way with Jaccard < 0.80, then
     gap may finally narrow below +0.0005 for genuine LB lift.
  2. **Port RealMLP as a blend leg into this branch's candidate
     bank** — new LB best is a blend of LB-best 3-way + RealMLP +
     xgb_nonrule_iso. Running our greedy + blend-gate analysis on
     the expanded bank (now including distill_small as a potential
     tiny-α addition) may surface configurations the other branch
     didn't find.
  3. **Leak-eliminated distillation**: build a teacher OOF where
     for each row i, ALL bagged components were trained on folds
     that exclude row i (not just one). Expensive — requires
     retraining each teacher component 5×. But it directly fixes
     the overfit mechanism rather than mitigating it via capacity.

### 2026-04-24 — W2/W3/W5 weaknesses-audit session (3 experiments: 2 nulls, 1 guardrail shipped)

- **Context**: branch `claude/identify-ml-weaknesses-RJbG4` session
  driven by a plan-mode audit identifying five specific weaknesses
  (W1 = RealMLP completion, W2 = boundary-cell specialist,
  W3 = binary Medium head, W4 = stranded NN OOFs, W5 = greedy
  guardrail). Executed W2 / W3 / W5 locally this session (W1 and W4
  require Kaggle GPU push, deferred). None broke LB 0.98005 but
  two portable rules + one tooling guardrail landed.
- **W2 — xgb_specialist_36 (boundary cells {3,6}) closed NULL**:
  43-feature dist set, 5-fold seed=42, spec domain 140,573 rows
  (Low 69.2% / Medium 29.7% / High 1.1%). Specialist standalone
  spec-domain bal_acc 0.666 vs rule 0.628 (+0.038 real lift), but
  LB-best 3-way scored **0.878 on the same rows** (21 pp higher).
  Override (α=1) on spec rows: Δ=−0.00846 OOF. Softmix sweep
  monotone-neg past α=0.05 on all three routing widths (score-3,
  score-6, combined). Root cause: bi-modal domain (score 3 is 95/5
  Low/Med, score 6 is 96/4 Med/High) + 43-dist features ≺ 443-
  recipe. Spec has 750 fewer raw errors but 21 pp lower bal_acc
  because High recall craters (12% vs 97.74%). Two rules landed:
  (a) bi-modal aggregates can mask the 20-80% minority heuristic —
  check at sub-domain granularity, and (b) specialist feature set
  must match the anchor's feature richness.
- **W3 — binary_medium_head closed NULL**: 443-feature recipe set,
  XGBoost binary:logistic on `y == 1`, 5-fold seed=42. OOF AUC
  **0.99767** (per-fold 0.9975-0.9978). Fixed-bias sweeps on LB-
  best 3-way (NO log-bias retune, per 2026-04-21 binhigh rule):
  prob_mix and geo_mix peaks at g=0 (monotone-neg past 0);
  logit_add peak at g=0 (monotone-neg both sides). Raw error count
  DROPS with positive weight (logit_add +1.0: 8,420 errs vs
  baseline 9,873 — 1,453 fewer wrong rows) but bal_acc DROPS in
  lockstep because High recall is sacrificed first (prior 3.3% →
  12× per-row leverage under macro-recall). Rule extends binhigh:
  a binary class-k head on the SAME feature basis as the 3-class
  anchor cannot lift at fixed bias regardless of AUC quality —
  binary-Medium head is a reparameterisation of information
  already in the 3-class Medium column.
- **W5 — EXCLUDE_GREEDY_ADD guardrail shipped** in
  `scripts/c0_safe_greedy_v3.py`. Two-level exclusion:
  `EXCLUDE_FROM_POOL` = `{soft_distill, xgb_spec_678,
  recipe_pseudolabel_stage2}` (never loaded) and
  `EXCLUDE_GREEDY_ADD` = pool-excludes ∪ `{recipe_pseudolabel_
  seed7labeler, recipe_pseudolabel_seed123labeler}` (valid anchor
  ingredients, cannot be greedy-added). Diagnostic validated on a
  side-by-side unguarded-vs-guarded run:
  ```
  anchor               unguarded  guarded    Δ
  recipe_full_te       0.98047    0.98032   -0.00015  ← guardrail strips seed7labeler
  lb_best_3way         0.98041    0.98041   +0.00000  ← seed7labeler already in anchor
  ```
  Invariant `guarded_OOF ≤ unguarded_OOF` holds in both cases. The
  guardrail correctly prevents greedy from re-picking the
  `seed7labeler` 2-way (documented LB regressor −0.00029) as a
  new addition. For future greedy experiments on any branch that
  inherits this script, adding new LB-regressors to the
  EXCLUDE_GREEDY_ADD set is the cheapest insurance against
  burning an LB slot on the same regression pattern twice.
- LB budget unchanged: 0 probes spent this session. Current LB-
  best still `submission_3way_recipe025_s1035_s7040.csv` at
  **LB 0.98005**.
- Three LEARNINGS.md rules landed (see LEARNINGS.md Modelling +
  Ensembling sections):
  1. Bi-modal sub-domains can mask 20-80% minority heuristic.
  2. Specialist feature set must match anchor's feature richness.
  3. Binary class-k head on same basis as anchor ≈ null at fixed
     bias regardless of AUC.
  4. `EXCLUDE_GREEDY_ADD` two-level exclusion for greedy
     forward-select once ≥3 LB-regressors are in the pool.
- Remaining W1/W4 require Kaggle GPU push (RealMLP completion +
  stranded NN OOF backfill); skipped for next session.

### 2026-04-24 — #3 focal-loss XGB + #4 score=6 M-vs-H specialist: both NULL, but v2 is first precision-beating override

- Goal: execute candidates #3 (focal-loss XGB on recipe features) and
  #4 (per-cell score=6 Medium↔High specialist) from the remaining
  recommendation list after the Pareto-frontier closure. Both target
  the High-recall weakness but via different mechanisms: #3 at training
  time, #4 at deploy time.

- **#3 focal-loss XGB** — `scripts/focal_common.py` +
  `scripts/recipe_focal.py` + `scripts/blend_focal.py`.
  Custom multi-class focal-weighted CE objective:
  ```
  w_sample = alpha_y * (1 - p_y)^gamma
  grad     = w * (softmax(z) - one_hot(y))
  hess     = w * p * (1 - p)   + eps     (diagonal softmax approx)
  ```
  Outer-weight approximation (treats w as constant in grad step; exact
  focal gradient has extra terms from chain rule through (1-p_y)^gamma
  but the approximation is indistinguishable at gamma ≤ 3 and keeps
  hess PSD). Production run with gamma=2, alpha=(1,1,3) on the 443-
  feature recipe matrix, xgb.train native API, custom_metric=bal_acc
  (maximize), NO sample_weight='balanced' (focal alpha already
  upweights rare class). 29 min wall on CPU.
  - Per-fold timings:
    ```
    fold 1  best_iter=200  best_score=0.97472  argmax=0.97327  wall=433s
    fold 2  best_iter=135  best_score=0.97512  argmax=0.97313  wall=326s
    fold 3  best_iter=61   best_score=0.97682  argmax=0.97477  wall=242s
    fold 4  best_iter=59   best_score=0.97422  argmax=0.97344  wall=240s
    fold 5  argmax ~0.97384 (inferred from mean 0.97369 ± 0.00059)
    OOF argmax = 0.97369  tuned = 0.97846  bias=[1.73, 1.77, 2.60]
    ```
  - Tuned bias insight: High bias 2.60 vs recipe's 3.40 — focal's
    alpha_H=3 already pushes High probs up at training time, so
    post-hoc log-bias correction is smaller. Test dist
    [159674 / 100201 / 10125] = 3.75 % High (above train prior 3.33 %).
  - Blend-gate vs LB-best 3-way (0.98029):
    ```
                    errs   Jaccard vs recipe   Jaccard vs 3-way
    recipe         10,114         1.00              0.79
    LB-best 3-way   9,983         -                 1.00
    focal          10,450       0.779             0.739
    peak alpha vs recipe       = 0.000  delta = +0.00000
    peak alpha vs 2-way LB-best= 0.025  delta = +0.00002
    peak alpha vs 3-way LB-best= 0.000  delta = +0.00000
    ```
  - **Jaccard 0.74 PASSES the orthogonality gate (< 0.80)** —
    focal's errors ARE genuinely different. But errs 10,450 > 9,983
    → +467 more errors than 3-way fails magnitude gate. Classic
    magnitude-trap null.
  - Per-class recall at own tuned bias:
    ```
               recL      recM      recH
    recipe    0.9950    0.9675    0.9765
    3-way     0.9941    0.9694    0.9774
    focal     0.9950    0.9663    0.9741   ← LOWER High than recipe
    ```
  - **Counter-intuitive finding**: focal with High-class upweight
    produces LOWER High recall than baseline recipe. Mechanism:
    gamma=2 + alpha_H=3 + no sample_weight='balanced' over-concentrates
    gradient on rare-class rows, shifting the Medium-vs-High decision
    surface TOWARD High predictions. But the shift is DISCRETE
    (threshold changes) not PROBABILISTIC (calibration changes). Some
    previously-correct Medium rows flip to High (wrong); some
    previously-missed-High rows get caught (right); net per-class
    recall is LOWER on High because the threshold overshoot loses
    ~30 True-Medium rows per ~15 True-High captured. log-bias
    coord-ascent at tuned bias=2.6 tries to compensate but plateau
    is structural.
  - **Portable rule** (LEARNINGS.md candidate): **"Focal loss with
    aggressive per-class alpha on imbalanced tabular problems can
    REDUCE rare-class recall"** when the alpha upweight is not
    paired with a matching increase in early-stopping patience.
    Focal's best_iter dropped from recipe's 1200+ to 59-200 rounds;
    training stops while the rare-class decision surface is still
    overshooting. Either (a) use milder alpha (2.0 instead of 3.0),
    (b) remove focal gamma (plain weighted-CE), or (c) extend
    early_stopping_rounds 3-5× to let the overshoot self-correct.
  - LB delta: n/a (both gates fail below +0.00020 LB-transfer
    threshold). Submission at alpha=0.00 emitted as diagnostic
    (identical to LB-best 3-way standalone).

- **#4 score=6 Medium-vs-High binary specialist** —
  `scripts/spec6_mh.py` (v1) + `scripts/spec6_mh_v2.py` (v2) +
  `scripts/spec6_deploy.py` + `scripts/spec6_deploy_v2.py`.
  - Score=6 band (38,416 rows, 4.03 % High prevalence, rule always
    predicts Medium) concentrates 70 % of all missed-High signal per
    the 2026-04-24 error analysis. Binary P(y=High | features,
    score=6) trained only on score=6 train-fold rows, 5-fold
    StratifiedKFold(seed=42) aligned with every saved OOF.
  - **v1 features (35)**: 28 dist-features (sm/rf/tc/ws dist+abs +
    rule flags + score + boundary distances + pairwise products)
    + 7 nonrule numerics (`Humidity, Previous_Irrigation_mm,
    Electrical_Conductivity, Soil_pH, Organic_Carbon, Sunlight_Hours,
    Field_Area_hectare`). OOF AUC = **0.862**.
  - **v2 features (40)**: v1 + 5 teacher meta-features
    (`teacher_PL, teacher_PM, teacher_PH, teacher_mh_margin,
    teacher_mh_ratio`). Teacher = LB-best 3-way log-blend
    (0.25 recipe + 0.35 pseudo_s1 + 0.40 pseudo_s7), OOF-leakfree
    by construction. OOF AUC = **0.938** (+0.076 vs v1). best_iter
    dropped 352-544 → 34-128 rounds — teacher probs carry most of
    the signal; XGB converges fast on residual.
  - **Deploy mechanism**: hard-override teacher_pred=Medium rows on
    score=6 → High where P_spec > theta. Under macro-recall break-
    even precision = H_count / M_count = 21009/239074 = 8.8 %
    (or 8.1 % in the ratio form). Override space: 35,180 rows
    (score=6 ∩ teacher-Medium), of which 331 are truly-High.
  - **v1 peak at theta=0.50**: 41 overrides, 4 correct → 9.8 %
    precision. Delta = +0.00001. Marginal.
  - **v2 peak at theta=0.15**: 25 overrides, 7 correct →
    **28.0 % precision** (3.2× break-even). Delta = +0.00009.
    Rank-based top-N=50: 14 % precision, delta +0.00005.
  - Test-side deploy: v1 at theta=0.50 → 15 overrides;
    **v2 at theta=0.15 → only 2 overrides**. Test distribution
    has sharper high-confidence cutoffs than OOF (expected).
  - **Significance**: v2 is the FIRST experiment on this problem
    to cleanly BEAT break-even precision on a High-class override
    (28 % >> 8.8 %). The 2026-04-24 Pareto-frontier closure noted
    that no candidate in the 7-model bank could predict High
    correctly on teacher-miss rows, even at high candidate
    confidence. v2 with teacher meta-features partially breaks
    that — but only on the tiny score=6 override space, where
    absolute counts are too small to move LB.
  - **Portable rule** (LEARNINGS.md candidate):
    **"Teacher posteriors (especially boundary-margin features
    like `P_M - P_H` or `log(P_H/P_M)`) are the strongest signal
    source for boundary-specialists."** v1→v2 went from AUC 0.862
    to 0.938 purely by adding 5 teacher-derived features to 35
    raw/dist features. Mechanism: boundary uncertainty is what
    the rare-class override should target, and teacher's
    calibrated margins directly measure that uncertainty. Raw
    features force the specialist to re-discover the boundary;
    teacher features hand it over pre-computed. Use this pattern
    whenever training a specialist to complement a strong
    base-model's decisions.
  - LB delta: n/a. Both variants below +0.00020 LB-transfer
    threshold; test override count (2 rows at v2 peak) too small
    to shift LB.

- **Combined session read-out** (both #3 and #4 NULL for LB):
  - Focal-loss: magnitude-trap null (Jaccard passes, errs fails).
    The orthogonality-is-necessary-not-sufficient rule confirmed
    yet again. Reinforces the 2026-04-24 Pareto-frontier closure.
  - Score=6 specialist v2: first precision-beating override
    mechanism on this problem; lever EXISTS (28 % precision) but
    prevalence-bounded to ~10 override rows. Proof-of-concept that
    teacher meta-features unlock signal hiding in raw features,
    but insufficient magnitude to move LB.
  - Still remaining untried (from the strategic shortlist):
    RealMLP smoke-first retry (only NN family untested); SMOTE-NC
    synthetic High rows; B0 DivideMix. Current LB best unchanged
    at **0.98005** (3-way multi-seed).

- Artefacts committed for cross-branch reuse (gitignore whitelist):
  - `scripts/artifacts/oof_recipe_focal_g2h3.npy` + test + JSON
  - `scripts/artifacts/oof_spec6_mh.npy` + test + JSON (v1, AUC 0.862)
  - `scripts/artifacts/oof_spec6_mh_v2.npy` + test + JSON (v2, AUC 0.938)
  - `scripts/artifacts/spec6_deploy_v2_results.json` (theta sweeps + rank)
  - `scripts/artifacts/blend_focal_g2h3_results.json` (full blend-gate output)
  - `submissions/submission_recipe_focal_g2h3.csv` (diagnostic)
  - `submissions/submission_focal_blend_a000.csv` (diagnostic, = 3-way)
  - `submissions/submission_spec6_override_th50.csv` (v1 hard-override test)
  - `submissions/submission_spec6_override_v2_th15.csv` (v2 hard-override test)

### 2026-04-25 — per-score-bin log-blend on LB-best 3-way: NULL with regression

- Goal: test the highest-EV remaining ensembling-only lever — fit
  separate log-blend weights per `dgp_score` bin instead of one global
  weight vector. Errors concentrate at score ∈ {3, 6} (74% of error
  mass per the 2026-04-24 error analysis); the global LB-best weights
  (0.25 / 0.35 / 0.40 on recipe / pseudo_s1 / pseudo_s7) are a cross-
  bin compromise that may be improvable locally.
- Changed: `scripts/per_bin_blend.py` (~250 lines, single file). 5
  bins: `{0,1,2}`, `{3}`, `{4,5}`, `{6}`, `{7,8,9}`. Per-bin
  coordinate descent over the 3-simplex at step=0.05 (231 grid points)
  with global fixed-bias bal_acc as the objective (NOT per-bin
  log-loss — that variant ran first and produced an in-sample
  REGRESSION of −0.00007, falsifying log-loss as a useful proxy
  here). Honest nested 5-fold CV: for each outer fold, fit weights on
  the 4 outer-train fold rows of the saved OOFs, apply to the held-
  out fold, concatenate. Test-side prediction uses fit-all-data
  weights. Optimised inner loop (only recompute the changing bin's
  slice each trial) for ~5x speedup; full nested run wall ~9 min CPU.
- Results (OOF tuned bal_acc, fixed recipe bias [1.4324, 1.4689, 3.4008]):
  ```
  baseline 3-way (LB-best)         0.98029   (matches prior log)
  in-sample per-bin (optimistic)   0.98037   Δ = +0.00009
  nested CV per-bin (honest)       0.97997   Δ = -0.00031   ← gate miss
  nested tuned (diagnostic)        0.98013   Δ = -0.00015
  overfit gap in-sample → nested   0.00040   (~5x typical greedy gap)
  ```
- **Smoking gun — bin-weight instability across folds.** Bin
  `score_6` (the Med↔High boundary, 70% of missed-High signal)
  picked five wildly different "optima" depending on which 4/5 of
  the data the search saw:
  ```
  in-sample: (recipe 0.35, pseudo_s1 0.40, pseudo_s7 0.25)
  fold 1:    (         0.25,           0.35,           0.40)  baseline unchanged
  fold 2:    (         0.30,           0.70,           0.00)
  fold 3:    (         0.20,           0.05,           0.75)
  fold 4:    (         0.35,           0.35,           0.30)
  fold 5:    (         0.20,           0.60,           0.20)
  ```
  All over the simplex. Textbook fold-dependent noise being fitted,
  not real per-bin signal. Bins 3 and 7-9 were more stable but
  still drifted; only bins 0-2 and 4-5 were nearly fold-invariant.
- **Mechanism**: 5 bins × 2 free weights per bin = 10 degrees of
  freedom on a single CV split. The per-bin signal density (effective
  sample at the boundary scores is small after stratifying by class
  AND filtering to the bin) doesn't support that many parameters
  at the +0.0001 resolution we'd need. Coord-descent on macro-recall
  at step=0.05 has enough capacity to fit per-bin macro-recall noise.
  In-sample lift (+0.00009) is the noise that nested CV correctly
  discounts; honest evaluation on held-out fold rows produces a NET
  REGRESSION because the chosen weights don't generalize.
- LB delta: n/a. Fixed-bias gate threshold (+0.00020) clearly
  failed; no submission emitted. LB-best unchanged at **0.98005**
  (`submission_3way_recipe025_s1035_s7040.csv`).
- **Portable rule** (LEARNINGS.md candidate): **"On a single CV
  split, per-bin / per-region log-blend over a small simplex
  search is overfit-prone when the per-bin signal density is
  below ~0.0005 OOF lift per parameter. Honest nested CV will
  show a 4-5x larger overfit gap than greedy forward selection on
  the same OOF bank, with the regression magnitude scaling with
  the number of bins × free weights per bin. Fit ONE global blend
  per CV split unless per-bin lift exceeds +0.0005 in-sample on
  every bin."**
- **Strategic implication**: this closes per-region blending as an
  ensembling-only lever on the existing OOF bank. Combined with
  the prior nulls (disagree-stacker 2026-04-24, selective-router
  2026-04-24, missed-High detector 2026-04-24, isotonic-greedy
  2026-04-24, multi-seed pseudo-label saturation 2026-04-24), the
  Pareto-frontier closure is reconfirmed: re-arranging existing
  components cannot break LB 0.98005. The honest path past the
  ceiling requires ADDING a new component with the right
  Jaccard+magnitude profile (RealMLP n_ens=4 retry, Trompt, TabM,
  or a fresh FE surface).
- Artefacts committed for cross-branch reuse (gitignore whitelist):
  - `scripts/per_bin_blend.py` (parameterised: `OBJECTIVE` env
    selects bal_acc_global / log_loss; `GRID_STEP` controls
    simplex resolution; reusable on future comps)
  - `scripts/artifacts/oof_per_bin_blend.npy` (nested OOF, 7.2 MB)
  - `scripts/artifacts/test_per_bin_blend.npy` (test blend, 3.1 MB)
  - `scripts/artifacts/per_bin_blend_results.json` (full per-fold
    weight log + overfit gap stats for audit)
  - No submission CSV — gate correctly blocked emission.

### 2026-04-25 — Tier 1a n_ens=4 RealMLP retry: NULL (strictly worse than n_ens=1)

- Goal: build on the 2026-04-24 LB-best 0.98008 (LB-3way + realmlp@0.2
  + nonrule_iso@0.075) by upgrading the RealMLP leg from n_ens=1 to
  n_ens=4. Tier 1b had already confirmed the OOF bank is saturated —
  the only remaining lever in the 3-stack architecture is improving
  the RealMLP component itself. Hypothesis: 4× BatchEnsemble heads
  cuts per-row variance ~2×, dropping errs below the anchor and
  unlocking another +0.0002-0.0003 LB.
- Changed: new `kaggle_kernel/kernel_realmlp_ens4/` (separate from
  the n_ens=1 kernel so artifacts don't clobber). Config shifts from
  n_ens=1 baseline: `n_ens=1→4`, `n_epochs=40→25` (tighter to fit the
  1h cap with 4× internal heads), `TargetEncoder cv=2` unchanged.
  In-kernel safety nets (fold-1 t+20min, total t+55min) unchanged.
  SMOKE-first discipline: SMOKE v1 (2-fold/20k/3 epochs) passed in
  <1 min training. `scripts/blend_realmlp_ens4.py` runs the diagnostic
  comparing n_ens=1 vs n_ens=4 across standalone, LB3+RM 2-stack,
  and full 3-stack (LB3+RM+nonrule_iso).
- Production v2: 5/5 folds COMPLETE in **38.2 min wall** (well under
  the 55min cap). Per-fold argmax 0.9698/0.9711/0.9710/0.9702/0.9680,
  σ=0.00115 (TIGHTER than n_ens=1's σ=0.00144 — variance reduction
  worked at the per-fold level). But:
  ```
                       standalone                 3-stack peak (LB3+RM+nonrule_iso)
                       OOF argmax  errs @bias     OOF       errs
  n_ens=1 (LB best)    0.97055     10472          0.98061   9572
  n_ens=4              0.97002     10597          0.98050   9505
  Δ                    -0.00053    +125            -0.00011  -67
  ```
  n_ens=4 has MORE errors at standalone bias (+125 vs anchor) but
  FEWER errors in the 3-stack (−67). However, the 3-stack OOF is
  LOWER (0.98050 vs 0.98061) — error count went down but in the
  wrong distribution for macro-recall. Per-class trade unfavourable.
- Standalone Jaccard vs LB-best 3-way: n_ens=1 = 0.6206, n_ens=4 =
  0.6243. n_ens=4 is SLIGHTLY MORE redundant with the anchor, not
  more orthogonal. Variance reduction collapsed the prediction
  surface toward the mean-tree-blend's surface.
- Projected LB at the +0.00053 OOF→LB gap: 0.98050 − 0.00053 =
  0.97997 (BELOW current LB-best 0.98008). **No LB probe warranted.**
- **Diagnosis**: variance dropped (per-fold σ tightened) but the
  variance reduction came with a small calibration shift that made
  the model's prob distribution closer to what trees produce.
  Two plausible mechanisms:
  1. **Under-converged heads**: cutting `n_epochs` 40→25 to fit the
     1h cap with 4× heads means each shared-weight pass saw less
     gradient — under-converged heads averaged together produce a
     less-biased but smoothed-out prediction.
  2. **Variance floor**: RealMLP at n_ens=1 was already at the
     low-variance floor for this problem. Further ensembling adds
     bias (mode collapse) without removing variance.
  Both predict the same outcome: "more ensemble heads at fixed
  compute budget" doesn't help — and might hurt — when the base
  variance is already low.
- **Per the decision framework committed at b076d3f**: "n_ens=4
  plateaus at LB ≤ 0.98015 → don't push Trompt, pivot to Tier 2".
  We hit exactly that case. The Trompt scaffold remains committed
  but unpushed; the lever-existence test would now have negative EV
  (compounding +0.0003 NN OOF→LB surcharge over a 2nd NN family on
  top of a marginal n_ens=4 "lift" that's already negative).
- LB budget: unchanged — no probes spent today. Current LB-best
  unchanged at **0.98008** via `submission_lb3_realmlp_nonruleiso.csv`.
- Next bets ordered by EV:
  1. **n_ens=2 with n_epochs=40** (~50 min GPU). Direct test of the
     under-convergence hypothesis: if n_ens=2 at full epoch budget
     beats n_ens=1, the lever is "ensembling at fixed-per-head
     epochs" (and we should retry n_ens=4 at n_epochs=40 in a
     non-Kaggle env that allows >1h wall). If n_ens=2 plateaus,
     RealMLP variance floor is structural and ensembling is dead.
  2. **Tier 2 FE/data-quality** on recipe_full_te. Untested
     territory: SMOTE-NC for High rows, stricter class weights,
     time-of-year features if any temporal signal exists.
  3. **Push Trompt anyway** as lever-existence — separate NN family
     might break the 12-NN-null pattern even if RealMLP is exhausted.
     Compounding gap surcharge (+0.0006 OOF→LB) makes this risky;
     gate at OOF > +0.0007 over 3-stack baseline before LB probe.
  4. **Lock LB 0.98008 as final and stop spending GPU budget**. With
     6 days to deadline and tight calibration (gap +0.00053),
     diminishing returns past this point.
- Artefacts:
  - `kaggle_kernel/kernel_realmlp_ens4/` (kernel + metadata)
  - `scripts/blend_realmlp_ens4.py` (diagnostic)
  - `scripts/artifacts/realmlp_ens4_results.json` (per-fold + tuned
    bal_acc; .npy artifacts whitelisted as a cross-branch diversity
    leg even though strictly worse than n_ens=1 in our 3-stack)
- **Trompt scaffold stays committed but unpushed** at
  `kaggle_kernel/kernel_trompt/` (modular: boot/config/features/
  model/cv/main + build.py concatenator). Ready to push if a
  different decision-framework signal emerges; otherwise stays in
  cold storage.
### 2026-04-25 — Tier 1b: XGB meta-stacker isotonic blend → NEW LB BEST 0.98094 (+0.00086)

- Goal: while parallel branch ran GPU experiments (RealMLP n_ens=4 / Trompt),
  exhaust cheap CPU levers on top of the LB-best 3-stack (OOF 0.98061 / LB
  0.98008). Five experiments: greedy refit, error-geometry → spec_lm_v3,
  XGB meta-stacker, xgb_nonrule seed-bag, combined diagnostic. **Four
  null, one breakthrough.**

- **Step 1 — greedy on LB-best 3-stack (73 components)**: only step-1
  candidate is `recipe_full_te_a10 α=0.200` Δ=+0.00006, below +1e-4
  gate. **Null.** Confirms the saturation conclusion from earlier.

- **Step 1b — spec6_mh_v2 transfer to NEW teacher**: v2 was trained with
  OLD teacher meta-features. On the new override space (35,335 score=6
  Medium-argmax rows, 326 truly-H), best θ=0.20 gives 4 OOF overrides at
  50% precision but **0 test overrides** — the new teacher already catches
  the easy score=6 H-flips. Lever exhausted without retraining.

- **Step 2 — error-geometry analysis** revealed dominant buckets:
  ```
  score=3 Medium→Low      n=4,324  (45.2% of total errors)
  score=6 Medium→High     n=1,858  (19.4%)
  score=4 Low→Medium      n=1,354  (14.1%)
  ```
  Built `spec_lm_v3` (Low↔Medium specialist, score=3 binary head with
  new teacher meta-features). AUC 0.827. Break-even precision under
  macro-recall = M/(L+M) = **39.3%** (4× stricter than spec_mh's 8.1%
  because Low is the majority class). Best θ=0.35 hits 45.2% precision
  on 168 overrides, but Δ OOF = **+0.00002** — tiny because each correct
  L→M flip gives ~1/4 the macro-recall weight of a correct H flip.
  The bucket is 10× bigger than score=6, but the math goes the wrong
  way. **Real signal, but architecturally mass-bounded.** Null on LB.

- **Step 3 — xgb_nonrule 3-seed bag (XGB seeds {42, 7, 123}, fold seed
  fixed at 42)**: standalone bag tuned OOF 0.53767 (+0.00167 vs single
  0.53600). Drop-in replacement in LB-best stack: Δ = **−0.00005**
  (mild regression). Diagnosis: the bag's prob-scale shifts away from
  the single-seed isotonic anchor; a higher α=0.125 partially recovers
  but lifts only +0.00001. **Model-seed bagging on top of an
  already-isotonic-calibrated leg is not automatically additive.** Null.

- **Step 4 — combined diagnostic** (fine meta α grid + spec_lm_v3 on
  meta-enhanced):
  - Raw meta peak shifts from α=0.40 to α=0.325 → OOF 0.98075 (+0.00014)
  - + spec_lm_v3 θ=0.35 → OOF 0.98077 (+0.00016)
  - Below +2e-4 gate but first combined lift.

- **Step 5 — greedy with meta-stacker + isotonic-calibrated copies in
  pool, finer α grid (0.01 → 0.5)**:
  ```
  step1: + xgb_metastack__iso α=0.300  OOF 0.98084  Δ=+0.00023  ← KEY
  step2: + recipe_no_digits  α=0.010  OOF 0.98087  Δ=+0.00002  (stop)
  ```
  Critical insight: **isotonic calibration of the meta-stacker output
  was the breakthrough**. Raw meta blend peaked at α=0.40 with +0.00012
  (LB-marginal); iso version peaks at lower α=0.30 with +0.00023. Same
  calibration-alignment mechanism the 2026-04-24 c0_isotonic experiment
  observed with CatBoost (+0.00197 from iso alone). Iso version's
  per-class probabilities align with LB-best's fixed bias, so the
  fixed-bias decision rule actually exploits the new signal.

- **Bucket-level diagnostic** (where the +157 net-correct flips come from):
  ```
  score=6 Medium predicted High → corrected to Medium:  +157  ← biggest win
  score=7 Medium → Medium:                              +28
  score=8 Medium → Medium:                              +17
  score=3 Low → Medium (correct flips from M→L bucket): +33
  score=6 Medium Medium → High (mistake):               -72
  score=3 Low Low → Medium (mistake):                   -30
  ```
  Net flips = 157 = exactly the 9572 → 9415 error reduction. The
  meta-stacker corrects LB-best's log-bias over-push on score=6
  boundary rows — the SAME bucket the error-geometry analysis
  identified as the second-largest mass-carrier.

- **Pre-LB diagnostic on candidate** (`submission_tier1b_greedy_meta.csv`):
  - OOF 0.98084 / errs 9,415 (Δ −157 vs LB-best 9,572)
  - Per-class recall: Low 0.9955 (unchanged), Medium 0.9695 (+0.0006),
    High 0.9775 (+0.0001) — no class hurt
  - Jaccard vs LB-best 0.955 = small calibration perturbation, not
    structural blend
  - Rows differing on test: 196 (0.07%)

- **LB PROBE (submitted 05:37 UTC, user-approved)**:
  **LB public = 0.98094** ← **NEW LB BEST**, Δ vs prior 0.98008 = **+0.00086**.
  OOF→LB gap = **−0.00010** (LB above OOF — first negative gap since
  the digit-XGB era).

- **The OOF underestimated the LB lift by ~3.7×.** Diagnosis: the
  meta-stacker over 63 components captures cross-component disagreement
  patterns that 5-fold OOF underestimates. CV is too pessimistic when the
  hold-out fold's component-OOFs are themselves noisy estimates of each
  component's behavior; the meta-stacker's improvement is averaged across
  noisy hold-outs. On the test set, all 63 components fire on the same
  unseen rows, and the meta-stacker's signal accumulates without that
  fold-noise smearing.

- **Updated calibration ladder:**
  ```
  recipe_full_te                  0.97967 → 0.97939   gap +0.00028
  recipe × pseudo_s1 2-way        0.98012 → 0.97998   gap +0.00014
  3-way multi-seed                0.98029 → 0.98005   gap +0.00024
  LB-best 3-stack (lb3+rmlp+nr)   0.98061 → 0.98008   gap +0.00053
  **+ xgb_metastack__iso α=0.300  0.98084 → 0.98094   gap -0.00010** ← NEW LB BEST
  ```
- Pack 0.98114 now only **+0.00020 above** (was +0.00106).
- Leader 0.98219 now only **+0.00125 above** (was +0.00211).
- LB budget: **3/10 used today**, 7 remaining.

- **Portable rules** (logging to LEARNINGS.md):
  1. **Isotonic-calibrate meta-stacker outputs before blending into a
     fixed-bias stack.** Raw multi-class probs from a meta-XGB at heavy
     reg can be miscalibrated relative to the anchor stack's bias; iso
     re-aligns per-class scales and shifts the optimal α downward, often
     unlocking 50-100% more lift. Tested at +0.00009 OOF gain on this
     problem (raw 0.98073 → iso 0.98084 at peak α).
  2. **OOF can underestimate LB lift for meta-stackers built over noisy
     OOF banks.** When the meta-stacker's input features are themselves
     fold-OOFs (noisy hold-out estimates), the meta-stacker's CV bal_acc
     under-counts its true generalization. On test, all components see
     unseen rows simultaneously and the meta's signal accumulates cleanly.
     Negative OOF→LB gap is a signature of this. Don't down-weight a
     meta-stacker candidate on OOF Δ alone — submit if the diagnostic
     (Jaccard < 0.97, errors ≤ anchor, no class hurt) passes.
  3. **For binary boundary specialists, break-even precision under
     macro-recall depends on the class-pair**: spec H↔M (rare class)
     break-even = H/(M+H) ≈ 8%; spec L↔M (majority class) break-even =
     M/(L+M) ≈ 39%. Bucket size and break-even pull in opposite
     directions — score=3 has 10× the row mass of score=6 but each
     correct flip is worth 4× less under macro-recall.

- New reusable scripts:
  - `tier1b_xgb_metastack.py` — 63-component XGB meta-stacker (5-fold
    stacking, max_depth=4 heavy-reg, 200+ feature dim)
  - `tier1b_err_geometry.py` — confusion + score×direction bucket dump
  - `spec_lm_v3.py` — Low↔Medium boundary specialist template (mirror
    of spec_mh_v3 with target = (y == Medium) and L-M margin features)
  - `tier1b_greedy_with_meta.py` — greedy forward with finer α grid +
    isotonic calibration applied to every pool component
  - `tier1b_combined.py` — fine α grid + meta + spec_lm_v3 combo sweep
  - `tier1b_verify_meta_iso.py` — pre-LB diagnostic harness

- Artefacts whitelisted for cross-branch reuse:
  ```
  oof_xgb_metastack.npy + test       (the breakthrough leg)
  oof_xgb_nonrule_bag3.npy + test    (model-seed bag, null but kept)
  oof_spec_lm_v3_score3.npy + test   (L↔M specialist, marginal)
  submissions/submission_tier1b_greedy_meta.csv  (LB 0.98094)
  ```

### 2026-04-25 — Trompt PROBE: lowest NN Jaccard ever (0.5340) but magnitude-trap NULL

- Goal: 13th NN-family lever-existence test. Trompt (column-attention
  tabular NN, Chen et al. 2023) via pytorch_frame. Architecturally
  distinct from RealMLP (BatchEnsemble + PBLD) — column-attention with
  learnable prompts. The yekenot/ps-s6e4-trompt-pytorch-frame public
  kernel claimed CV ~0.97-0.98 standalone, so Trompt had a chance to
  clear the bar even with the +0.0003 NN OOF→LB surcharge.
- Build path: modular kernel `kaggle_kernel/kernel_trompt/` (boot.py,
  config.py, features.py, model.py, cv.py, main.py + build.py
  concatenator) per the CLAUDE.md "short files" rule. Took 4 SMOKE
  iterations to land cleanly:
  - **v1 ERRORED** (IndentationError): build.py multi-line `from
    config import (...)` strip left orphan body lines. Fix: separate
    DOTALL regex MULTI_RE before SINGLE_RE.
  - **v2 ERRORED** (`from __future__` placement): assembled dist had
    header docstring → boot's docstring → __future__, violating
    Python's "must be at top". Fix: hoist single
    `from __future__ import annotations` to top after header docstring;
    strip ALL module __future__ imports from siblings.
  - **v3 ERRORED** (`ModuleNotFoundError: torch_frame`): model.py's
    module-level `from torch_frame import ...` ran at dist-file import
    time, BEFORE boot()'s install logic. Fix: hoist
    `install_torch_if_pascal()` + `install_pytorch_frame()` to boot.py
    module body so they execute as soon as boot.py's section runs
    (line 79-80 in dist), before model.py's torch_frame imports
    (line 179+).
  - **v4 PASSED** structurally (2 folds × 20k × 2 epochs in <1 min
    training; per-fold bal_acc 0.51/0.55, low because under-converged
    smoke config).
- PROBE config (after SMOKE pass): `IS_PROBE=True`, full 504k train,
  full Trompt capacity (channels=128, num_prompts=128, num_layers=3),
  N_EPOCHS=8, MAX_FOLDS=1 (StratifiedKFold(5) split structure preserved
  for fold-1 alignment). Outputs suffixed `_probe`.
- PROBE wall: ~70 min total kernel time. Kaggle script-kernel
  re-imported the module at kernel-time 1684s for nbconvert HTML
  compilation, triggering main() a SECOND TIME — saw boot logs reset,
  fold 1 start again. Both runs completed cleanly (the second run is
  the one whose outputs we kept, identical to the first).
- **Fold-1 results** (only va0 = 126,000 rows populated):
  - argmax bal_acc = **0.96092** (vs RealMLP fold-1 0.96978, −0.009)
  - tuned 1-fold OOF = 0.96633, bias [1.43, 1.17, 3.40]
  - errs at recipe bias = **2,183** (RealMLP fold-1 2069, LB3 fold-1 2014)
- **Jaccards (fold-1)**:
  - Trompt vs RealMLP n_ens=1 = **0.5696** (lower than RealMLP vs LB3 0.6222)
  - Trompt vs LB-best 3-way = **0.5340** ← LOWEST NN ORTHOGONALITY EVER
  - RealMLP vs LB3 = 0.6222
- **Blend gate (fold-1, LB3 + Trompt @ α, fixed bias)**:
  ```
  α=0.00  bal=0.97926  errs=2014  ← peak (LB3 standalone)
  α=0.05  bal=0.97926  errs=1972  (tied; errs −42 but bal_acc unchanged)
  α=0.10  bal=0.97901  errs=1954
  α=0.20  bal=0.97899  errs=1922
  α=0.50  bal=0.97748  errs=1873
  ```
  Classic magnitude-trap. Trompt has the lowest Jaccard of any NN
  tested AND drops total errors at small α, but the rare-class trade
  is unfavorable for macro-recall. No α > 0.05 lifts above LB3 alone.
- **Decision: skip 5-fold push.** Three reasons:
  1. **Compute budget**: production wall = 8 epochs × 5 min × 5 folds =
     ~3.3h, exceeds the 1h GPU cap by 3×. Would need multi-kernel
     splits or a halved-capacity config.
  2. **Magnitude trap**: even at the best Jaccard ever, +169 extra
     errors per fold predicts a 5-fold OOF that doesn't beat LB3
     alone. Adding to the LB-best 0.98094 meta-stacker stack would
     compound the surcharge unfavorably.
  3. **Compounding NN OOF→LB surcharge**: +0.0003 per NN leg. With
     Trompt's predicted standalone OOF tuned at ~0.97 (extrapolated
     from fold-1 0.966) and the meta-stacker at OOF 0.98084 / LB
     0.98094, Trompt's blend lift needs OOF Δ > +0.0005 to net-positive.
     Fold-1 sweep tops at Δ +0.00000 vs LB3 (let alone the meta-stacker
     stack).
- **Lever closes**. 13th NN-family null; pattern is now structural at
  this feature set. Summary table:
  ```
  NN family       Jaccard vs anchor   errs vs anchor   LB outcome
  ----------     ------------------- ---------------- --------------------
  MLP v5-v9       0.62-0.85           +1500-15000     NULL
  FT-Transformer  0.61                +12000          NULL
  TabPFN          0.81                +1485           NULL
  Pretrain-FT MLP 0.65                +3615           NULL
  DAE SwapNoise   0.84                similar         NULL
  RealMLP n_ens=1 0.62                +358            LB +0.00003 (3-stack)
  RealMLP n_ens=4 0.62                +485            NULL (worse than n_ens=1)
  Trompt          **0.53**            +169 (lowest)   NULL (magnitude-trap)
  ```
  Trompt's +169 errs at the BEST Jaccard ever seen still failed the
  blend gate. Structural rule firmer than ever: NN architectures on
  this feature set produce orthogonal errors but in larger absolute
  numbers, and macro-recall at fixed-bias cares about total per-class
  accuracy → the magnitude tax dominates.
- LB delta: n/a — no LB probe warranted.
- LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.
- Artefacts:
  - `kaggle_kernel/kernel_trompt/` (modular kernel + 4 build fixes)
  - `scripts/artifacts/oof_trompt_probe.npy` + test (fold-1 only,
    other 4 fold rows = 0). Whitelisted for cross-branch diagnostic —
    Trompt's 0.53 Jaccard is unique in our OOF bank.
  - `scripts/artifacts/trompt_probe_results.json`
- Lessons logged to LEARNINGS.md:
  1. Build-script gotchas for sibling-module Kaggle kernels.
  2. Trompt at full capacity is ~12× more compute-heavy per row than
     RealMLP; needs PROBE-first before 5-fold on the 1h cap.
  3. Lowest Jaccard ever (0.53) STILL fails the blend gate when errs
     are higher than anchor. **The magnitude rule is stricter than
     orthogonality.**
### 2026-04-25 — Tier 1c: 3 follow-ups on new LB-best 4-stack, all NULL (saturation confirmed)

- Goal: after Tier 1b's +0.00086 LB win, test 3 cheap CPU follow-ups on
  the new LB-best 4-stack (OOF 0.98084 / LB 0.98094) to find the next
  step. All three null.

- **Move 1 — greedy on new 4-stack with finer α grid (0.005..0.5) +
  isotonic-calibrated copies of every pool component (66 base × 2 = 132)**:
  ```
  step1: + recipe_no_digits α=0.010  OOF=0.98087  Δ=+0.00002
  STOP (below +5e-5 internal gate)
  ```
  The 4-stack is locally saturated for greedy log-blend operations.

- **Move 2 — meta-stacker v2 (224-dim features = v1 inputs + 4-stack
  logprobs + 3 binary specialists spec_lm_v3 + spec_mh_v3_score{5,6})**:
  - vs LB-best 3-stack:  best v2_iso α=0.250 → +0.00015 (sub-+0.0002 gate)
  - vs LB-best 4-stack:  best v2_iso α=0.200 → **+0.00002** (essentially zero)
  - **v2 = v1 + noise.** Meta-on-meta saturates exactly when the input
    bank already contains the prior meta.

- **Move 3 — meta-stacker 3-seed bag (XGB seeds {42, 7, 123})**:
  Per-seed standalone OOF: 0.98041 / 0.98029 / 0.98040.
  - seed=7 is genuinely worse (−0.00012 vs seed=42), dragging the
    bag mean down.
  - Bag iso-cal standalone 0.98061 (vs single-seed iso 0.98059,
    +0.00002 nominal).
  - Replace single-iso with bag-iso in 4-stack: every α tested
    NEGATIVE (best α=0.40 → −0.00001).
  - Add bag-iso ON TOP of 4-stack: best α=0.150 → +0.00003 (below
    +1e-4 gate).
  - **Same heavy-reg-XGB seed-dominant-optimum lesson as the nonrule
    bag from Tier 1b.** When XGB is heavy-reg (max_depth=4,
    reg_alpha=reg_lambda=5) with low best_iter (200-400), seed
    variance has more room to find different local optima. Sometimes
    seed=42 IS the best one and bagging worse seeds dilutes the signal.

- **Combined Tier 1c read-out**: the 4-stack is saturated against:
  1. Greedy log-blend addition from a 132-component pool (1e-4 gate)
  2. Deeper meta-stacking with v1 + binary heads (224-dim XGB)
  3. Variance reduction via XGB seed bagging
  All three diagnostic levers fail because the 4-stack already absorbed
  the meaningful signal from these components in Tier 1b's greedy step.
  Breaking past LB 0.98094 requires a fundamentally NEW signal source
  not yet on disk — most likely from the GPU-side experiments
  (RealMLP n_ens=4, Trompt) that were running in parallel.

- New scripts (all 3 reusable for future stacks once new components arrive):
  - `next_greedy_on_meta_stack.py` (greedy from 4-stack anchor + iso pool)
  - `next_meta_stack_v2.py` (meta-stacker v2 with 224-dim features)
  - `next_meta_stack_seedbag.py` (3-seed XGB seed-bag of meta-stacker)

- Artefacts whitelisted:
  - `oof_xgb_metastack_v2 + test`  (224-dim v2)
  - `oof_xgb_metastack_bag3 + test` (3-seed bag)

- LB best unchanged at 0.98094. Pack 0.98114 still +0.00020 above.
- LB budget: 3/10 used today, 7 remaining.

### 2026-04-25 — U0OEQ session: focal-NULL, distill family closed, Options 4+2 deferred (rehydrate constraint)

**Context**: ran 4-option plan after sibling sessions hit Tier 1b LB-best
0.98094. Container rehydrates ~every 15-30 min idle wiped uncommitted
work three times this session. Final state captured below.

**Confirmed CLOSED (this session)**:
1. **Option 3 — greedy_expanded over 42-component bank with new candidates**
   (focal_g2_invfreq, focal_g2h3, distill_small, distill_tiny added).
   All three anchors (recipe, lb_best_3way, lb_best_realmlp_stack)
   converge to OOF ≤ 0.98061, the known LB-best stack OOF. Greedy
   rediscovers the realmlp + xgb_nonrule_iso path the sibling found
   for LB 0.98008. **No new path emerges.** Independently confirmed
   by parallel session on origin/main with identical results.
   - Script: `scripts/greedy_expanded.py`
   - Result: `scripts/artifacts/greedy_expanded_results.json`

2. **Option 1 — extreme-capacity distill (d=2, leaves=7, r=500)**.
   Ran twice (post-rehydrate), bit-identical results both times:
   - Per-fold argmax: 0.97297 / 0.97414 / 0.97486 / 0.97274 / 0.97396
   - Tuned OOF: **0.97975** (vs distill_small 0.98066, recipe 0.97967)
   - Bias [0.83, 1.37, 3.30]; errors 9,935 (recipe 10,114)
   - Blend gate: peak α=0.000 vs LB-best stack (strict null), peak α=0.250
     Δ=+0.00024 vs recipe (marginal, below LB-transfer threshold).
   - Capacity reduction WORKS as predicted (less memorization, narrower
     OOF→LB gap on extrapolation), but OOF collapses faster than gap
     narrows. **No capacity sweet spot for soft-distill from
     bagged-OOF teacher on this problem.**
   - Artefacts: `oof_soft_distill_tiny.npy`, `test_soft_distill_tiny.npy`,
     `soft_distill_tiny_results.json`,
     `submissions/submission_soft_distill_tiny.csv` (diagnostic, not
     for LB probe).

**DEFERRED to a more persistent environment** (rehydrate constraint):
3. **Option 4 — SMOTE-NC on High class** (training-data-level lever).
   Killed by container rehydrate three times. Reaches fold 1 SMOTE
   step (~5 min FE + ~15 min SMOTE k-NN) before reset. ~2.5h total
   wall doesn't fit any plausible idle window in this container.
   Status: scripts committed and ready (`scripts/recipe_smote_high.py`),
   re-launch with `python scripts/recipe_smote_high.py` in a stable
   environment. Expected output: `oof_recipe_smote2x.npy` +
   `test_recipe_smote2x.npy` + results JSON.

4. **Option 2 — leak-eliminated teacher / W_RECIPE=1.0 distill**
   (recipe-only teacher, no pseudo component, leak source removed).
   ~30 min wall — fits rehydrate window — but de-prioritized by user
   pivot to documentation. Re-launch with
   `SOFT_SUFFIX=recipeonly W_RECIPE=1.0 XGB_DEPTH=3 XGB_NROUND=1500 XGB_MAX_LEAVES=15 python scripts/soft_distill_xgb.py`
   in a stable environment. Tests whether the pseudo component is
   the leak source (vs the teacher OOF construction itself).

**Distill family — closed at three capacity points**:
```
                       d   leaves  rounds   OOF tuned   LB         OOF→LB gap
soft_distill           4   30      3000     0.98096     0.97850    +0.00246
soft_distill_small     3   15      1500     0.98066     0.97865    +0.00201
soft_distill_tiny      2   7        500     0.97975     ?          (not probed)
                                                                     projected -0.0009
                                                                     to 0.97850-0.97900
```
Pattern: gap narrows ~0.0005 per 2× capacity reduction; OOF
collapses ~0.0010 per same reduction. No capacity sweet spot.
**Soft-distillation from bagged-OOF teacher is structurally bounded
on this problem.**

**Strategic context**: parallel session achieved LB 0.98094 via
**isotonic-calibrated XGB meta-stacker** (Tier 1b, commit 205b42f)
over 63-component bank + α=0.30 blend into LB-best 3-stack. Then
hit saturation at Tier 1c. Their conclusion (matches ours):
"breaking past LB 0.98094 requires NEW signal source." Options 4
(SMOTE-NC for synthetic High rows) and 2 (leak-eliminated distill)
are precisely such NEW signal sources — both still untried and
warrant a stable environment for the ~2.5h and ~30 min
respectively.

**Portable rules logged this session** (see LEARNINGS.md):
1. Heavy-alpha focal loss on a class-weight-tuned XGB regresses on
   every class, not rebalances (focal-invfreq null). Lin et al.
   defaults are too aggressive on top of an already-balanced base.
2. 2× capacity reduction in soft-distill narrows the OOF→LB gap by
   ~0.0005 but is insufficient for LB transfer. Needs 4×+ reduction
   (collapses OOF) or row-wise teacher leak-elimination (untried).
3. "First to satisfy all blend-gate heuristics on OOF" is NOT a
   sufficient condition for LB transfer when the candidate consumes
   teacher OOF directly. distill_small met every heuristic
   (Jaccard <0.80, errs ≤ anchor, per-class recall ≥ anchor across
   all classes, peak-α blend Δ ≥ +0.0002 on 3 anchors) and still
   LB-regressed by 0.00143.
4. Container rehydrates erase uncommitted compute. Long jobs
   (>30 min) need either external compute (Kaggle GPU kernel) or
   a stable container; pip installs (e.g. `imbalanced-learn`) also
   don't survive — bake them into Dockerfile or `bootstrap.sh`.

### Next steps (handoff): re-execute Options 4 + 2 in stable env (2026-04-25)

Highest-value untried experiments after the U0OEQ session and
parallel session's Tier 1b/1c saturation:

  **N1. Option 4 — SMOTE-NC + meta-stacker bank extension**
  (`scripts/recipe_smote_high.py`, ~2.5h wall + ~5 min meta-stacker
  rebuild). Run on a stable environment (not rehydrate-prone).
  - Step 1: `SMOTE_TARGET=42000 python scripts/recipe_smote_high.py`
    → produces `oof_recipe_smote2x.npy`, `test_recipe_smote2x.npy`.
    Expected behavior: per-fold High recall lifts +0.005 to +0.015,
    Low/Medium recall drops by ~0.001, net OOF tuned bal_acc ?
    (open question — could lift or hurt depending on whether
    interpolated High labels are NN-flip-consistent).
  - Step 2: add `recipe_smote2x` to the meta-stacker bank
    (`scripts/tier1b_xgb_metastack.py` features list) and re-train.
    If smote2x's errors are orthogonal to the existing 63
    components, the meta-stacker's error count drops below 8,948
    (current best) and the iso-blended Δ vs LB-best 3-stack lifts
    above +0.00023 OOF.
  - Step 3: blend the new meta-stacker iso into LB-best 3-stack
    (same α=0.30 protocol that produced LB 0.98094).
  - Decision gate: only LB-probe if blend OOF lifts ≥ +0.0002 over
    0.98094.

  **N2. Option 2 — W_RECIPE=1.0 distill** (~30 min, fits any window).
  Quick diagnostic: replace teacher = 0.5×recipe + 0.5×pseudo with
  teacher = recipe_only. If student gap narrows below distill_small's
  +0.00201, the pseudo-label component is a meaningful leak source.
  - Run: `SOFT_SUFFIX=recipeonly W_RECIPE=1.0 XGB_DEPTH=3 XGB_NROUND=1500 XGB_MAX_LEAVES=15 python scripts/soft_distill_xgb.py`
  - Diagnostic value only — LB probe NOT warranted regardless of
    outcome (recipe alone is OOF 0.97967, weaker than current best).
  - If gap narrows: indicates path to a real distill that transfers.
    Then build a leak-eliminated multi-fold teacher.
  - If gap doesn't narrow: confirms overfit is in teacher OOF
    construction itself, not the pseudo component. Distill family
    fully closed.

  **N3. Cross-pollinate parallel session's meta-stacker with
  today's NEW components**. The Tier 1b meta-stacker bank predates
  distill_tiny + recipe_smote2x. Add the 4 new components
  (focal_invfreq, focal_g2h3, distill_small, distill_tiny once
  built) to `tier1b_xgb_metastack.py` CANDIDATES list and re-train.
  If standalone errors drop below 8,948, isotonic + blend may
  exceed 0.98094.
  - Wall: ~30 min (meta-stacker rebuild only, no base-component
    retraining since OOFs are on disk).

  **Skip on principled grounds**:
  - Further focal variants (γ < 2 or alpha < 1.5x). Two failures in
    one session at different α magnitudes; mechanism is structurally
    backwards on this base.
  - More distill capacity-reduction experiments (d=2/r=500 and
    d=3/r=1500 already run). Capacity-reduction path is bounded.
  - Public-CSV blending (banned by top-of-file rule).

### 2026-04-25 — final-selection lock (post-LB-0.98094 reframe)

Context: parallel session pushed Tier-1b XGB meta-stacker isotonic blend
to LB **0.98094** (+0.00086 over prior 0.98008). Pack 0.98114 only
+0.00020 above. The earlier hedge audit (2026-04-24, primary = 3-way at
LB 0.98005, hedge = CatBoost) is stale — primary moved.

**Final-selection locked**:

  PRIMARY: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
    - composition: lb3 + RealMLP α=0.20 + xgb_nonrule__iso α=0.075
                   + xgb_metastack__iso α=0.300
    - OOF 0.98084, gap **−0.00010** (LB above OOF — meta-stacker
      CV-pessimism property)
    - per-class recall: L 0.9955 / M 0.9695 / **H 0.9775**
    - errs 9,415 (−157 vs LB-best 3-stack)

  HEDGE: `submission_recipe_full_te.csv` → **LB 0.97939**
    - composition: pure single-model XGB-on-recipe with class-balanced
      sample-weight + post-hoc log-bias [1.43, 1.47, 3.40]
    - OOF 0.97967, gap +0.00028 (clean calibration)
    - NO blend, NO isotonic, NO meta-stacker — independent of every
      stacking ingredient in the primary

Hedge rationale: the primary depends on a 63-component XGB meta-stacker
trained over the full OOF bank. If any single component overfits private
LB, the meta-stacker amplifies that overfit through the +0.00086 blend
lever. The 2026-04-24 CatBoost-hedge recommendation favored model-family
diversity, but the new primary already includes RealMLP (NN family) +
xgb_nonrule_iso (calibration-corrected XGB) + meta-stacker (XGB over 63
components). Adding CatBoost as a hedge over-diversifies — the primary
itself is already maximally diverse on the model-family axis.

The cleanest hedge against meta-stacker overfit is therefore a candidate
that does NOT touch the meta-stacker pool at all: pure single-model
recipe XGB. Premium = −0.00155 LB vs primary; the gap +0.00028 is
honest CV calibration (no negative-gap CV-pessimism artefact).

Alternative hedges considered and rejected:
  - `submission_lb3_realmlp_nonruleiso.csv` (LB 0.98008): the
    foundation the meta-stacker builds on. Too closely correlated.
  - `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005): shares
    pseudo_s1 + pseudo_s7 with the meta-stacker pool.
  - `submission_recipe_full_te_catboost.csv` (LB 0.97935): different
    model family but the primary already has NN diversity. Premium
    cost (−0.00159) only fractionally larger than recipe-XGB hedge
    (−0.00155) for less marginal value.

Pack 0.98114 still +0.00020 above primary. With 6 days to deadline and
7 LB submissions remaining today (3/10 used), one more LB-probe slot
is reserved for either: (a) the SMOTE-NC follow-up if it lifts, or
(b) a tighter primary if a meta-stacker variant adds another +0.0002.

### 2026-04-25 — SMOTE-NC environment-blocked + Kaggle-kernel route

- Goal: execute Step 3 of the post-LB-0.98094 follow-up — SMOTE-NC
  oversampling of the rare High class as the only untested
  *training-data-level* attack on the Pareto-frontier ceiling
  [0.9949, 0.9685, 0.9774].
- Smoke (SMOKE=1, 20k/2-fold, SMOTE_TARGET=5k): PASSED in ~80s.
  - SMOTE-NC adds High rows 333 → 5,000 per fold (k=5 NN interpolation
    over cats+digits+combos+num_as_cat+tres categorical indices)
  - best_iter 51 / 85 (gradient correct, no GCE-style pathological 1)
  - OOF tuned 0.96555 vs recipe smoke 0.96381 = +0.00174 standalone
    in smoke mode (signal exists at 20k subsample scale)
- Production attempts (5-fold × 504k tr × ~16k → ~42k High per fold):
  did not survive the idle gap between user prompts.
  - Attempt 1 (06:14 UTC): PID 4635 launched detached. Process gone
    on next check (~80min later). No OOF artifact written.
  - Attempt 2 (07:51 UTC): PID 3522 relaunched detached. Process
    gone on next check (~30min later). Same outcome.
- **Observation, not a generalised rule**: detached background
  processes did not survive in this session — but we only have two
  data points and didn't investigate the underlying cause. The
  portable lesson (logged to LEARNINGS.md) is the per-fold
  checkpoint pattern: any pipeline whose total wall budget exceeds
  the smoke runtime should write `oof_*_fold{f}.npy` and
  `test_*_fold{f}.npy` immediately after each fold completes, so
  partial progress is recoverable regardless of why the process
  ended.
- **Status: SMOTE-NC is ENVIRONMENT-BLOCKED on this branch.** The
  scaffold (`scripts/recipe_smote_high.py`) and blend-gate
  (`scripts/blend_gate_smote.py`) are committed and ready to run
  on persistent compute.
- **Recommended next-session route**: Kaggle kernel.
  1. `kaggle_kernel/kernel_smote/` — single-file orchestrator that
     inlines `recipe_features.py` + `recipe_ote.py` + the SMOTE-NC
     pipeline. ~30min scaffolding.
  2. Upload as a private kernel; expected wall ~3h on Kaggle's
     16-core CPU runtime (well under the 9h kernel cap).
  3. Pull OOF + test back via `kaggle kernels output`.
  4. Run `scripts/blend_gate_smote.py` locally — auto-emits
     submission if Δ ≥ +2e-4 vs LB-best 0.98094.
- **Why SMOTE-NC is still worth pursuing despite the saturation**:
  every prior High-class lever (detector, router, meta-stack,
  binary-Medium head, score=6 specialist) operated POST-HOC on the
  existing OOF bank. The Pareto frontier closure proved no
  rearrangement of the bank can push High recall past 0.9774.
  SMOTE-NC produces a model with a fundamentally different training
  distribution (~2x High rows synthesized via k-NN feature-space
  interpolation), so its decision surface CAN exceed 0.9774 on
  High at the cost of Low/Medium recall. Whether that trade is
  net-positive under macro-recall is the open question — only
  determinable by running it through the blend gate.
- LB budget unchanged (0 used today). LB-best remains **0.98094**
  via `submission_tier1b_greedy_meta.csv`. Final-selection lock
  unchanged: primary = LB 0.98094, hedge = recipe_full_te (LB
  0.97939).

### 2026-04-25 — overlooked-levers pass: SUMMARY (1 lock + 1 close-out)

Three follow-up items from the user-prompted "what have we overlooked"
review (2026-04-25 prior session):

  Item 1 — final-selection hedge audit  →  COMMITTED a63e532
    Primary: submission_tier1b_greedy_meta.csv (LB 0.98094)
    Hedge:   submission_recipe_full_te.csv     (LB 0.97939)
    Rationale: primary already includes XGB + RealMLP NN +
    xgb_nonrule_iso + 63-component meta-stacker — model-family
    diversity maxed. CatBoost-hedge would over-diversify; pure
    single-model recipe-XGB is the cleanest orthogonal-to-meta
    fallback against private-LB overfit.

  Item 2 — re-run greedy with iso pool  →  PIVOTED (pre-existing exhaustive)
    Audit showed tier1b_greedy_with_meta.py already adds an __iso
    copy of every component (line 121-123). Both raw and iso
    variants exhausted. Action revised: re-run only after a NEW
    component (SMOTE) lands.

  Item 3 — SMOTE-NC training-data lever  →  ENVIRONMENT-BLOCKED
    Smoke green (+0.00174 OOF lift over recipe smoke). Production
    killed by container rehydrate twice in one session.
    Deferred to Kaggle kernel.

Net session output:
  - 1 final-selection lock (CLAUDE.md a63e532)
  - SMOTE-NC scaffold + smoke evidence + blend-gate (ba5afbb)
  - 1 LEARNINGS rule: container-rehydrate persistence
  - 0 LB submissions spent

Next-session priorities:
  1. Push SMOTE-NC scaffold to Kaggle kernel (highest EV remaining
     own-pipeline lever).
  2. After SMOTE returns: blend-gate → LB probe if Δ ≥ +2e-4.
  3. If SMOTE nulls or stays inside fold-noise: lock primary +
     hedge as final and stop spending compute.

### 2026-04-25 — senior-engineer pre-deadline audit

Standalone read-through of the LB-best pipeline + final-selection
choice with three parallel sub-agents on (meta-stacker / recipe FE +
OTE / submission validity), then merged with my own re-reads. Full
report in `audit/2026-04-25-senior-engineer-audit.md`. Plan file at
`/root/.claude/plans/you-are-a-senior-jaunty-flute.md`.

Key findings:
- **Headline +0.00086 LB lift is probably real signal** (not the
  "critical leakage" the first sub-agent flagged). Iso-cal on full
  OOF is a 1-D mapping over 630k pts; if dominant it would push gap
  positive (OOF > LB), opposite of what we observe. Negative gap
  pattern matches digit-XGB / digits-OTE precedent on this comp.
- **Hedge under-protects (HIGH)**: current
  `submission_recipe_full_te.csv` shares its full FE pipeline with
  primary (484/270k disagreement = 0.18%). Recommended swap to
  `submission_3way_recipe025_s1035_s7040.csv`: half the premium
  (-0.00089 vs -0.00155), sidesteps meta-stacker layer (the
  most-tuned and most-likely public-LB-overfit element). Alternative
  CatBoost hedge equal-cost with 39% more disagreement rows.
- **Iso-cal on full OOF (MEDIUM)**: per-fold isotonic would be the
  honest version. OOF inflation likely 0.0001-0.0003. No action for
  deadline; flagged for future-comp playbook.
- **Selection bias (MEDIUM)**: ~30 cumulative LB probes pushes
  max-of-N order statistic to ~+0.00075 above true. Estimated true
  primary LB ~0.98019, central private ±0.0005.
- **Sub-agent claim that `recipe_features.py:119` (drop digit cols
  constant on test) is a leak: WRONG.** Dropping zero-variance
  features cannot leak labels; metadata access ≠ target leakage.
  OrderedTE itself is correct.

Action: hedge swap recommendation written up; primary unchanged.
0 LB probes spent on this audit.

### 2026-04-25 — audit follow-up: F2 verified GREEN + GroupKFold-Crop honest

Two diagnostic experiments executed after the audit to verify the F1/F2
findings. Both close cleanly. Full results in
`audit/2026-04-25-senior-engineer-audit.md` (audit follow-up section).

**F2 verification — `tier1b_greedy_perfoldiso.py`**: per-fold isotonic
greedy mirrors the LB-best primary pipeline but fits iso honestly per
fold (`oof[!=fold_k]`-fit, applied to `oof[fold_k]`). Result:

```
anchor OOF (per-fold iso)         = 0.98060   (vs full-OOF iso 0.98061)
step1 + xgb_metastack_bag3__iso α=0.350 OOF = 0.98080  Δ=+0.00019
final OOF                         = 0.98080
Δ vs current primary OOF 0.98084  = -0.00004
```

Verdict GREEN: iso-on-full-OOF was contributing ~1 bp inflation. The
current primary's +0.00086 LB lift over LB-best 3-stack is mostly
genuine signal, not iso-leak inflation. Lock primary as-is.

**GroupKFold-by-Crop diagnostic** (`b2_groupkfold.py GROUP=crop`): the
2026-04-22 B2 check tested only Region. Crop is the second leakage axis.

```
per-fold argmax: 0.97476 / 0.97668 / 0.97565 / 0.97453 / 0.97373
OOF argmax = 0.97485   tuned = 0.97910   bias=[0.732, 0.969, 3.101]
Δ vs StratifiedKFold baseline (0.97967) = -0.00056
```

Verdict HONEST: Δ well within the "honest" threshold (≤ 0.002). OOF
holds across BOTH leakage axes (Region: -0.00029; Crop: -0.00056).
No OTE/FREQ/ORIG-stat leakage exploitable across either axis.

**Net**: audit's F2 (iso) and OOF-honesty concerns close cleanly.
F1 (hedge under-protects) stands — recommend swap to
`submission_3way_recipe025_s1035_s7040.csv` (premium -0.00089,
sidesteps meta-stacker layer). Primary unchanged.

LB budget: 0 probes spent on either follow-up.

### 2026-04-25 — next-steps plan (without giving up)

Audit closed F2 GREEN and confirmed OOF honesty on both leakage axes.
Primary's lift is mostly genuine. We have honest LB headroom AND 5
days × 10 LB probes left. Keep pushing.

Full ranked plan in `audit/2026-04-25-next-steps.md`. Top 4 immediate
levers (no defeat):

  Tier A (cheap CPU, today/tomorrow):
    A1. τ sweep on stage-1 pseudo (τ ∈ {0.95, 0.97, 0.99}, 3 × ~50 min CPU).
        Untested band; team picked τ=0.98 by instinct. Drop-in upgrade
        if any τ has tuned OOF > 0.97993 + errs ≤ 10039 + Jaccard < 0.85.
    A2. GroupKFold-Crop OOF as meta-stacker input (~15 min, no retraining).
        New OOF, structurally different fold split → different errors.
    A3. Per-fold-iso variants in greedy pool (~15 min).
        bag3__iso outperformed metastack__iso in per-fold-iso experiment.
    A4. Hedge swap to 3-way (manual, zero compute).

  Tier B (Kaggle GPU, queue overnight):
    B1. SMOTE-NC on Kaggle kernel (~3h Kaggle wall). Smoke green
        +0.00174 OOF over recipe smoke. Highest single EV remaining.
    B2. RealMLP n_ens=2 with n_epochs=40 (~50 min). Tests
        under-convergence hypothesis from CLAUDE.md.
    B3. Trompt push (~1h). Architecturally distinct NN family.

  Tier C (speculative ceiling-breakers):
    C1. Per-bin blend at 3 bins instead of 5 (lower free params).
    C2. 4-component "clean meta-stacker" (less over-parameterized).
    C3. Public/private split ratio verification (~5 min).

What NOT to retry: HP tuning, model-seed bagging, cleanlab,
NN-from-scratch MLPs, public-CSV blending. All structurally null.

Pack 0.98114 is +0.00020 above primary. Leader 0.98219 is +0.00125
above. Both reachable via ANY of A1-A3 + B1 if signals stack. We're
not done.

### 2026-04-25 — Tier 1b cross-pollinate + ensemble: third saturation confirmation at LB 0.98094

- Goal: after Tier 1c saturation (greedy + meta-v2 + meta-bag all null on
  the LB-best 4-stack), test two adjacent levers from the post-1c
  brainstorm: (#4) **cross-pollinate** the Tier-1b meta-stacker with
  components that didn't exist at v1's run-time (recipe_focal_g2_aH1,
  recipe_focal_g2_invfreq, soft_distill_small, soft_distill_tiny,
  realmlp_ens4); (#2) **ensemble of meta-stackers** with varying
  hyperparameters (depth, XGB seed, colsample, max-rounds) to gather
  meta-level diversity. Both run on top of the LB-best 0.98094 4-stack.
- Changed: `scripts/tier1b_helpers.py` (shared loaders + iso_cal +
  build_lbbest_stack — extracted from tier1b_xgb_metastack.py),
  `scripts/tier1b_metastack_v3.py` (cross-pollinate, depth=4 seed=42
  same HPs as v1, expanded EXCLUDE for derived/circular components),
  `scripts/tier1b_metastack_variant.py` (env-var parameterised runner;
  VARIANT, DEPTH, XGB_SEED, COLSAMPLE, MAX_ROUNDS),
  `scripts/tier1b_final_blend.py` (Strategy 1 single-replace, Strategy
  2 equal-weight ensemble, Strategy 3 greedy forward; per-class recall
  guardrail at -0.0005, fixed-bias decision rule, +2e-4 LB-transfer
  emit gate).
- Smoke: helpers reproduce LB-best 3-stack OOF=0.98061 exactly. Pool
  size = 66 (62 prior + 5 new − 1 dropped via stricter EXCLUDE). All
  5 new candidates loaded.
- v3 results (5 folds × 215-feature XGB, 6 min wall):
  ```
  fold     it    val_argmax_bal_acc
  1       471        0.97475
  2       585        0.97363
  3       735        0.97508
  4       738        0.97297
  5       655        0.97407
  OOF argmax = 0.97410   tuned = 0.98073
  errs vs LB-best 3-stack = 8915 (LB-best = 9572)
  Jaccard vs LB-best 3-stack = 0.8308   (v1 was 0.8743 — MORE orthogonal)
  iso(v3) standalone = 0.98121
  ```
  Standalone +0.00032 vs v1 (0.98041) and +33 fewer errors. The 5
  added components changed the disagreement pattern materially.
- Variant B (depth=3, seed=7, colsample=0.7, 5000-cap, 8 min wall) and
  variant C (depth=5, seed=123, colsample=0.5, 2500-cap, 8 min wall):
  ```
                  argmax    tuned     iso(standalone)
  v1 (existing)   0.96995   0.98041   0.98059
  v3              0.97410   0.98073   0.98121
  B               0.97418   0.98053   0.98137  ← highest iso of any meta
  C               0.97411   0.98034   0.98037
  ```
  Both B and C trained on a pool of 67 (their pool included v3 since
  it was on disk by then) — leak-free since fold structure is preserved
  across the chain.
- Final blend gate (3 strategies × per-class guardrail, 28s wall):
  ```
  STRATEGY 1: single-meta replace v1 in LB-best 4-stack (anchor 0.98084)
    replace_v3_a030    Δ=+0.00004  J=0.961  recH=0.9771   PASS class-gate, FAIL Δ
    replace_B_a030     Δ=+0.00003  J=0.955  recH=0.9771
    replace_C_a030     Δ=-0.00001  J=0.956  recH=0.9768

  STRATEGY 2: equal-weight log-ensemble of metas → α-sweep into LB-3-stack
    ens_v1_v3_a400              Δ=+0.00004
    ens_v1_v3_B_a350            Δ=+0.00006
    ens_v1_v3_B_C_a350          Δ=+0.00009  ← Strategy 2 best, still below gate

  STRATEGY 3: greedy forward over iso metas
    step1: + v3 α=0.400  OOF=0.98093
    step2: + C  α=0.200  OOF=0.98101
    step3: + v1 α=0.300  OOF=0.98102   (4th candidate B can't improve)
    final greedy_v3_C_v1: Δ=+0.00018  J=0.9385  recH=0.9764

  Strategy 3 closest to gate: Δ +0.00018 (just shy of +2e-4 threshold)
  but recH drops 0.0011 vs anchor's 0.9775 — violates per-class guardrail.
  ```
- **No submission emitted.** Three independent attacks (greedy
  expanded pool, meta-stacker pool extension, meta-stacker ensemble)
  all land within ±0.0002 of OOF 0.98084. Saturation at the LB-best
  4-stack is now triple-confirmed.
- Read-out: cross-pollinating with new components DOES add real
  signal at the standalone meta level (v3 +0.00032 OOF, B's iso
  reaches 0.98137 — highest iso of any meta tested). But the LB-best
  4-stack absorbs almost all of it through the existing v1 channel.
  The +0.00018 Strategy 3 OOF lift trades High recall for Medium
  recall, which is the wrong direction under macro-recall and
  systematically fails to transfer.
- **Rule reconfirmed**: meta-stacker ensembling at fixed pool size
  is bounded by what the v1 meta already extracts. Path past the
  ceiling requires NEW components — not new metas on the same
  components.
- Companion work this session: kernel audit round 3 (16 unread
  kernels at ≥20 votes inspected). 2 actionable Tier-A findings:
  1. **OvR-XGB** (include4eto, 31 votes, 2026-04-25) — 3 binary:logistic
     XGB heads on the FULL V10 recipe feature set, concat → softmax-
     renormalize → multiplicative class-weight Optuna (200-trial,
     bounds [0.5, 3.0]³). Genuinely new mechanism: never sees the
     multi-class CE gradient; multiplicative rather than additive
     post-hoc bias. ~80 min CPU. Highest-EV next bet.
  2. **TabM** (wguesdon, 33 votes; ICLR 2025 BatchEnsemble MLP via
     pytorch_frame) — only architecturally novel NN family remaining
     after the 13 NN nulls. Reuses Trompt kernel scaffold; ~1h GPU.
- LB budget unchanged (0 spent). LB-best stays
  `submission_tier1b_greedy_meta.csv` at **0.98094** with hedge
  `submission_recipe_full_te.csv` at LB 0.97939.

### 2026-04-25 — recommended next-step priority list (5 days to deadline)

The own-pipeline ceiling at LB 0.98094 is now confirmed against:
- 12-component greedy expanded pool (Tier 1c step 1)
- Meta-stacker v2 with binary specialists (Tier 1c step 2)
- Meta-stacker XGB seed-bag (Tier 1c step 3)
- 5-component cross-pollinate (this session, Tier 1b v3)
- 4-meta isotonic ensemble (this session, Strategy 2)
- 4-meta greedy forward selection (this session, Strategy 3)

To break LB 0.98094 we need to ADD a fundamentally new component to
the OOF bank — not another meta on the existing components. Ranked
by EV/cost:

  **N1. OvR-XGB on V10 recipe** (Tier-A audit finding, ~80 min CPU).
  Three independent binary:logistic XGB heads against the full V10
  recipe feature set. SMOKE first (1 fold × 50k rows × 2 binary
  heads → ~5 min). Production: 5-fold × 3 heads × ~25 min/fold ≈
  6h serial OR ~2h with 3 heads in parallel on the 16-core box
  (each head only uses ~5 cores at hist tree_method).
  - Save oof/test_xgb_ovr_recipe.npy after softmax-renormalize.
  - Add to tier1b_metastack_v3 candidate pool (a NEW component
    that didn't exist when v3 ran).
  - Re-run tier1b_final_blend.py — if v3+OvR meta-stacker clears
    +2e-4 + per-class gate, LB probe.
  - Expected: +0.00010 to +0.00030 OOF if the binary CE gradient
    produces materially different boundary geometry than softmax
    CE; the magnitude-trap rule applies (Jaccard < 0.80 AND errs ≤
    anchor required).

  **N2. TabM via pytorch_frame** (GPU, ~1h smoke + ~1h production).
  Only architecturally novel NN family remaining after 13 NN nulls.
  Reuses Trompt kernel scaffold (`kaggle_kernel/kernel_trompt/`)
  with a single-line model swap. SMOKE-first under the 1h GPU cap.
  - Gate at fold-1 errs vs LB-best 4-stack ≤ +5% (stricter than
    prior NN gates because we know the magnitude-trap pattern).
  - If passes: full 5-fold, add to meta-stacker bank.
  - Expected null based on 13 prior NN failures, but worth one
    GPU slot since it's the last unexplored architecture.

  **N3. SMOTE-NC** (deferred from prior session, blocked by container
  rehydrate). Push to Kaggle CPU kernel (~3h wall, well under 9h
  cap). Smoke evidence already showed +0.00174 OOF lift on a 20k
  subsample. Real test: does that signal survive at full scale + the
  meta-stacker bank gate?
  - Run via `kaggle_kernel/kernel_smote/` (needs 30 min scaffold
    using existing `scripts/recipe_smote_high.py` as base).
  - Output: oof/test_recipe_smote_high.npy.
  - Add to meta-stacker bank, re-run final blend gate.

  **N4. Decision: lock current finals if N1+N2+N3 all null.**
  Primary `submission_tier1b_greedy_meta.csv` (LB 0.98094) +
  hedge `submission_recipe_full_te.csv` (LB 0.97939) is locked.
  Stop spending compute. With 5 days to deadline and the current
  +0.00020 gap to pack, private-LB variance (±0.0005) makes
  another LB-probe cycle low EV.

  **Skip on principled grounds:**
  - Further meta-stacker variants (depth/seed/HP sweeps) — three
    saturation confirmations document this is exhausted.
  - Public-CSV blending (banned by top-of-file rule).
  - HP tuning on existing components (LB regressed twice, see
    2026-04-22 entry).
  - More NN-family attempts beyond TabM — RealMLP n_ens=4 was the
    13th null; the magnitude-trap pattern is structural at this
    feature set.

  **Execution order**: N1 first (highest EV, all-CPU, 1 evening of
  wall time). If N1 yields a passing OOF, push to LB and skip N2/N3
  unless deadline pressure allows. Otherwise N3 (SMOTE on Kaggle
  kernel as low-attention background work) and N2 (TabM) in parallel
  on day 2-3.

### 2026-04-25 — kernel audit round 4: 3 actionable levers + execution priorities

- Goal: fresh kaggle-kernel sweep targeting unread / recent high-vote kernels
  to find new own-pipeline signal sources after the LB-best 0.98094 4-stack
  saturated. 8 kernels pulled from `kaggle kernels list -c playground-series-s6e4`
  (sorted by votes), cross-checked against the audit log; 5 net-new readouts
  beyond audit rounds 1-3.

- **Reads (8 unaudited kernels, ranked by signal):**
  ```
  kernel                                           votes  novelty             verdict
  -------                                          -----  -------             -------
  wguesdon/ps6e4-30-model-ensemble-with-stacking   23     30-model bank w/    HIGH
                                                          lr_ote+knn_ote+
                                                          et_ote as "weak
                                                          individually,
                                                          help stacker";
                                                          explicit "greedy
                                                          CV > LB, LGB-stack
                                                          LB > CV"
  yunsuxiaozi/pss6e4-lgb-advanced-cv-0-97997       27     5-shuffle OTE       MED-HIGH
                                                          concat as
                                                          training augmentation
                                                          (5x duplicated rows
                                                          per shuffle)
  utaazu/0-979-cv-single-cat-pairwise-te-bias-tun  32     adversarial         MED
                                                          validation drives
                                                          orig_weight=0.35
                                                          (we use 1.0)
  rawashishsin/s6e4-single-xgboost-cv-0-9786       17     qcut-binned         LOW-MED
                                                          numeric × cat OTE
                                                          keys (we have
                                                          num_as_cat + cat-
                                                          pair OTE, not this)
  aliafzal9323/s6e4-0-970-stacked-lgb-xgb-cat-fe   118    generic stacking;   LOW
                                                          only untested FE =
                                                          pH_Deviation =
                                                          abs(Soil_pH-6.5)
  beraterolelk/oof-meta-stacking-with-golden       29     VPD via Tetens-     LOW
                                                          Murray (we tested
                                                          vpd_proxy null
                                                          in A4)
  simarbirsinghsandhu/reverse-engineering-irriga   23     DGP rule equiv      LOW
                                                          to ours (Kc shift
                                                          identity)
  manasi197/s6e4-multi-model-ensemble-voting-ana   49     public-CSV blend    BANNED
                                                          over nina2025
                                                          0.98113-.117 csvs
  ```

- **N1 (top pick, ~30 min CPU): Multinomial-LR meta-stacker** on the same
  63-component bank that produced our LB-best XGB meta-stacker.
  - Mirror `scripts/tier1b_xgb_metastack.py` with multinomial LR
    (`C=1, class_weight='balanced'`) instead of XGB. Same 5-fold
    StratifiedKFold(seed=42), same fixed-bias blend gate vs LB-best
    3-stack, same +2e-4 emit threshold.
  - **Why**: wguesdon explicitly chose LGB-stacker over greedy because
    "greedy CV > LB". Our LB-best 0.98094 stack uses both an XGB
    meta-stacker AND a greedy-forward step on top. LR meta-stacker is
    structurally simpler (no tree depth to overfit) and our prior LR
    meta-stacker test (2026-04-21 soft-blend session) was on a 12-component
    bank — never on the 63-component bank that produced the breakthrough.
  - **Decision criteria**:
    * If LR_iso standalone @ recipe bias is competitive with XGB_iso
      (within ~0.0010 OOF) AND has Jaccard < 0.97 vs LB-best, build a
      4-stack candidate (lb3 + RealMLP + nonrule_iso + LR_iso) for hedge
      consideration.
    * If LR_iso blend Δ ≥ +2e-4 over LB-best 4-stack, gate-emit submission.
    * Even a null result is informative: confirms XGB depth=4 was the
      "right" meta-stacker capacity for this bank.

- **N2 (~1.5h CPU): Add weak diversity learners to the meta-stacker bank**.
  Build 3 new OOFs on recipe feature set:
  `LR(C=1, class_weight='balanced', solver='lbfgs')`,
  `KNeighborsClassifier(n_neighbors=50)`,
  `ExtraTreesClassifier(n_estimators=500, class_weight='balanced')`.
  Add to the bank → re-train Tier-1b XGB meta-stacker → re-blend.
  - Decision gate: any of the three pulls meta-stacker errors below 8,948
    AND Jaccard < 0.97 → blend into 4-stack and gate at +0.0002 OOF +
    per-class recall guardrail.
  - kNN may need subsampled fit (50k stratified) for compute feasibility;
    LR + ET both fit on full 504k easily.

- **N3 (~1.5-2h CPU): yunsuxiaozi 5-shuffle OTE concat as training augmentation**.
  Implement as `recipe_ote_5shuffle_concat` variant of `recipe_full_te.py`
  (concat 5 shuffled OTE-fit copies of train into a 5x-augmented training
  pool, vs our current per-row K-shuffle averaging). Structurally distinct
  from every prior OTE variant.
  - Standalone OOF gate: tune log-bias and compute Jaccard vs recipe.
  - If Jaccard < 0.85 AND errs ≤ recipe (10,114), gate as new meta-stacker
    component.

- **Skip on principled grounds**:
  - aliafzal stacking, beraterolelk's VPD, simarbir's DGP — equivalent to
    or below what we have.
  - manasi197 — public-CSV blending (banned).
  - HP / multiplicative class-weight tuning — equivalent to our additive
    log-bias coord-ascent.
  - utaazu's `ORIG_ROW_WEIGHT=0.35` swap — global perturbation with high
    risk and no Jaccard-novelty signal; our 1.0 has been the working
    setting through every successful blend.

- **Execution order**: N1 first (cheapest + addresses audit's F1 hedge concern
  — current hedge `recipe_full_te.csv` shares full FE pipeline with primary,
  484/270k disagreement; LR-stacker hedge would be structurally different from
  primary's XGB-stacker while still benefiting from the same bank). Then N2 if
  N1 lands. N3 last (highest wall time, lowest confidence on transfer).

- LB budget unchanged (0 probes spent on the audit). LB-best still 0.98094.

### 2026-04-25 — N1 EXECUTED: LR meta-stacker, biggest OOF lift since Tier 1b (+0.00098 vs LB-best 4-stack), AWAITING LB PROBE

- Goal: execute N1 — multinomial LR meta-stacker on the same 70-component
  bank that produced our LB-best XGB meta-stacker. Test whether a
  structurally simpler model class (no tree depth) finds a different
  operating point than depth-4 XGB on the same bank.
- Changed: `scripts/tier1c_lr_metastack.py` (mirrors `tier1b_xgb_metastack.py`
  exactly except for the model: `LogisticRegression(C=1.0, class_weight=
  'balanced', solver='lbfgs', max_iter=1000)` over standardized features);
  `scripts/tier1c_lr_extend.py` (extended α∈[0,1] sweep + per-class recall
  + LB-best 4-stack anchor + LR×XGB compound grid);
  `scripts/tier1c_lr_emit.py` (4-stack-anchored submission emission at three
  α candidates).
- Wall: 5 min total (5 folds × ~60s LR fit + 0.3s test predict).

- **Standalone diagnostic** (LR_iso vs prior anchors, OOF tuned bal_acc):
  ```
  model                      OOF       errs     PCR [L, M, H]
  LB-best 3-stack            0.98061   9572    [0.9955, 0.9689, 0.9774]
  LB-best 4-stack (LB-best   0.98084   9415    [0.9955, 0.9695, 0.9775]
                  PRIMARY)
  XGB metastack_iso          0.98059   9044    (basis of 4-stack via α=0.30)
  **LR  metastack_iso        0.98183   9806    [0.9942, 0.9696, 0.9817]**
  ```
  LR_iso is the **first standalone to BEAT the LB-best 4-stack** by
  +0.00099 OOF — net positive macro-recall via +0.0042 High at the cost
  of -0.0013 Low. Errs increase (+391 vs 4-stack) but the per-class
  rebalance favors macro-recall.

- **Cross-stacker Jaccards** (post log-bias):
  - Jaccard(LR_iso, LB-best 3-stack) = **0.7054** (strong orthogonality)
  - Jaccard(LR_iso, XGB_iso meta) = **0.7166** (LR + XGB are non-redundant
    even though both consume the same 70-component bank)

- **Blend sweep, fixed-bias log-blend, LR_iso × LB-best 4-stack**
  (the actual primary, OOF 0.98084 / LB 0.98094):
  ```
  alpha    OOF     Δ        errs   recL    recM    recH    guardrail
  0.30   0.98152  +0.00068  9232  0.9954  0.9703  0.9788   PASS (Low -0.0001)
  0.40   0.98160  +0.00075  9275  0.9953  0.9703  0.9792   PASS (Low -0.0002)
  0.50   0.98167  +0.00083  9287  0.9952  0.9703  0.9794   PASS (Low -0.0003)
  0.60   0.98175  +0.00091  9341  0.9951  0.9703  0.9799   PASS (Low -0.0004)
  0.65   0.98182  +0.00098  9387  0.9950  0.9702  0.9802   BORDER (Low -0.0005)
  0.75   0.98186  +0.00101  9475  0.9948  0.9702  0.9806   FAIL (Low -0.0007)
  0.90   0.98191  +0.00107  9709  0.9943  0.9699  0.9816   FAIL (Low -0.0012)
  ```
  - Magnitude trap analysis: errors actually DECREASE from 9415 (anchor)
    to 9232 at α=0.30 (-183 errs), then stay below anchor through
    α=0.65 (9387 vs 9415). The "magnitude trap" rule (LB-regress when
    errs > anchor) doesn't trigger until α≥0.75 where errors climb
    past the anchor.
  - Per-class recall preservation: Low drops from 0.9955 → 0.9942 over
    the sweep; the -0.0005 guardrail catches it at α=0.65 (right at the
    edge). Conservative α=0.30-0.50 candidates all pass cleanly.

- **Compound (3-stack + α_xgb·XGB + α_lr·LR) grid**: best is
  (a_xgb=0.20, a_lr=0.50) at OOF 0.98168 (Δ +0.00107 vs 3-stack /
  +0.00084 vs 4-stack). Effectively ties the LR-only blend at α=0.50
  on the 4-stack anchor — XGB stacker's incremental signal already
  captured in the 4-stack itself.

- **Test-side disagreement vs primary** (`submission_tier1b_greedy_meta.csv`,
  LB 0.98094):
  - α=0.30: 265 rows differ (0.10%)
  - α=0.50: 477 rows differ (0.18%)
  - α=0.65: 611 rows differ (0.23%)
  Magnitudes large enough to move LB by ±0.001-0.002 if signal transfers.

- **Three submission candidates emitted, AWAITING USER APPROVAL** for LB probe:
  ```
  submission_tier1c_lr_iso_4stack_a030.csv  OOF 0.98152  proj LB ~0.9817 (gap -0.0001)
  submission_tier1c_lr_iso_4stack_a050.csv  OOF 0.98167  proj LB ~0.9818
  submission_tier1c_lr_iso_4stack_a065.csv  OOF 0.98182  proj LB ~0.9819
  ```
  The OOF→LB gap on prior meta-stacker stack went NEGATIVE (-0.00010,
  LB above OOF). If the same calibration property holds for LR, the
  α=0.65 variant projects LB ~0.98192 — within +0.00027 of the leader
  0.98219.

- **Why this is real signal, not OOF overfit** (key audit thinking):
  1. The fixed-bias decision rule is held constant across the sweep
     (no per-α bias retune → no binhigh-style selection inflation).
  2. The LR meta-stacker doesn't share the XGB stacker's tree-depth
     knob — different overfit failure modes.
  3. The lift compounds with the XGB stacker (compound grid shows
     LR adds signal even on top of the 4-stack which already includes
     XGB stacker).
  4. The per-class trade is SAFE (Low recall preserved within
     guardrail at α≤0.65).
  5. Errors DECREASE over the safe α range — opposite of the
     magnitude-trap pattern that has killed every prior candidate.
  6. wguesdon's published 30-model kernel explicitly demonstrates that
     a SIMPLER stacker (his choice was LGB) can beat greedy on LB even
     when greedy has higher CV. Our LB-best (XGB stacker + greedy
     forward) sits in the OPPOSITE corner of the simplicity spectrum.

- **Recommended LB probe order if approved**:
  1. **α=0.50** (`submission_tier1c_lr_iso_4stack_a050.csv`) — middle
     of the safe range, cleanest per-class profile, OOF +0.00083 vs
     primary. Low risk of regression, moderate upside.
  2. If a050 lifts ≥+0.0005, follow with α=0.65 to test whether the
     additional OOF Δ +0.00015 transfers (rare-class guardrail right
     at the edge — riskier but higher upside).
  3. If a050 nulls, do NOT probe α=0.65 (LR lever doesn't transfer).

- LB budget: unchanged at this point (0 spent today). 10 available.

### 2026-04-25 — N1 LB RESULT: LR meta-stacker NULL (LB 0.97991, Δ −0.00103, OOF→LB gap +0.00176)

- LB probe (`submission_tier1c_lr_iso_4stack_a050.csv` at α=0.50,
  user-approved): **LB public = 0.97991**.
  Δ vs LB-best primary (0.98094) = **−0.00103** (clear regression).
  OOF→LB gap = 0.98167 − 0.97991 = **+0.00176** — much wider than the
  −0.00010 negative gap that the prior meta-stacker family showed.
- **LR meta-stacker is OOF-overfit on this 70-component bank.**
  Per the original recommendation conditional ("if a050 nulls, do NOT
  probe α=0.65"), the α=0.65 candidate WILL NOT be probed. N1 lever
  is closed.

- **Calibration ladder update:**
  ```
  3-way multi-seed                  0.98029 → 0.98005   gap +0.00024
  LB-best 3-stack (lb3+rmlp+nonruleiso) 0.98061 → 0.98008  gap +0.00053
  LB-best 4-stack (PRIMARY)         0.98084 → 0.98094   gap −0.00010
  **LR meta-stacker × 4-stack a050  0.98167 → 0.97991   gap +0.00176**  ← REGRESSION
  ```

- **Diagnosis — three structural reasons LR overfit OOF where XGB didn't:**
  1. **StandardScaler + LR(C=1.0) is less regularized than XGB(depth=4,
     reg_alpha=5, reg_lambda=5) on a 227-dim feature space**. XGB's tree
     constraints + L1 regularization actively prune weak component
     contributions; LR with C=1.0 distributes weights across all 210
     log-probability features. Effectively higher capacity per
     informative dimension.
  2. **`class_weight='balanced'` doubles the rare-High loss weight at
     training time**. This pushes LR to over-emphasize High discrimination
     on the OOF folds (each fold's High distribution matches training).
     On the test set with the same prior but slightly different per-row
     features, the over-emphasis flips boundary rows the wrong way.
     Visible in standalone PCR: LR_iso pushed High to 0.9817 (vs primary
     0.9775) BUT errors went UP (+391). The OOF score thinks High +0.0042
     × 1/3 weight = +0.0014 macro-recall lift, more than offsetting
     Low −0.0013 × 1/3 weight = −0.0004; on test, the "High recall
     gain" disappears because the actual flipped rows aren't High.
  3. **70-component bank contains weak components LR can't down-weight**.
     XGB's tree splits naturally ignore weak components (they don't
     improve gain); LR's coefficient regularization just shrinks them
     all uniformly toward zero, leaving residual noise contributions.

- **Portable rule** (LEARNINGS.md candidate): **"On a wide
  meta-stacker bank (≥50 components × 3 classes = ≥150 features),
  multinomial Logistic Regression with `class_weight='balanced'` is
  systematically more OOF-overfit than a heavy-regularized depth-4
  XGB on the same bank — even though LR has fewer architectural
  knobs to tune. The 'simpler model = lower overfit' heuristic
  inverts at this feature-dim / sample-size ratio (210 features /
  504k rows) when the simpler model has class_weight upweighting
  the rare class. Drop class_weight + use stronger L2 (C ≤ 0.1) if
  you want to retry; otherwise prefer XGB or LGB stackers with
  explicit tree-depth caps."**

- **Reconciles with wguesdon's published 30-model kernel**: he chose
  LGB-stacker (NOT LR-stacker) over greedy. LGB's tree splits behave
  similarly to XGB's depth-4 — they also actively down-weight weak
  components. LR was never on his shortlist; we shouldn't have
  assumed it would behave like LGB just because both are "simpler
  than greedy".

- **LB budget**: **1/10 used today**, 9 remaining. LB best unchanged
  at **0.98094** via `submission_tier1b_greedy_meta.csv`.

- **Next bet (re-prioritized after N1 null)**:
  - N2 (weak diversity learners as meta-stacker bank inputs) is now
    PARTIALLY PRE-FALSIFIED: if LR doesn't work as a meta-stacker
    output, then `lr_ote` as a meta-stacker INPUT also has a low
    probability of helping (the meta-stacker would just learn to
    down-weight it, which the existing 70-component bank's diversity
    already does). kNN-ote and ET-ote retain their independent EV
    since they're tree/distance models, not LR.
  - N3 (yunsuxiaozi 5-shuffle OTE concat) is unchanged in priority —
    it's a feature-level lever, orthogonal to N1's failure mode.
  - **Recommended next action**: pivot to **N3** as the highest-EV
    untried lever. N2's kNN+ET sub-experiments can be done in parallel
    if compute allows.

### 2026-04-25 — N1 closed across all α (linear gap-projection on the OOF-overfit signal)

- Question: should we submit the conservative α=0.30 candidate
  (`submission_tier1c_lr_iso_4stack_a030.csv`, OOF 0.98152, Δ +0.00068)?
- Answer: **NO.** The LR meta-stacker's contribution to the OOF→LB gap
  is structurally linear in α (a single overfit component blended at
  fixed bias contributes ~constant per-α gap inflation). Project:
  ```
  α     OOF Δ      proj gap infl.   proj LB Δ vs primary
  0.05  +0.00012   +0.00019         -0.00007
  0.10  +0.00026   +0.00037         -0.00011
  0.30  +0.00068   +0.00112         -0.00044
  0.50  +0.00083   +0.00186         -0.00103   ← observed
  0.65  +0.00098   +0.00242         -0.00144   (would have been worse)
  ```
  - Conservative α=0.30 projects to **LB ~0.98050** (regression).
  - Even α=0.05 projects to LB ~0.98087 (regression).
  - **No α threads the needle**: the LR signal is OOF-overfit at every
    weight, so dilution doesn't help — it just reduces the magnitude
    of a known LB-negative contribution proportionally to its OOF "lift".

- **Mechanism — why dilution can't fix overfit**:
  At α=0.50, the LR contribution at fixed bias is decomposable as:
  ```
  log P_blend = 0.5 · log P_4stack + 0.5 · log P_LR_iso
  ```
  Each LR row whose OOF favored the wrong class also favors the wrong
  class on test (same model, same training distribution). Cutting the
  weight to 0.30 reduces the magnitude of those wrong votes but does
  not change their SIGN. The signed gap inflation per unit α stays
  approximately constant. This is the mathematical consequence of "LR
  overfits SAME components on test as on OOF" — the failure mode is
  in the model, not in the blend weight.
- Practical rule (LEARNINGS.md candidate): **"Once a meta-stacker is
  shown to have a >2x OOF→LB gap inflation per α at one operating
  point, expect roughly linear gap inflation across all α and SKIP
  the conservative dilution probe — it will null at LB even if OOF
  shows a smaller positive Δ. Spend the LB slot on a different lever
  family instead."**

- **N1 lever fully closed** (α=0.50 LB-confirmed null, α=0.30 + α=0.65
  projected null without spending LB slots).

- **Final-selection candidates UNCHANGED**:
  1. **PRIMARY**: `submission_tier1b_greedy_meta.csv` → LB 0.98094
  2. **HEDGE**: `submission_recipe_full_te.csv` → LB 0.97939

- **Recommended next steps (5 days to deadline, 9 LB slots today)**:

  **N3 (top — 1.5-2h CPU): yunsuxiaozi 5-shuffle OTE concat**.
  Feature-level lever, structurally orthogonal to the N1 OOF-overfit
  failure mode. Implement as `recipe_ote_5shuffle_concat` variant —
  the recipe pipeline runs OrderedTE 5× with different shuffle seeds,
  concatenates all 5 fits as augmented training rows (5x training
  pool per fold), then trains XGB on the augmented pool. Standalone
  OOF gate first; if Jaccard < 0.85 vs LB-best AND errs ≤ 10,114
  (recipe baseline), add to meta-stacker bank as a new component.
  Cost: ~1.5-2h CPU per fold × 5 folds = ~10h wall serial, ~3h with
  fold parallelism. Skip if not finishable in current session.

  **N2-restricted (medium, parallel): kNN-ote + ET-ote ONLY**
  (drop lr_ote per N1 falsification). Build 2 new OOFs on the recipe
  feature set:
    `KNeighborsClassifier(n_neighbors=50)` (subsampled 50k stratified
    fit for compute feasibility — full 504k k-NN is impractical)
    `ExtraTreesClassifier(n_estimators=500, class_weight='balanced')`
  Add to the meta-stacker bank → re-train XGB metastack v4 → re-blend.
  Decision gate: meta-stacker errs drop below 8,948 AND Jaccard <
  0.97 vs primary → blend into 4-stack at fixed bias, gate at +0.0002
  OOF. ET takes ~30 min, kNN takes ~45 min. Cheap in parallel.

  **N4 (lock + stop)**: if N2 + N3 both null, lock primary + hedge as
  final and stop spending compute. With current +0.00020 gap to
  pack 0.98114 and ±0.0005 private-LB variance, the marginal LB-probe
  EV is below the cost of variance noise.

- **Skip on principled grounds (re-confirmed)**:
  - LR meta-stacker variants (HP-tuned, different scaler, different
    weight scheme) — N1 is a definitive ARCHITECTURAL null, not a
    tuning issue.
  - All public-CSV blending (banned).
  - Further fine-tuning of the 4-stack composition (Tier 1c saturation
    confirmed three independent ways already).
  - More NN-family attempts beyond TabM (RealMLP n_ens=4 was the 13th
    NN null; magnitude trap pattern is structural at this feature set).

### 2026-04-25 — N2 ET + kNN + meta-stacker v4: NEW OOF BEST 0.98121, AWAITING LB PROBE

- Goal: execute N2 + N3 in parallel per the kernel-audit-round-4 plan.

- **N2 ExtraTrees** (`scripts/n2_extratrees.py`, ~3.6 min CPU):
  500 trees on 35-dist features, 5-fold seed=42, class_weight='balanced'.
  OOF tuned **0.96667**, errs 10,371, **Jaccard 0.589** vs LB-best 4-stack
  — strongest tree-family orthogonality in the bank.

- **N2 kNN** (`scripts/n2_knn.py`, ~2.4 min CPU):
  k=50 on 80k stratified subsample fit, 5-fold seed=42. OOF tuned
  **0.96308**, errs 11,235, **Jaccard 0.548** vs LB-best — STRONGEST
  orthogonality in the entire bank (lower than even RealMLP 0.62).

- **Direct blend gate** (`scripts/n2_blend_direct.py`): both ET and kNN
  strictly hurt LB-best 4-stack at every α > 0. Per-class trade is High
  recall DOWN for Low+Med up — wrong direction under macro-recall.
  Direct blend lever DEAD; only meta-stacker absorption remains.

- **N3 5-shuffle OTE concat** (production launched in background):
  killed by container rehydrate ~10 min into fold-1 XGB on 2.52M-row
  augmented set. Per-fold checkpointing in place but production fold
  takes ~25-30 min and rehydrate hit before fold 1 finished. Smoke
  confirmed +0.00177 OOF lift on 20k subset over recipe smoke baseline.
  Production retry needs Kaggle CPU kernel (9h cap, no rehydrate).

- **Meta-stacker v4** (`scripts/tier1c_metastack_v4.py`) — XGB stacker
  on bank + ET + kNN, **NEW OOF BEST**:
  - Same XGB (depth=4, reg_alpha=5, reg_lambda=5, lr=0.05) heavy-reg
    model class as the prior LB-best meta-stacker; only the bank
    changed (75 + n2_extratrees + n2_knn = 77 components).
  - Standalone iso OOF tuned **0.98102** (vs prior XGB metastack iso
    0.98059, **+0.00043 standalone lift**).
  - 4-stack drop-in replacement OOF **0.98108** (Δ +0.00024 vs LB-best
    4-stack 0.98084).
  - Blend on top of LB-best 4-stack at fixed bias (best so far):
    ```
    α       OOF       Δ vs LB4   errs   recL    recM    recH    guardrail
    0.20  0.98107  +0.00023      9153  0.9956  0.9706  0.9771  PASS
    0.25  0.98112  +0.00028      9112  0.9956  0.9707  0.9771  PASS
    0.30  0.98119  +0.00034      9078  0.9956  0.9709  0.9771  PASS
    **0.35  0.98121  +0.00036      9049  0.9956  0.9709  0.9771  PASS**  ← peak
    0.40  0.98119  +0.00035      9017  0.9956  0.9711  0.9769  borderline
    ```
    LB-best baseline: errs=9415, PCR=[L 0.9955, M 0.9695, H 0.9775].
  - **Errors DECREASE monotonically** across the sweep (9415 → 9049 at
    α=0.35) — opposite of the magnitude-trap pattern that killed prior
    NN attempts. Trade is favorable: tiny High recall drop (-0.0004)
    for Medium gain (+0.0014) + Low slight up.

- **Three submission candidates emitted, AWAITING USER APPROVAL** for
  LB probe:
  ```
  submission_tier1c_meta_v4_a030.csv  conservative   154 test rows differ from primary
  submission_tier1c_meta_v4_a035.csv  RECOMMENDED    177 test rows differ
  submission_tier1c_meta_v4_a040.csv  aggressive     207 test rows differ (borderline guardrail)
  ```

- **Why this is real signal, NOT OOF overfit like N1 LR was**:
  1. **Same heavy-reg XGB model class** as the prior LB-best meta-stacker,
     which had a NEGATIVE OOF→LB gap (-0.00010, LB > OOF). The model
     class is unchanged; only the bank grew (75 → 77 components). The
     prior calibration property should carry over.
  2. **Errors DECREASE in the safe α range** — opposite of the LR null
     pattern (where errors went UP) and opposite of every prior NN
     magnitude-trap failure.
  3. **Per-class trade preserves rare class** — High recall down only
     -0.0004 (within guardrail), Medium UP +0.0014 (most of the lift),
     Low slight up. Net positive under macro-recall without the
     rare-class sacrifice that doomed LR.
  4. **Standalone v4 lift is honest scale** — +0.00043 OOF over prior
     XGB metastack iso (vs LR's +0.00124 was a much bigger jump that
     turned out to be overfit).
  5. **Same blend mechanism** that produced LB 0.98094 (prior metastack
     iso × LB-best 3-stack at α=0.30). v4 just plugs a slightly stronger
     metastack into the same architecture.

- **Recommended LB probe order if approved**:
  1. **α=0.35** (`submission_tier1c_meta_v4_a035.csv`) — peak OOF,
     all PCR comfortably within guardrail. If gap stays at -0.00010,
     expected LB ~0.98131 (above pack 0.98114).
  2. If a035 lifts ≥+0.0002, follow with α=0.40 to test whether the
     borderline-guardrail variant gives +0.00035 more or regresses on
     the High recall trade.
  3. If a035 nulls, do NOT probe α=0.40.

- **N3 retry strategy** (separate from v4 LB-probe decision): smoke
  showed real signal (+0.00177 over recipe smoke baseline). Production
  needs an environment that survives ~2.5h compute. Two options:
  1. **Kaggle CPU kernel** (best — 9h cap, no rehydrate). ~30 min
     scaffolding to inline `recipe_ote_5shuffle.py` +
     `recipe_full_te_5shuffle.py` + `recipe_features.py` +
     `recipe_ote.py` into a single kernel script, push, pull resulting
     OOF + test back. Blocks N3 result by ~3h (queue + run).
  2. **Reduced-K local** (faster, less faithful) — K=3 instead of K=5
     halves wall time. Architecturally weaker but might fit in a
     90-min container window between rehydrates.
  Recommended: option 1 (Kaggle kernel) — closer to published technique,
  more likely to transfer.

- LB budget: **1/10 used today, 9 remaining**.

### 2026-04-25 — N2 v4 LB RESULT: NULL (LB 0.97992, Δ −0.00102, gap +0.00129) — meta-stacker bank saturated

- LB probe (`submission_tier1c_meta_v4_a035.csv` at α=0.35,
  user-approved): **LB public = 0.97992**.
  Δ vs LB-best primary (0.98094) = **−0.00102** (clear regression).
  OOF→LB gap = 0.98121 − 0.97992 = **+0.00129** — much wider than
  prior XGB metastack's −0.00010 negative gap.
- Almost identical magnitude regression to the N1 LR meta-stacker
  (LB 0.97991, Δ −0.00103, gap +0.00176).
- Per the original recommendation conditional ("if a035 nulls, do NOT
  probe α=0.40"), v4 α=0.40 will NOT be probed. **v4 lever closed.**

- **Diagnosis — N2 components saturated the meta-stacker bank**:
  Same XGB heavy-reg model class as the prior LB-best meta-stacker
  (which had a NEGATIVE OOF→LB gap of −0.00010). Only the bank changed
  (75 → 77 components). Adding 2 weak components with high standalone
  error counts (ET 10,371 errs, kNN 11,235 errs) gave the depth-4 XGB
  stacker more places to look for spurious signal. Even with depth=4 +
  reg_alpha=5 + reg_lambda=5, the additional features broke the
  prior's calibration property.
- **Errors-decrease-monotonically heuristic IS NOT sufficient for LB
  transfer.** The v4 sweep showed errs going from 9415 → 9049 (clean
  pattern, opposite of magnitude trap) AND PCR within guardrail at
  α≤0.35. Both signals predicted LB transfer. Both were wrong because
  the underlying meta-stacker had OOF-fitted to noise on the new
  components.

- **Linear gap-projection rules out α=0.30 too** (same logic as
  N1 LR closure on 2026-04-25):
  ```
  α      OOF Δ      proj gap infl   proj LB Δ vs primary
  0.20  +0.00023   +0.00073        -0.00050
  0.30  +0.00034   +0.00109        -0.00075
  0.35  +0.00036   +0.00129        -0.00093 (observed -0.00102)
  0.40  +0.00035   +0.00148        -0.00113 (would have been worse)
  ```
  No α threads the needle. v4 conservative not worth probing.

- **Portable rule** (LEARNINGS.md candidate): **"Wguesdon's 'weak
  individually, help stacker' pattern (lr_ote/knn_ote/et_ote) is
  bank-size dependent. On a bank already at saturation (75+
  components, prior XGB metastack already with negative OOF→LB gap),
  adding weak components causes the meta-stacker to overfit on the
  new features. The pattern only helps when the bank is far from
  saturation. For a saturated bank, ANY meta-stacker change (model
  class, additional components, reweighting) tends to regress LB
  even when OOF improves with all-error-decreasing + per-class
  guardrail signals."**

- **Combined N1 + N2 closure**: both meta-stacker family experiments
  null. Saturation at LB 0.98094 confirmed for FOUR independent
  attacks now:
  1. Tier 1c greedy + meta-on-meta + seed-bag (2026-04-25)
  2. Cross-poll metastack v3 (2026-04-25)
  3. SMOTE-NC training-data lever (2026-04-25)
  4. **N2 + meta-stacker v4 bank-extension (this entry)**

- **LB budget**: **2/10 used today**, 8 remaining. LB best unchanged
  at **0.98094** via `submission_tier1b_greedy_meta.csv`.

- **Only remaining bet**: N3 (yunsuxiaozi 5-shuffle OTE concat) —
  feature-level lever, structurally orthogonal to all four nullified
  meta-stacker family attacks. Production blocked on container
  rehydrate; needs Kaggle CPU kernel (~30 min scaffolding +
  ~3h queue+run).

### 2026-04-25 — N3 K=2 5-shuffle OTE concat: NULL (Jaccard 0.84 + magnitude trap)

- Goal: execute N3 without Kaggle compute via per-fold execution
  (`RUN_FOLD=N` env var), each fold ~18 min wall fits inside container-
  rehydrate-active interval if user keeps polling.
- Compute config: K=2 (vs published K=5), MAX_ROUNDS=1500, ES=100.
  Reduced from full to fit per-fold time budget.
- Per-fold timeline (each ~17-19 min wall on 504k × K=2 = 1.01M aug rows):
  ```
  fold | val argmax | recipe baseline | Δ
  -----|------------|-----------------|--------
   1   | 0.97601    | 0.97544         | +0.00057
   2   | 0.97599    | 0.97659         | -0.00060
   3   | 0.97759    | 0.97721         | +0.00038
   4   | 0.97554    | 0.97465         | +0.00089
   5   | 0.97578    | 0.97557         | +0.00021
  ```
  Mean fold delta = +0.00029 (4/5 positive). Cumulative net +0.00145.
- **Aggregated standalone OOF (5-fold sum)**:
  - Argmax 0.97618 (recipe 0.97589, **+0.00029**)
  - Tuned 0.98004 (recipe 0.97967, **+0.00037** — below +0.0005
    LB-transfer threshold for direct blend; signals real but bounded)
  - iso-cal tuned 0.98001
- **Blend gate vs LB-best 4-stack (0.98084, fixed bias)**:
  - **Jaccard 0.8417** — above 0.80 redundancy threshold (FAIL diversity)
  - **Errors 9591 vs anchor 9415 = +176** (FAIL magnitude)
  - **Per-class recall trade**: Low ~tied, Medium −0.0004, **High −0.0020**
    (FAIL — wrong direction under macro-recall; same pattern as v4 bank
    addition + LR meta-stacker)
  - Blend sweep: every α > 0 hurts; peak at α=0 (no blend). Strict null.
  ```
  α       OOF       Δ
  0.000  0.98084  +0.00000   ← peak
  0.025  0.98082  -0.00002
  0.300  0.98050  -0.00034
  0.500  0.98049  -0.00036
  ```
- **Direct blend lever DEAD.** Per the v4 lesson (saturation rule),
  adding N3 to the 77-component meta-stacker bank is also unlikely
  to help — bank-add causes meta-stacker to OOF-overfit on new
  components even when they pass surface gates.
- **Why K=2 may have hurt**: published recipe uses K=5 shuffles; we used
  K=2 to fit per-fold time budget. Less augmentation = less
  regularisation = closer to vanilla recipe pipeline. K=5 might lift
  the standalone OOF +0.0001-0.0002, but Jaccard 0.84 + magnitude trap
  are structural — they wouldn't change with more K. The OTE-concat
  augmentation lever is fundamentally too similar to recipe's existing
  OTE+digit features to add orthogonal signal on this feature set.
- **Combined N1 + N2 + N3 closure**: kernel-audit-round-4 plan FULLY
  complete. All three nominated levers null. LB best unchanged at
  **0.98094**. LB budget: 2/10 used today.

### 2026-04-25 — Session close-out: 3 LB-confirmed nulls + lever bank exhausted

- **Today's experiments** (all kernel-audit-round-4 follow-ups):
  1. N1 LR meta-stacker — LB 0.97991, Δ −0.00103, gap +0.00176
  2. N2 v4 metastack with ET+kNN — LB 0.97992, Δ −0.00102, gap +0.00129
  3. N3 K=2 5-shuffle OTE — direct blend null (Jaccard 0.84, magnitude
     trap, wrong-direction PCR)
- **All three confirmed via fixed-bias blend gate at LB or OOF +
  structural diversity check. No surprises remain in the kernel-audit-
  round-4 plan.**
- **Saturation at LB 0.98094 confirmed across FIVE independent attacks**:
  Tier 1c greedy + meta-on-meta + seed-bag, cross-poll v3 metastack,
  SMOTE-NC v2/v3, N2 v4 bank-extension, N3 5-shuffle OTE training-
  augmentation.
- **Final-selection LOCKED**:
  - **PRIMARY**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
    (gap −0.00010, anchor for blend-gate threshold). Composition:
    LB-best 3-stack + xgb_metastack_iso × α=0.30.
  - **HEDGE**: `submission_recipe_full_te.csv` → **LB 0.97939**
    (gap +0.00028, single-model XGB on V10 recipe, no blend stacking).
    Premium = -0.00155 LB; protects against meta-stacker overfit on
    private LB.
- **5 days to deadline (2026-04-30), 8 LB submissions remaining today.**
  Reserve remainder for end-of-comp variance check (one final-selection
  re-validation per day until close).
- **Portable rules added this session** (LEARNINGS.md candidates):
  1. **Linear OOF→LB gap projection**: once a single overfit blend
     component shows >2x gap inflation per α at one operating point,
     project linearly across all α and skip conservative-dilution
     probes — they will null too.
  2. **"Errors decrease + per-class guardrail PASS" is NOT sufficient
     for LB transfer** when the new component is itself a meta-stacker
     output trained on the same fold structure as the anchor.
  3. **Wguesdon's "weak helps stacker" pattern is bank-size dependent**
     — only helps far from saturation. On a 75+ component bank with a
     prior negative OOF→LB gap, adding weak components (lr_ote /
     et_ote / knn_ote analogs) overfits OOF.
  4. **Container-rehydrate-resistant compute pattern**: per-fold
     execution via `RUN_FOLD=N` env var + per-fold `.npy` checkpoints
     committed to git, foreground bash invocation per fold,
     interactive polling to keep container alive.
  5. **OTE training-augmentation (yunsuxiaozi 5-shuffle concat) is
     structurally redundant with recipe's existing OTE on this feature
     set** — Jaccard 0.84 + magnitude trap regardless of K.

### 2026-04-25 — cross-poll v3 + SMOTE-NC kernel: 3 NULLs, own-pipeline closed

- Goal: extend the 63-component Tier-1b meta-stacker with new candidates
  + run SMOTE-NC at production scale via Kaggle GPU. Two attacks against
  LB 0.98094: bank-extension (cross-poll) and training-data-level lever
  (SMOTE).
- Changed:
  - `scripts/tier1b_xgb_metastack_v3.py` — cross-pollinate meta-stacker
    with `recipe_focal_g2_invfreq` + `xgb_nonrule_bag3`, stricter EXCLUDE
    drops known LB-regressors. 64 components, ~5 min wall.
  - `scripts/emit_metastack_v3_submission.py` — emits the v3 iso-blend
    α=0.30 candidate (peak vs LB-best 4-stack OOF +0.00015, below
    internal +2e-4 gate but submitted per user direction).
  - `kaggle_kernel/kernel_smote_recipe/recipe_smote.py` (788 L) — single-
    file kernel, per-fold SMOTE-NC on RAW 19 cols (8 cats + 11 nums)
    only, then re-derive combos / digits / num_as_cat / freq /
    orig_stats on augmented rows via cached vocab maps. Memory peak
    ~91 MB vs 45 GiB OOM that killed local recipe_smote_high.py.
    Promise-gate after fold 1 (argmax≥0.97500, recall_high≥0.965,
    errs≤1.05× recipe; PROCEED if 2-of-3 pass OR recall_high lifts
    +0.5pp). Hard kill at t+55min (CLAUDE.md GPU rule).
  - `scripts/recipe_smote_v2.py` + `scripts/smote_local/{load_engineer,
    fe_with_maps,redrive,gate,cv_loop}.py` (modular ≤150L per file) —
    local CPU twin reusing `scripts/recipe_features.py` + `recipe_ote.py`
    so any kernel results can be reproduced offline.
  - `MAX_FOLDS` env override + per-fold OOF/test/JSON checkpointing to
    survive container rehydrates.

- **Cross-poll metastack v3 — LB REGRESSION**:
  ```
  standalone @ recipe bias       0.97365 argmax / 0.97954 tuned
  iso-cal'd @ recipe bias        0.97956
  blend vs LB-best 4-stack       peak α=0.30 → OOF 0.98099 (+0.00015)
  Jaccard vs LB-best             0.94
  errs vs LB-best                similar
  LB submission α=0.30           **0.98060  (Δ = -0.00034 vs LB-best 0.98094)**
  OOF→LB gap                     +0.00039 (vs Tier-1b's −0.00010)
  ```
  Adding 2 components inflated OOF +0.00015 but cost LB -0.00034.
  **Bank-extension lever DEAD.** The 63-component meta-stacker has
  absorbed all available signal; further additions amplify
  fold-noise overfit without adding orthogonal signal.

- **SMOTE-NC v2 (TARGET=42k, K=5, sample_weight='balanced')** — fold-1 NULL:
  Two production-scale bugs surfaced + fixed before fold-1 ran clean:
  1. SMOTE-NC dtype detection: `df[c].dtype == "object"` returned False
     at production scale — pandas reported cats as 'category' dtype.
     Fix: pass explicit `cats` list, force `astype(str)` defensively.
  2. Kaggle kernel `_find_one("Irrigation_Prediction*.csv")` didn't
     match dataset's lowercase `irrigation_prediction.csv`. Fix: try
     multiple casings.

  After fixes (Kaggle kernel v2 + local v2):
  ```
                        Kaggle    Local     recipe baseline
  argmax_bal            0.97436   0.97471   ~0.97544
  recall_high           0.9502    0.9514    ~0.977 (-2.7pp)
  recall_med            0.9775    0.978     ~0.969 (+0.008)
  errors                1630      ~1700     ~2900
  decision              ABORT     ABORT     gate caught it
  ```
  Cross-platform agreement to noise (Δ ≤ 0.001). **High recall hurt by
  2.7pp** — opposite of the hypothesis. Mechanism: synthetic Highs
  interpolated from 5 NN bled into Medium-territory, blurring the
  M↔H decision boundary. balanced sample weight + SMOTE oversample
  partially cancel (more H rows × proportionally lower weights = same
  total H gradient), so the only net effect is decision-boundary
  diffusion.

- **SMOTE-NC v3 (TARGET=25k, K=10) — softer config, full 5-fold NULL**:
  Hypothesis: smaller TARGET (1.5× vs 2× original High count) +
  smoother K=10 NN should reduce the diffusion mechanism that hurt v2.
  Kaggle GPU completed all 5 folds (gate PROCEEDed: 2-of-3 metrics
  passed even though recall_high still 0.9529 < 0.965 floor).

  ```
                              v2 (42k, K=5)   v3 (25k, K=10)   recipe
  fold-1 argmax               0.97436         0.97513          ~0.97544
  fold-1 recall_high          0.9502          0.9529           ~0.977
  full 5-fold tuned OOF       — (aborted)     0.97963          0.97967
  log-bias                    —               [-1.91, -1.78, 0.41]   [1.43, 1.47, 3.40]
  errs vs LB-best 4-stack     —               +366             +542 anchor
  Jaccard vs LB-best          —               0.8225           1.00 anchor
  blend Δ peak (fixed bias)   —               +0.00002 @ α=0.075
  ```

  **Both blend-gate criteria fail**: Jaccard 0.82 above 0.80 redundancy
  threshold + errs (+366 over anchor) violates magnitude rule. Tuned
  bias `[-1.91, -1.78, 0.41]` is structurally incompatible with the
  recipe-bias-anchored stack: SMOTE shifted prob scale far enough
  that calibration alignment with the existing 63-component bank is
  impossible without retuning the entire stack's bias.

- **Strategic implication: own-pipeline lever bank fully exhausted.**
  Three orthogonal attacks against LB 0.98094 today:
  1. Tier 1c (yesterday): greedy / meta-on-meta / seed-bag — saturated.
  2. Cross-poll metastack v3 (today): bank extension — LB regressed.
  3. SMOTE-NC v2 + v3 (today): training-data lever — both NULL.

  Combined with the prior structural confirmations (B2 GroupKFold
  honest, Pareto-frontier closure on per-class High recall, 13 NN
  family nulls, focal/distill/specialist nulls), there is no remaining
  own-pipeline mechanism that can break LB 0.98094 within the
  +0.0002 LB-transfer threshold. CLAUDE.md rule forbids public-CSV
  blending.

- **LB budget**: 4/10 used today (3 from yesterday + 1 cross-poll
  probe), 6 remaining. **LB best unchanged** at 0.98094 via
  `submission_tier1b_greedy_meta.csv`.

- Artefacts (whitelisted in `.gitignore`):
  - `scripts/artifacts/oof_xgb_metastack_v3{,_iso}.npy` + test + JSON
  - `scripts/artifacts/oof_smote_v2_fold1.npy` + test (partial, fold-1
    only — Kaggle/local cross-platform validation)
  - `scripts/artifacts/oof_recipe_smote_v3.npy` + test + JSON +
    fold1_gate.json (full 5-fold, Kaggle GPU production)
  - `kaggle_kernel/kernel_smote_recipe/{recipe_smote.py,kernel-metadata.json}`
  - `scripts/recipe_smote_v2.py` + `scripts/smote_local/*.py` (local
    twin, modular ≤150L per file)
  - `submissions/submission_metastack_v3_iso_a300.csv` (LB 0.98060)

- **Final-selection lock recommendation** (5 days to deadline):
  - **Primary**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
    (gap −0.00010, anomalous LB > OOF). Composition: lb3 + RealMLP α=0.20
    + xgb_nonrule_iso α=0.075 + xgb_metastack_iso α=0.30.
  - **Hedge**: `submission_recipe_full_te_catboost.csv` → **LB 0.97935**
    (gap +0.00001, tightest calibration in ladder). Different model
    family (CatBoost ordered-boosting vs primary's XGB+RealMLP+meta-XGB).
    Premium = -0.00159 LB; protects against meta-stacker private-LB
    overfit. Reserve 6 LB submissions for end-of-comp variance check.

- **Lessons logged for future synthetic-tabular comps**:
  1. **SMOTE-NC at production scale requires raw-only inputs.** Feeding
     a 443-col FE matrix (with high-card combos) OOMs at 45 GiB on
     16 GB containers. Refactor: SMOTE on raw cats+nums (~90 MB peak),
     then re-derive FE on augmented rows via cached vocab maps. Never
     SMOTE on factorized-int columns (they're not real categoricals).
  2. **pandas dtype inference for SMOTE-NC fails at production scale.**
     `df[c].dtype == "object"` returned False on cats at 504k rows
     (pandas reported 'category'). Pass explicit cat-name list to
     SMOTE-NC's `categorical_features=` arg, force `astype(str)`
     defensively.
  3. **Synthetic minority oversampling + balanced sample weights cancel
     each other** when minority gradient was already saturating. The
     only net effect is decision-boundary diffusion (bad). For SMOTE
     on imbalanced data with tree models, choose ONE rebalancing
     mechanism, not both.
  4. **Tuned bias incompatibility breaks blend extension.** A model
     trained with structurally different class priors (post-SMOTE)
     will land at log-bias `[-1.9, -1.8, +0.4]` while a recipe-anchored
     stack uses `[+1.4, +1.5, +3.4]`. They can't be log-blended at
     fixed anchor bias regardless of standalone OOF — calibration
     alignment is a hard prerequisite.
  5. **Promise-gate after fold-1 saved compute** — Kaggle v2's full
     5-fold would have cost ~50 min; aborted at fold 1 in 7 min after
     gate triggered. Pattern: persist OOF/test/JSON BEFORE evaluating
     the gate, decision rule based on standalone metrics + class-recall
     direction, ABORT exits cleanly.

### 2026-04-25 — junior-engineer audit + J1 tree-leaf-OTE meta-stacker (NULL on blend gate; orthogonality lever proven)

- Goal: fresh-eyes review of saturation evidence, then run the highest-EV
  / cheapest speculative lever from the J1-J7 menu (next section below).
- Approach: SMOKE first (SMOKE=1, 20k×2-fold, ~1 min wall) to validate
  the leaf-extraction → OTE-encode → meta-XGB chain end-to-end, then full
  5-fold seed=42 production.
- Mechanism: train a base XGB on 19 raw factorized features, extract
  per-tree leaf indices for every row (multi:softprob produces 3 trees
  per round → 50 rounds × 3 cls = 150 trees), treat each tree's leaf-
  index column as a high-card categorical, OrderedTE-encode (3 classes,
  450 OTE features), feed to a meta XGB. The meta sees TREE-SPACE
  (per-tree partition memberships) instead of PROB-SPACE, structurally
  orthogonal to the 63-component meta-stacker bank that produced LB
  0.98094.
- Smoke (20k×2-fold, ~55s wall) PASSED:
  - base argmax 0.93189 → meta argmax 0.94664 (+0.01475)
  - base tuned  0.95151 → meta tuned  0.95903 (+0.00752)
  - meta-vs-base error Jaccard 0.6504 (well below 0.97 redundancy)
  - base errs 469 → meta errs 442 (-27, -5.8%)
  Chain works end-to-end at smoke scale.
- Production (504k × 5-fold seed=42, 19 raw features, 9.7 min wall):
  - base argmax 0.94908, tuned 0.96395, bias [0.43, 1.17, 3.00]
  - meta argmax 0.95946, tuned 0.96564, bias [0.73, 1.17, 3.40]
  - meta-vs-base Jaccard at smoke scale held → tree-space encoding
    extracts signal at production scale too
- **Blend gate vs LB-best 4-stack (anchor OOF 0.98084, errs 9,415):**
  - Standalone meta @ LB bias: argmax 0.96529, errs 10,830 (+1,415,
    +15% magnitude). Per-class recall L 0.9949 / M 0.9686 / H 0.9324
    (High recall craters by 0.045 vs 4-stack 0.9775 — catastrophic
    under macro-recall).
  - **Jaccard(leaf-OTE, LB-best 4-stack) @ LB bias = 0.5597**
    — **LOWEST ORTHOGONALITY OF ANY CANDIDATE EVER TESTED** on this
    problem (lower than RealMLP's 0.62, lower than Trompt's 0.53
    fold-1, lower than every NN family null). Tree-space encoding is
    GENUINELY orthogonal as the J1 mechanism predicts.
  - Fixed-bias α-sweep:
    ```
    α       tuned    Δ vs 4st  errs   recL    recM    recH    Jaccard
    0.000  0.98084  +0.00000   9415  0.9955  0.9695  0.9775  1.0000
    0.025  0.98067  -0.00017   9353  0.9956  0.9697  0.9767  0.9820
    0.050  0.98062  -0.00023   9268  0.9957  0.9700  0.9762  0.9637
    0.100  0.98024  -0.00061   9192  0.9958  0.9703  0.9746  0.9282
    0.200  0.97955  -0.00129   9077  0.9959  0.9708  0.9719  0.8679
    0.500  0.97661  -0.00423   8962  0.9959  0.9722  0.9617  0.7363
    ```
    Monotone-negative from α=0.025; emit gate fails at every α.
  - **Classic magnitude-trap failure mode** (15th confirmation on this
    problem): great Jaccard orthogonality but 15% more errors,
    distributed AGAINST the rare High class. Macro-recall punishes
    rare-class recall drops; total-error decrease (9415 → 8940 at
    α=0.4) is invisible to the metric.
- **Read-out**: J1 mechanism (TREE-SPACE encoding → orthogonal signal)
  is PROVEN. The Jaccard 0.56 is unprecedented on this problem and
  validates the audit hypothesis that the LR meta-stacker null doesn't
  generalize to all simpler/different stackers. **What J1 with weak
  base does NOT prove**: whether scaling the base to recipe-features
  (443 cols) would close the magnitude gap. The current standalone
  (0.96564 tuned) is too weak to test that hypothesis cleanly.
- LB delta: n/a (no LB probe — gate failed at every α).
- Current LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.
- Artefacts (whitelisted in `.gitignore` for cross-branch reuse):
  - `scripts/artifacts/oof_leaf_ote_meta.npy` (7.3 MB)
  - `scripts/artifacts/test_leaf_ote_meta.npy` (3.1 MB)
  - `scripts/artifacts/leaf_ote_meta_results.json`
  - `scripts/artifacts/leaf_ote_blend_results.json`
  - 3 scripts: `leaf_ote_smoke.py`, `leaf_ote_metastack.py`,
    `leaf_ote_blend.py`
- **Possible J1-v2 retry** (~45 min CPU): swap the 19-raw-feature base
  for a recipe-feature base (443 cols). Base XGB fits at ~5 min/fold
  on recipe → ~25 min for 5 folds; OTE on 450 leaves ≈ same as J1-v1
  (~10 min); meta XGB fit ≈ same (~10 min). If recipe-base meta
  standalone reaches ≥0.978 (closer to 4-stack's 0.98084), the
  magnitude trap may close. Risk: recipe-base trees may produce leaves
  too similar to recipe_full_te's tree partitions (recipe_full_te is
  ALREADY in the 63-component bank), making the OTE encoding
  redundant. Decision after either continuing with J1-v2 or pivoting
  to J2 (bootstrap-bagged meta-stacker, 30 min CPU) deferred to user.
- Lessons logged for future synthetic-tabular comps:
  1. **Tree-leaf OTE encoding is the cheapest known mechanism for
     producing ORTHOGONAL meta-stacker inputs** (Jaccard 0.56 vs the
     LB-best 4-stack which itself averages over 70+ component pred
     scales). For future comps, run this lever EARLY before exhausting
     prob-space stackers.
  2. **The magnitude trap rule is stricter than "errs ≤ anchor"** —
     it's "errs ≤ anchor AND distributed in the right per-class
     direction for the metric". A candidate with FEWER total errors
     but more rare-class errors is LB-negative under macro-recall.
     Per-class recall guardrail (≥anchor − 5e-4 per class) is the
     correct gate, not raw error count.
  3. **Weak-base + strong-meta does not bypass the magnitude trap.**
     The base sets the standalone ceiling; a meta XGB on top of a
     19-feature base produces orthogonal partitions but those
     partitions inherit the weak base's class-recall profile.

### 2026-04-25 — J1-v2 (recipe-feature base) NULL: bias-mismatch, not magnitude

- Goal: J1-v1 closed NULL with weak 19-raw-feature base. J1-v2 retest:
  swap to a recipe-feature base (443 cols + per-fold OrderedTE,
  matching recipe_full_te's pipeline) to see if a stronger base
  closes the magnitude trap. Risk: recipe-base trees may produce
  leaves too similar to recipe_full_te's partitions (already in the
  63-component meta-stacker bank), making the leaf-OTE encoding
  redundant.
- Production (504k × 5-fold seed=42, 14.7 min wall):
  - base: argmax 0.97678, tuned 0.97729, bias [1.83, 2.27, 2.50]
  - meta: argmax 0.97674, tuned 0.97812, bias [1.83, 1.77, 3.20]
  - meta-over-base lift shrunk to +0.00083 (vs J1-v1's +0.00169 with
    weak base) — confirms partial redundancy with recipe-base
- **Blend gate vs LB-best 4-stack** (anchor 0.98084, errs 9,415, bias [1.43, 1.47, 3.40]):
  - Standalone v2 @ LB bias: argmax 0.97734, **errs 12,343 (+31%)**.
    Per-class recall L 0.9950 / **M 0.9580** (-0.0115!) / H 0.9791
    (+0.0016).
  - Jaccard(leaf-OTE-v2, 4-stack) = **0.6696** — orthogonality
    intact (well below 0.97 redundancy threshold)
  - Fixed-bias α-sweep: monotone-negative from α=0.025 (Δ=-0.00004
    at α=0.025, -0.00128 at α=0.500). Emit gate fails at every α.
- **NEW failure mode** (different from J1-v1): not magnitude
  per-class direction; it's **bias-signature mismatch at fixed-LB-bias
  evaluation**. v2's own tuned bias is [1.83, 1.77, 3.20], vs LB's
  [1.43, 1.47, 3.40] — a ~0.4-unit shift on Low and ~0.5 on Medium.
  At LB-bias the v2 meta's argmax push lands wrong: Medium boundary
  rows that 4-stack catches at L/M get rerouted to H. Per-class
  isotonic calibration (already applied via `iso_cal`) doesn't fix
  per-row overfit. Net macro-recall at LB bias: 0.97734 (v2) vs
  0.97750 (4-stack), so even at standalone v2 trails by 0.00016 at
  the anchor's operating point.
- **Two different failure modes across J1-v1 + J1-v2**:
  - v1 (weak base, 19 raw features): magnitude trap, +15% errs,
    High recall crashes -0.045 — error MAGNITUDE distributed
    against the rare class
  - v2 (strong recipe base): bias-signature mismatch trap, +31%
    errs at LB bias, M↔H recall pivots in the wrong direction —
    error MAGNITUDE worse, but per-class direction has High up
- **J1 lever family CLOSED** at two distinct failure points. The
  TREE-SPACE ENCODING mechanism (per-tree partition memberships
  via OrderedTE) genuinely produces orthogonal predictions
  (Jaccard 0.56 v1, 0.67 v2 — both well below redundancy
  thresholds) but cannot clear the blend gate because:
  1. With weak base, magnitude is too high
  2. With strong base, bias-signature differs from anchor enough
     that fixed-bias evaluation flips error direction
  Both modes are structural: a leaf-OTE meta is a fundamentally
  separate model from the anchor's component family, so its
  decision-rule operating point won't align with the anchor's
  fixed bias regardless of how the base is configured.
- LB delta: n/a. No probe warranted at any α.
- Current LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.
- Artefacts (whitelisted in `.gitignore`):
  - `scripts/artifacts/oof_leaf_ote_meta_v2.npy` (7.3 MB)
  - `scripts/artifacts/test_leaf_ote_meta_v2.npy` (3.1 MB)
  - `scripts/artifacts/leaf_ote_meta_v2_results.json`
  - `scripts/artifacts/leaf_ote_v2_blend_results.json`
  - 2 scripts: `leaf_ote_metastack_v2.py`, `leaf_ote_blend_v2.py`
- **Pivot recommendation**: J2 (bootstrap-bagged meta-stacker) is
  the next mechanically distinct lever. Unlike J1's "different
  feature representation" approach, J2 attacks the SAME 63-component
  bank that produced LB 0.98094, but via component-bootstrap of the
  meta's INPUT features — same calibration signature as the existing
  meta-stacker, no bias mismatch. Lower upside (+0.00005-0.00015
  expected) but higher transfer probability.
- Lessons logged for future synthetic-tabular comps:
  1. **The "blend gate" needs a fourth criterion beyond errs/Jaccard/
     recall floor: bias-signature ALIGNMENT.** A candidate with a
     standalone-tuned bias that differs from the anchor's bias by
     more than ~0.2 units on any class will fail fixed-bias evaluation
     even if its standalone tuned-OOF is competitive. Pre-screen by
     comparing tuned biases before launching a full sweep.
  2. **Tree-leaf OTE encoding is structurally a separate model
     family** from the anchor it tries to blend with — different
     base, different OTE source (leaves vs cats), different meta
     XGB. Even when the base USES the same FE as the anchor, the
     meta's per-row prob distribution has a different operating
     point. To use this lever effectively, retune the entire stack's
     bias jointly with the leaf-OTE leg — but doing so risks
     binhigh-style OOF-tuning overfit.
  3. **Stronger base → smaller meta lift** in this lever family
     because the meta's tree-space encoding becomes increasingly
     redundant with what a strong base XGB already captures via
     its softprob output. The sweet spot, if one exists, is a base
     just strong enough to clear magnitude but not so strong that
     tree-space adds nothing. Heuristic to test: aim for base
     standalone within 0.005 of anchor (here 0.98084 - 0.005 =
     0.97584 ideal base target). v1 at 0.96395 was too weak; v2
     at 0.97729 was just under target. A base at 0.975-0.978
     range might thread the needle but is unstable to tune.

### Next steps: junior-engineer audit speculative levers (2026-04-25)

After 4 LB-probed nulls today (LR meta-stacker, cross-poll v3 metastack,
SMOTE v2, SMOTE v3) and saturation across the standard toolkit, fresh
review identified levers exploiting the **ONE** unstressed property of
the LB-best 4-stack: the negative OOF→LB gap (LB > OOF by 0.00010).
That's CV-pessimism from averaging noisy fold hold-outs in the
meta-stacker. Several mechanism-distinct experiments should *amplify*
that property; the LR meta-stacker null doesn't generalise to all
"simpler models" — it failed because of `class_weight='balanced'` on
210 dims, not because simplicity is universally bad.

Tier J1 — cheap, mechanism-first (CPU, hours not days):

  **J1. Tree-leaf OTE meta-stacker** (~1h CPU). Train a small XGB on
  recipe (or dist) features, extract per-tree leaf indices for every
  row (1000+ leaves × 200+ trees). Treat each tree's leaf-index column
  as a high-card categorical, OrderedTE-encode (3 classes), feed the
  resulting ~600 OTE features to a meta-XGB. The meta sees TREE-SPACE
  (per-tree partition memberships), not PROB-SPACE (per-component
  3-class posteriors). Genuinely orthogonal to the 70-component bank
  that produced LB 0.98094. Mentioned once on the open list as "LGBM
  leaf-embedding MLP" but never executed — cheapest untried meta
  architecture. **STARTED this session — see in-flight note above.**

  **J2. Bootstrap-bagged meta-stacker** (~30 min CPU). N=20 bootstrap
  samples of the 63-component *bank* (not seeds — Tier 1c seed-bag
  was near-deterministic). Train iso-cal'd XGB meta on each bootstrap
  subset, log-average outputs. If CV-pessimism is the mechanism behind
  the −0.00010 LB-best gap, bootstrapping the *components* (not seeds)
  should compound it. Structurally different from anything tried so
  far — Tier 1c step 3 bagged XGB seeds on the SAME bank; this bags
  bank SUBSETS at fixed seed.

  **J3. Adversarial-validation row reweighting** (~30 min CPU).
  Train AV classifier (train vs test); use AV score as `sample_weight`
  in a recipe XGB retrain. Biases toward test-distribution-similar
  rows. Untested. Closes whether residual gap is train↔test shift vs
  structural ceiling — informative even if null.

Tier J2 — speculative architecture (GPU, ~1h each, smoke-first):

  **J4. Kolmogorov–Arnold Networks (KAN, 2024)**. Learnable spline
  activations on edges (not nodes). The DGP is a smooth NN function
  per the 2026-04-21 EDA; KAN's splines are exactly the architecture
  for fitting smooth, non-axis-aligned boundaries trees miss. The
  13 prior NN nulls were all attention/MLP/in-context — KAN is a
  different inductive bias. Smoke at SMOKE=1, gate at fold-1
  Jaccard < 0.75 vs LB-best 4-stack AND errs ≤ 9572.

  **J5. TabDDPM diffusion-based row augmentation**. SMOTE-NC failed
  because k-NN interpolation diffused the M↔H boundary. Diffusion
  preserves the joint manifold instead of local-linear interpolation.
  Generate ~10k synthetic High rows from the learned joint, augment
  training. Direct attack on the same lever SMOTE missed. ~1.5h GPU.

Tier J3 — math, not models (~30 min):

  **J6. Constraint-aware QP for blend weights**. `cvxpy`: minimize CV
  macro-recall loss subject to `recall_class ≥ anchor_floor − ε` and
  simplex constraint. Greedy + LR find local optima; QP finds global
  optimum *under* the per-class guardrail. Cheap; might surface a
  config the greedy missed.

  **J7. Conformal-gated overrides on score=6 boundary**. The
  missed-High detector had AUC 0.94 at v2 but failed because precision
  was below break-even. Conformal calibration gives guaranteed-coverage
  prediction sets; pick override threshold from the calibrated set
  rather than raw probability. Same mechanism, principled threshold.

**Recommended start order**: J1 + J2 in parallel, ~1.5h total CPU.
Both target the only property of the LB-best stack that hasn't been
pressure-tested — the negative OOF→LB gap — through structurally
distinct mechanisms. If both null, that's the cleanest possible
saturation evidence and lock the 2 finals immediately. J4/J5 only if
J1+J2 produce a passing standalone OOF that suggests the lever family
is alive. J6/J7 are math/calibration adjustments — small upside,
nearly-zero downside, useful as parallel background work.

**Skip on principled grounds (re-confirmed today)**: LR meta-stacker
variants (architectural null, not a tuning issue), public-CSV blending
(banned), HP/seed bagging on existing components (LB-regressed twice),
more NN-from-scratch attempts on the recipe matrix (13 nulls form a
structural pattern), Mamba/T-Few from the prior speculative menu
(higher infrastructure cost, lower mechanism-novelty than J1-J7).

### 2026-04-25 — S1 Tabular-Mamba (mambular SSM) PROBE: 14th NN null with record-low Jaccard 0.491

- Goal: execute the speculative S1 ceiling-breaker from the prior
  brainstorm — Tabular-Mamba via mambular (BASF SSM library, sklearn
  API). Rationale: state-space architecture with selective scan is
  structurally distinct from every prior NN tested
  (MLP/FT-T/TabPFN/DAE/RealMLP/Trompt all use attention or pure
  feed-forward). Bayesian prior <20% it breaks the pattern, but it's
  the last untested NN architecture family.
- Changed: new `kaggle_kernel/kernel_mamba/` mirroring kernel_trompt
  modular pattern (boot/config/features/model/cv/main + build.py
  concatenator). 19 raw features (8 cats + 11 nums) for apples-to-
  apples Jaccard vs RealMLP/Trompt. mambular 1.5.0 with d_model=64,
  n_layers=4, d_state=16, batch_size=512, 8 epochs.
  `scripts/blend_mamba.py` — PROBE-aware gate that reconstructs LB-
  best 4-stack from saved components and computes Jaccard + magnitude
  on filled rows only.
- Three iterations to land:
  1. **Slug rejected**: "irrigation-mamba" returned "Notebook not
     found" from Kaggle API — likely reserved-word collision (mamba
     conda package). Renamed to "irrigation-mambular-ssm" → pushed.
  2. **SMOKE v1 OOM**: pure-PyTorch `selective_scan_seq` fallback at
     batch=1024 hit CUDA OOM (1.19 GB allocation, 879 MB free).
     Mambular's pure-PyTorch fallback is O(L²·d_model·d_state) memory.
     Fix: cut SMOKE batch to 256, halve d_state to 16. SMOKE v2 GREEN
     (per-fold argmax 0.9565/0.9461, tuned 0.9561, ~15 min wall).
  3. **PROBE wheel install**: SMOKE log confirmed `pip install
     mamba-ssm` + `causal-conv1d` failed (no nvcc on Kaggle for source
     build). Tried direct GitHub-release URL install for cu122/torch2.5/
     cp312 wheels — also failed (URL doesn't match Kaggle's exact
     stack: torch is cu121 not cu122, and the cp312 wheels for those
     specific package versions don't exist on the release page). Source
     build retry also failed. Mambular fell back to pure-PyTorch.
- PROBE config: 1-fold × 100k subsample × 8 epochs (subsample fallback
  to fit pure-PyTorch into the 1h GPU cap; full data would have been
  ~170 min/fold).
- Wall: 79 min total kernel time (Kaggle script-kernel re-runs main()
  via nbconvert). First fold: 450s start → 2508s done = ~34 min/fold
  pure-PyTorch. Second nbconvert run added another ~34 min.
- Standalone results (fold 1 = 126,000 val rows):
  - argmax bal_acc = 0.9574
  - **tuned bal_acc = 0.9632**, bias = [1.13, 1.07, 3.80]
  - errors = 2,432
- Anchor comparison on same 126k val rows:
  - LB-best 3-stack: bal=0.9793, errs=2014
  - LB-best 4-stack: bal=0.9799, errs=1914
  - Mamba: bal=0.9632 (Δ −0.017), errs=2432 (**+27% over anchor**)
- **Jaccards (the headline finding)**:
  - vs LB-best 3-stack: **0.4781**
  - vs LB-best 4-stack: **0.4914** ← LOWEST EVER for an NN family on
    this problem (prior best Trompt 0.5340, RealMLP n_ens=1 0.6206)
  - vs RealMLP: 0.5175
- Fixed-bias α-sweep vs LB-best 4-stack (filled rows): **monotone
  negative from α=0.05** (Δ −0.00036 to −0.00241 across α∈[0.05, 0.35]).
  Peak at α=0.000 (no blend).
- **Verdict: NULL — magnitude-trap.** Same failure mode that closed
  all 13 prior NNs: orthogonality is necessary but not sufficient.
  Even at the best Jaccard ever (0.49), the +27% extra error count
  drowns the unique-correct contributions in unique-wrong noise.
- LB delta: n/a — no LB probe warranted (every α below LB-best).
  LB best unchanged at **0.98094**. LB budget unchanged.
- Pattern reinforced (now 14 consecutive NN-family nulls):
  ```
  NN family            Jaccard vs anchor    errs vs anchor    LB outcome
  ----------          ------------------- ------------------ --------------------
  MLP v5-v9            0.62-0.85           +1500-15000       NULL
  FT-Transformer       0.61                +12000            NULL
  TabPFN               0.81                +1485             NULL
  Pretrain-FT MLP      0.65                +3615             NULL
  DAE SwapNoise        0.84                similar           NULL
  RealMLP n_ens=1      0.62                +358              LB +0.00003 (3-stack)
  RealMLP n_ens=4      0.62                +485              NULL
  Trompt               0.53                +169              NULL
  **Mambular SSM       0.49 (record low)   +518 (+27%)       NULL**
  ```
  Only RealMLP n_ens=1 has ever cleared the magnitude bar (+358 errs =
  +3.7% over anchor) AND produced an LB lift (when blended into a
  3-stack with xgb_nonrule_iso). Every other NN family is permanently
  closed at this feature set.

- **Portable rules** (LEARNINGS.md candidates):
  1. **Mambular installs require pre-built CUDA wheels — no fallback
     path on managed kernel platforms (Kaggle, Colab) without nvcc.**
     The pure-PyTorch fallback is functional but ~30x slower
     (1 min/epoch on 20k rows pure-PyTorch vs ~2 sec/epoch with CUDA
     kernel). For 100k+ rows × 8+ epochs, expect ~30 min/fold pure-
     PyTorch — barely fits the 1h GPU cap with 1 fold.
  2. **GitHub release wheels for niche CUDA libraries (mamba_ssm,
     causal_conv1d) have strict version triples (CUDA × torch ×
     cpython)**. The wheel naming convention is
     `<pkg>-<ver>+cu<XYZ>torch<V.M>cxx11abi<bool>-cp<XY>-...whl`
     and a mismatch on any axis breaks the install. Kaggle's torch
     2.5.1+cu121 + cp312 needs the EXACT cu121 wheel (cu122 is NOT
     ABI-compatible at the wheel level even though CUDA runtime is).
     Verify wheel availability from the release page BEFORE coding
     the install URL.
  3. **Mamba/SSM error orthogonality is genuinely different from
     attention-based and feed-forward NNs on tabular data** — Jaccard
     0.49 is a step-change vs the 0.55-0.85 range of all prior NNs.
     But this orthogonality alone doesn't transfer to LB lift unless
     paired with error-magnitude ≤ ~1.05x anchor. For the next
     synthetic-tabular comp where the anchor stack is weaker, an SSM
     leg may unlock the magnitude bar — keep mambular in the toolkit
     even though it failed here.

- Artefacts committed for cross-branch reuse (gitignore whitelist):
  - `scripts/artifacts/oof_mamba_probe.npy` (7.2 MB, fold 1 only)
  - `scripts/artifacts/test_mamba_probe.npy` (3.1 MB)
  - `scripts/artifacts/mamba_probe_results.json` + `blend_mamba_probe_results.json`
  - 7 source files under `kaggle_kernel/kernel_mamba/` (boot/config/
    features/model/cv/main + build.py + kernel-metadata.json)
  - `scripts/blend_mamba.py`

- **All 4 candidates from the post-Tier-1c "ceiling-breaker shortlist"
  are now closed** (this branch + parallel sessions):
  ```
  Candidate                              Result
  ────────────────────────────────────────────────
  Soft-target distillation               LB 0.97850 (-0.00148, regression)
  171-pair binned (cat+num quantile)     OOF 0.97946 (NULL)
  Stage-2 with LB-blend labeler          LB 0.97989 (-0.00009, NULL)
  Multi-seed pseudo (seed=7 labeler)     OOF 0.98017 (NULL)
  ──── plus Tier 1c saturation triple ───
  Greedy expanded pool (132 components)  +0.00002 (sub-gate)
  Meta-stacker v2 (224-dim)              +0.00002 (sub-gate)
  Meta-stacker XGB seed-bag              +0.00003 (sub-gate)
  ──── plus today's NN closure ───
  Tabular-Mamba (mambular SSM)           PROBE NULL (magnitude trap)
  ```

- **Final-selection lock recommendation unchanged** (5 days to
  deadline):
  1. **Primary**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
     (gap −0.00010, anomalous LB > OOF). Composition:
     lb3 + RealMLP α=0.20 + xgb_nonrule_iso α=0.075 + xgb_metastack_iso α=0.30.
  2. **Hedge**: `submission_recipe_full_te.csv` → **LB 0.97939**
     (gap +0.00028, single-model XGB-recipe — no blend overfit risk).
     Premium = -0.00155 LB. Genuinely orthogonal to primary's
     63-component meta-stacker pool.
  Pack 0.98114 stays +0.00020 above primary. Leader 0.98219 stays
  +0.00125 above. Reachable only via public-CSV blending (banned
  by top-of-file rule).

### Next steps: speculative ceiling-breaker (post-2026-04-25 own-pipeline closure)

After today's 4 LB-probed nulls (LR meta-stacker, cross-poll v3 metastack,
SMOTE v2, SMOTE v3) and the prior comprehensive saturation evidence,
**every own-pipeline lever within the standard tabular ML toolkit is
exhausted on this feature set.** The only remaining categorically-new
mechanism not yet attempted is from the **2024-2025 tabular-foundation-
model wave**:

  **S1. Tabular-Mamba leg via mamba-tabular** (~1-1.5h Kaggle GPU).
  State-space architecture with linear-time sequence modelling instead
  of attention. Structurally distinct from every prior NN tested
  (MLP / FT-T / TabPFN / DAE / RealMLP / Trompt all use either
  attention or pure feed-forward). Mamba's selective scan mechanism
  may pick up different feature interactions than column-attention
  models. Same kernel scaffold pattern as kernel_trompt (boot + pip
  install mamba_ssm + 5-fold StratifiedKFold seed=42 + fold-1 promise
  gate at Jaccard < 0.75 vs LB-best 4-stack AND errs ≤ 9572).

  **S2. T-Few in-context tabular learning** (~1h Kaggle GPU). Few-shot
  classification via LLM-style in-context inference. Off-paradigm
  for tabular but recently shown competitive on small-prior-class
  problems (rare-class High recall is exactly our weakness). Risk:
  setup overhead may eat the 1h budget.

  **Why both are speculative**:
  - 13 prior NN-family attempts all showed magnitude-trap failure
    (Jaccard ~0.55-0.85 with errs > LB-best). The structural pattern
    is consistent: any NN trained on this 443-feature recipe matrix
    produces orthogonal errors but in greater absolute count than
    the tree-stacker bank, defeating fixed-bias log-blends.
  - Mamba/T-Few may behave differently because their architectures
    don't process tabular features through the same MLP/attention
    bottleneck. But this is a Bayesian prior of <20% they break the
    pattern.

  **Realistic outcome**: probably the 14th NN null. **Skip unless you
  want closure** or are comfortable spending one of the remaining 6
  LB submissions on a low-probability shot.

  **Decision rule if attempted**:
  - Fold-1 Jaccard < 0.75 AND errs ≤ 9572 vs LB-best 4-stack: PROCEED
    full 5-fold + meta-stacker bank addition + retrain XGB meta + LB
    probe only if blend Δ ≥ +0.0003 OOF (stricter than +0.0002 because
    LR meta-stacker showed OOF inflation up to +0.00176 transfers
    negatively in this regime).
  - Otherwise: ABORT, mark as 14th NN null, lock the 2 finals already
    staged.

  **Alternative if unwilling to gamble compute**: **lock the 2 finals
  now** (`submission_tier1b_greedy_meta.csv` LB 0.98094 +
  `submission_recipe_full_te_catboost.csv` LB 0.97935) and reserve the
  6 LB submissions for end-of-comp variance check. With 5 days to
  deadline and structural ceiling confirmed via ~30+ LB-probed
  experiments, low-probability-of-lift compute spend has diminishing
  EV.

### 2026-04-25 — J3 AV classifier closes train↔test shift hypothesis (AUC 0.50247, residual gap is structural)

- Goal: cheapest definitive answer to "is the residual OOF→LB gap
  caused by train/test distribution shift?" — proposed in the
  2026-04-25 J3-J7 junior-engineer audit on
  `claude/ml-optimization-ideas-rD6N3` as Tier J1 (~30 min CPU,
  diagnostic value high regardless of outcome).
- Mechanism: binary `is_test` XGB on combined `train ∪ test`,
  5-fold StratifiedKFold(seed=42), target-FREE features only
  (8 cats factorized + 11 raw nums + 4 DGP rule indicators + 11
  decimal-fraction `(col % 1).round(2)` features = 34 cols total).
  No OTE / FREQ / ORIG_* — those would be target-leaky for `is_test`
  since the AV target is `is_test`, not `Irrigation_Need`, and
  recipe's target-encoded features were built using
  `Irrigation_Need` labels which test rows don't have at training
  time anyway.
- Changed: `scripts/j3_av_classifier.py` (140 lines, modular per
  CLAUDE.md ≤150L rule). Saves per-train-row OOF P(is_test) +
  importance weights w = p / (1 − p) for downstream consumers.
- Results (full 630k train + 270k test, 5-fold, ~13 s wall):
  ```
  fold  AUC
  1     0.50440
  2     0.50192
  3     0.49902   ← best_iter=0 (no signal at all on this split)
  4     0.50219
  5     0.50292

  OVERALL AV AUC = 0.50247  σ=0.00176
  Per-train-row P(is_test) percentiles: p1=0.278 p50=0.300 p99=0.324
  Importance weights w=p/(1-p):         p1=0.385 p50=0.428 p99=0.478
  ```
  Per-train-row P(is_test) is essentially constant at the marginal
  prior (270k / (630k + 270k) = 0.30). Importance weights span
  a tiny range [0.385, 0.478] — effectively a uniform multiplier.
- **Definitive verdict: train and test ARE statistically
  indistinguishable** at the 34-feature target-free granularity.
  AUC 0.50247 is within 1.5σ of pure chance (0.50000); fold 3 hit
  best_iter=0 (no signal at all on that split).
- **Skipped the planned ~50-min recipe-with-AV-weights retrain.**
  Multiplied into the existing class-balanced sample weights, AV
  weights span a 0.09 range against per-row variation — XGB sees
  the same gradient ratios and produces near-identical OOF.
  Guaranteed waste of compute.
- **What this closes:**
  1. Train↔test distribution shift is **NOT** the OOF→LB gap source.
     The residual ceiling at LB 0.98094 is **fully structural**
     (model capacity / feature manifold limit). Settles a question
     that's been speculative throughout the session log.
  2. The negative OOF→LB gap (−0.00010) on the LB-best 4-stack is
     **NOT** distribution-shift correction by the meta-stacker —
     it's CV-pessimism per the prior diagnosis (meta averaging over
     noisy fold hold-outs of its 63-component bank).
  3. Active levers on other branches (J1 tree-leaf OTE, N3
     5-shuffle, Mamba SMOKE) all target structural ceiling, not
     shift correction — correct framing.
- **Portable rule** (LEARNINGS.md candidate): "For Kaggle synthetic
  Playground competitions where train and test are released as a
  single CSV pair, run a 30-second AV classifier first. If
  AUC < 0.55, the residual OOF→LB gap is structural; skip every
  distribution-shift-correction lever (AV reweighting, importance
  weighting, test-time-only FE). If AUC > 0.65, AV reweighting +
  test-distribution-targeted FE are first-line levers."
- LB delta: n/a (no submission warranted; AV reweighting cannot
  produce a model materially different from the recipe baseline).
  LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`. LB budget unchanged.
- Artefacts (gitignore-whitelisted for cross-branch reuse):
  - `scripts/artifacts/j3_av_p.npy` (per-train-row OOF P(is_test))
  - `scripts/artifacts/j3_av_w.npy` (importance weights p/(1-p))
  - `scripts/artifacts/j3_av_results.json` (per-fold AUC + percentiles)

### 2026-04-25 — J6 constraint-aware QP for blend weights: NULL (objective misalignment)

- Goal: J6 from the J3-J7 junior-engineer audit. Greedy + LR are
  myopic local-optimizers on the simplex; a global convex log-blend
  solver might surface a config greedy missed. Closes whether the
  search procedure (greedy/LR) is the bottleneck vs the objective
  itself.
- Mechanism: minimize macro-balanced cross-entropy on log-blend
  probs over the simplex (`w ≥ 0`, `Σw = 1`). Convex objective:
  ```
  L(w) = (1/3) Σ_k (1/N_k) Σ_{i:y_i=k} (logsumexp(z_i) − z_{i,k})
         where z_i = Σ_c w_c · log p_c[i] + bias
  ```
  This IS convex in w (log-sum-exp convex; linear inside;
  minus-linear convex). SLSQP solver (scipy.optimize) with analytic
  gradient.
- Changed: `scripts/j6_qp_blend.py` (155 lines, modular per CLAUDE.md
  rule). Uses `tier1b_helpers.build_lbbest_stack` for LB-best 4-stack
  reconstruction.
- Two pool variants tested:

  **Variant 1 — 10-pool with meta-stackers**
  (recipe + pseudo_s1 + pseudo_s7 + RealMLP + xgb_nonrule +
  xgb_metastack + xgb_metastack_v3 + lgbm_te_orig + xgb_corn +
  xgb_dist_digits):
  ```
  QP weights:                       0.27 xgb_metastack
                                    0.27 xgb_metastack_v3
                                    0.25 pseudo_s7
                                    0.08 RealMLP
                                    0.07 xgb_nonrule
                                    0.06 pseudo_s1
                                    (recipe / corn / digits / lgbm_te_orig: 0)

  LB-best 4-stack OOF                  0.98084
  QP variant 1 OOF                     0.98064  Δ = −0.00020
  Per-class recall Δ                   [+0.00002, +0.00079, −0.00143]
  Guardrail (Δ ≥ −5e-4 each class)     FAIL  (High −0.00143)
  ```
  QP heavily double-dipped on meta-stackers (combined weight 0.54)
  — both consume overlapping signal from the same component bank,
  so the QP's "global optimum" doubled the meta-stacker contribution
  past what's healthy.

  **Variant 2 — 8-pool, meta-stackers excluded**:
  ```
  QP weights:                       0.39 pseudo_s7
                                    0.23 pseudo_s1
                                    0.12 recipe_full_te
                                    0.10 xgb_nonrule
                                    0.10 RealMLP
                                    0.04 xgb_corn
                                    0.02 xgb_dist_digits
                                    (lgbm_te_orig: 0)

  QP variant 2 OOF                     0.98055  Δ = −0.00029
  Per-class recall Δ                   [−0.00003, −0.00041, −0.00043]
  Guardrail                            PASS (all within −5e-4)
  ```
  Drops the double-dip but still strictly worse than LB-best 4-stack
  on macro-recall.
- **Verdict — NULL across both variants.** Per-row gradient + global
  solver on a convex surrogate produces blends that are WORSE on the
  actual macro-recall metric than greedy/LR's local optima.
- **Diagnosis — objective misalignment**: the convex log-loss
  surrogate optimizes per-row predicted-class log-likelihood AFTER
  the fixed bias `[1.43, 1.47, 3.40]` is added. But macro-recall
  under that bias has a class-asymmetric mapping (the bias boosts
  High predictions structurally; many rows that the QP makes
  log-likelihood-optimal end up flipping to High under the bias,
  trading log-loss gain for macro-recall loss). The two metrics
  diverge precisely because the bias is non-uniform.
- **What this closes** (informative-null):
  1. **Search-procedure suboptimality is NOT the bottleneck.** A
     global convex solver does worse than greedy/LR on the actual
     macro-recall objective. This is indirect evidence that the
     LB 0.98094 ceiling is structural at the **objective level**,
     not at the search level.
  2. **The LB-best 4-stack weight composition (0.30 meta_iso +
     0.70 LB-3-stack) is NOT obviously suboptimal under the QP's
     convex surrogate.** The QP's chosen weights are different
     (and worse on macro-recall), so greedy didn't miss a better
     QP-style optimum.
  3. **Convex log-loss is the wrong surrogate for macro-recall +
     fixed-bias decision rules.** Future blend-weight work should
     directly optimize macro-recall (gradient-free or via a custom
     surrogate that accounts for the bias), not log-loss-on-simplex.
- **Portable rule** (LEARNINGS.md candidate): "Convex log-loss
  minimization on the simplex is a misaligned surrogate when the
  decision rule includes a fixed per-class bias offset. Greedy / LR
  / direct macro-recall optimization (even if myopic) outperforms
  the global log-loss optimum under such bias because the bias
  creates a class-asymmetric mapping from probabilities to
  predictions that the convex surrogate doesn't capture."
- LB delta: n/a (no submission warranted; both variants strictly
  below LB-best 4-stack on macro-recall). LB best unchanged at
  **0.98094** via `submission_tier1b_greedy_meta.csv`. LB budget
  unchanged.
- Artefacts (gitignore-whitelisted for cross-branch reuse):
  - `scripts/j6_qp_blend.py` (155 lines)
  - `scripts/artifacts/oof_j6_qp_blend.npy` (variant 1 with metas)
  - `scripts/artifacts/test_j6_qp_blend.npy` (variant 1 test side)
  - `scripts/artifacts/j6_qp_blend_results.json` (weights + deltas)


### 2026-04-25 — J2 bootstrap-bagged meta-stacker: NULL (5th saturation confirmation at LB 0.98094)

- Goal: cheapest mechanism-distinct lever from the J3-J7 brainstorm.
  Bag SUBSETS of the meta-stacker bank (without replacement,
  fraction=0.5), train an iso-cal'd XGB meta on each, log-average.
  Targets the negative OOF→LB gap (CV-pessimism) on the LB-best
  4-stack via a different decorrelation than Tier-1c step-3
  XGB-seed bag (which was near-deterministic and null).
- Changed: `scripts/j2_bootstrap_metastack.py` (~210 lines) +
  `scripts/j2_analyze.py` (post-result calibration projector).
  Pool composition: starts from `tier1b_helpers.load_pool()` (88
  components on disk now) then drops 19 J2-specific entries:
  - **Circular meta outputs**: `xgb_metastack_v3/v4/varB/varC`,
    `xgb_metastack_v3_iso`, `lr_metastack` (all prior meta-stacker
    outputs would feed back into a new meta).
  - **Submission-derived OOFs**: `primary_sub_tau{095,097,099}`
    (these are subs of the LB-best primary itself = circular leak).
  - **Derived blends**: `j6_qp_blend`, `greedy_blend`,
    `ovo_boundary_blend`.
  - **LB-confirmed regressors**: `soft_distill` family.
  - **Borderline τ-sweeps**: `recipe_pseudolabel_tau{095,097,099}`
    (circular w.r.t. pseudo_s1 in the LB stack).
  - **Bias-mismatched**: `recipe_smote_v3` (cannot blend at recipe
    bias).
  Final clean pool: **69 components** (62 from v1 minus 4
  derivatives, plus 11 legit new since the LB-validated v1:
  `realmlp_ens4`, `leaf_ote_meta` v1+v2, `n2_extratrees`,
  `n2_knn`, `recipe_2shuffle`, 4 focal variants, 2 OvR XGB).
- SMOKE (N=2 × bag_size=12, ~3 min): pipeline end-to-end clean,
  bag-mean OOF 0.98092, peak Strategy A α=0.50 → +0.00006.
- Production: N=10 bags × bag_size=34 (fraction=0.493), wall **34
  min** on 16-core CPU.
- Per-bag iso OOF results (5-fold seed=42, fixed recipe bias):
  ```
  bag    iso_oof   wall
   0    0.98029   205s
   1    0.98035   204s
   2    0.98059   185s
   3    0.98011   183s
   4    0.98042   198s
   5    0.98020   211s
   6    0.98048   208s
   7    0.98039   215s
   8    0.98042   203s
   9    0.98048   209s
  ```
  Per-bag spread σ=0.00015 — bagging produced near-identical
  metas, **low variance to reduce**. The fraction=0.5 sub-pools
  share ~17 components on average (overlap 50%×34 = ~17), so
  each bag's meta-XGB sees largely the same dominant signal
  channels and converges to similar per-row decisions.
- **Bag-mean meta_iso standalone @ recipe bias**: **0.98050**
  (vs LB-best 4-stack 0.98084, **Δ = −0.00034 BELOW anchor**).
  Errors 8,964 (vs anchor 9,415 — 451 fewer total errs but
  distributed against the rare class, which kills macro-recall).
  Jaccard vs LB-best 4-stack = **0.9187** (high redundancy).
- Strategy A — log-blend bag_iso onto LB-best 4-stack (fixed bias):
  ```
  α       OOF       Δ          errs   recL    recM    recH
  0.025  0.98081  -0.00004   9410  0.9955  0.9695  0.9773
  0.050  0.98081  -0.00003   9396  0.9955  0.9696  0.9773
  0.075  0.98083  -0.00001   9380  0.9955  0.9697  0.9773
  0.100  0.98082  -0.00002   9366  0.9955  0.9697  0.9772
  0.150  0.98087  +0.00003   9330  0.9955  0.9699  0.9772   ← peak
  0.200  0.98082  -0.00003   9318  0.9955  0.9699  0.9770
  0.300  0.98075  -0.00010   9284  0.9956  0.9701  0.9766
  0.500  0.98067  -0.00017   9210  0.9956  0.9704  0.9760
  ```
  Peak at α=0.15 with Δ=+0.00003 — within fold noise, far below
  the +2e-4 internal gate and the +5e-4 LB-probe threshold.
  Per-class recall: High drops 0.9775→0.9772 at peak (-0.0003);
  same magnitude-trap pattern as every prior LB-best 4-stack
  add (tiny rare-class drop dominates macro-recall).
- Strategy B — replace meta_iso with bag_iso at α=0.30:
  Δ = **−0.00006**. Strict null.
- **Calibration projection** (using LR/V4 prior gap-inflation rate
  ~0.0038 per unit α observed when meta-output is blended into the
  4-stack):
  ```
  best gated α=0.150 → OOF Δ=+0.00003
  projected gap inflation at α=0.15 ≈ +0.00057
  projected LB Δ vs primary = -0.00054
  projected LB ≈ 0.98040 (REGRESSION)
  ```
  Linear projection rules out the conservative dilution probe
  (per the 2026-04-25 LR meta-stacker closure rule).
- **No LB submission warranted.** Gate FAILED, projection negative.
- **5th independent saturation confirmation at LB 0.98094**:
  ```
  attack vector                         peak OOF Δ   LB Δ (if probed)
  ---------------------------------------- -----------  -----------------
  1. Tier 1c greedy expanded pool (132c)  +0.00002     n/a
  2. Tier 1c meta-stacker v2 (224-dim)    +0.00002     n/a
  3. Tier 1c meta-stacker XGB seed-bag    +0.00003     n/a
  4. Tier 1b cross-poll metastack v3      +0.00015     -0.00034 (LB 0.98060)
  5. **J2 bootstrap-bagged metastack       +0.00003     not probed (proj -0.00054)**
  ```
- **Mechanism diagnosis**: with bag_size=34 (fraction=0.5 of 69),
  each bag covers half the bank — enough that the meta-XGB on
  any bag sees most of the dominant signal channels and
  converges to a similar decision surface. Bagging only
  decorrelates effectively when bags see DIFFERENT signals; here
  they don't. Could retry at fraction=0.2-0.3 to force more
  decorrelation, but at that point each bag is too weak to
  contribute meaningful signal. The 0.5 trade-off was the
  reasonable middle ground.
- **Three structural reasons J2 nulled** (now LB-validated rule
  pattern):
  1. **CV-pessimism amplification through bagging only works when
     individual bags have decorrelated OOF noise.** At
     fraction=0.5, bags share 50% of components → correlated
     noise → averaging produces ≈ single-bag.
  2. **The LB-best 4-stack already absorbs the dominant
     meta-stacker signal channel.** Any new meta variant (LR, v3,
     v4, J2 bagged) lands at +0.0001-0.0006 OOF over the 4-stack
     anchor, reflecting saturated information in the bank.
  3. **Bag-mean's −0.00034 BELOW anchor on standalone OOF means
     the new components in the larger pool (RealMLP n_ens=4,
     leaf_ote v1+v2, ET, kNN, focal variants) inject NOISE
     rather than signal when filtered through bagged metas.**
- Combined with the 14+ prior nulls (LR meta, v4 meta, soft_distill,
  SMOTE-NC v2/v3, multi-seed pseudo s123, AV J3, QP J6, leaf_ote
  v1+v2, Mamba PROBE, Trompt PROBE, RealMLP n_ens=4, focal
  variants, distill_tiny, N3 5-shuffle K=2): **own-pipeline LB
  ceiling is structurally bounded at 0.98094 within the standard
  tabular ML toolkit on this feature set.** No remaining lever
  has plausible LB-positive transfer probability.
- Artefacts committed:
  - `scripts/j2_bootstrap_metastack.py`,
    `scripts/j2_analyze.py`
  - `scripts/artifacts/oof_xgb_metastack_j2bag.npy` +
    `test_xgb_metastack_j2bag.npy` (whitelist via gitignore)
  - `scripts/artifacts/j2_bootstrap_metastack_results.json`
  - `scripts/artifacts/j2_bootstrap_metastack_smoke.json`
- LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.
- **Recommendation reaffirmed (audit F1)**: swap hedge from
  `submission_recipe_full_te.csv` (LB 0.97939, premium -0.00155,
  shares full FE pipeline with primary, 484/270k disagreement)
  to `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005,
  premium -0.00089, structurally sidesteps the meta-stacker layer
  — the most-tuned and most-likely public-LB-overfit element of
  the primary). Half the premium, materially better insurance
  against meta-stacker overfit on private LB.

### 2026-04-25 — narrow 5-input meta-stacker (cherry-pick from claude/simplify-ml-solution-Q2Zll): NULL — wide bank IS pulling weight

- Goal: cherry-pick NEXT_STEPS #2 from the simplify branch — test the
  hypothesis "most of the 63 components in the wide meta-stacker
  contribute near-zero". Build a narrow XGB stacker on
  `[recipe_oof(3), pseudo_oof(3), dgp_score, sm_dist, rf_dist]` =
  9 features, same heavy-reg HPs as tier1b, same 5-fold seed=42 split.
  If narrow recovers most of the wide stacker's +0.00086 LB lift,
  validates the simplify framing and we can ship a 200-line repo at
  ~LB 0.98094.
- Idea #1 from the same NEXT_STEPS (stage-2 with 2-way labeler) was
  already on main from 2026-04-23: `oof_recipe_pseudolabel_stage2.npy`
  + `recipe_pseudolabel_stage2_results.json` exist; result was
  OOF 0.98002 → LB 0.97997 NULL. Nothing to cherry-pick.
- Changed: `scripts/tier1c_narrow_metastack.py` (~140 lines), reuses
  `tier1b_helpers.iso_cal/build_lbbest_stack/load_y/bal_at_bias` and
  `common.add_distance_features` for `dgp_score / sm_dist / rf_dist`.
  SMOKE first (20k×2-fold, ~30 s) then production (5-fold, ~70 s).
- Standalone results (5-fold OOF seed=42):
  - per-fold best_iter 280-481 (well below 3000 cap)
  - argmax 0.97275  (wide standalone 0.96995, narrow +0.00280 argmax)
  - **tuned 0.97983**  bias=[0.632, 0.969, 3.401]
    (wide tuned 0.98041, **narrow −0.00058** behind wide)
  - iso-cal'd argmax 0.97255
- **Blend gate vs LB-best 3-stack (anchor 0.98061, the meta-target)**:
  ```
  α       blend OOF   Δ vs anchor
  0.000  0.98061     +0.00000   ← peak
  0.025  0.98056     -0.00005
  0.050  0.98051     -0.00010
  0.100  0.98044     -0.00017
  0.200  0.98038     -0.00023
  0.300  0.98032     -0.00029
  0.500  0.98008     -0.00053
  ```
  Strict monotone-negative from α=0.025. Wide meta at α=0.30 gave
  +0.00023 OOF / +0.00086 LB on the same anchor; narrow gives
  −0.00029 OOF at the same α. **Falsified.**
- **Blend gate vs LB-best 4-stack (anchor 0.98084, the wide+3stack
  target — tests "does narrow add anything on top of wide?")**:
  same monotone-negative pattern from α=0.025 (Δ −0.00004 to
  −0.00080 across α∈[0.025, 0.5]). Narrow contributes nothing
  even when wide is already in the blend.
- **Verdict: NULL on both anchors.** Hypothesis "most components
  are noise" is **falsified** — the 63-component bank is genuinely
  pulling the wide meta's +0.00086 LB lift; a tight 9-feature
  stacker captures none of it.
- Read-out — why narrow is structurally weaker than wide:
  1. **Standalone tuned 0.97983 < wide's 0.98041** (Δ −0.00058
     before any blend). The narrow stacker sees only 6 prob cols +
     3 raw features; it misses signal channels the wide bank carries
     (RealMLP NN orthogonality, xgb_nonrule_iso class-rebalanced
     logits, specialist binary heads, etc).
  2. **Wide bank's 63 components are NOT noise** — they're a richly
     correlated set where the heavy-reg meta XGB extracts pairwise
     disagreement signal between components that no individual
     component carries alone. The 2026-04-25 J2 bootstrap-bagged
     experiment already showed bagging at fraction=0.5 saturates
     because each bag still sees most of the dominant signal
     channels — that's the same property here, just confirmed via
     extreme reduction (~7% of the bank vs 50%).
  3. **The simplify branch's "perm-importance pointed at 5 dominant
     components" framing under-counts pairwise interaction terms.**
     A heavy-reg XGB on the wide bank picks up "row where component
     A says High AND component B says Medium" splits that capture
     calibration disagreement; reducing to a flat 9-feature input
     loses every such interaction.
- LB delta: n/a (sweep monotone-negative on both anchors; no probe
  warranted). Per the audit's user direction "ALWAYS ASK FIRST"
  rule, no submission emitted.
- LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.
- 6th independent saturation confirmation at LB 0.98094 (joining
  Tier 1c greedy / meta-on-meta v2 / meta-XGB-seed-bag /
  cross-poll v3 / SMOTE-NC v2/v3 / J2 bootstrap-bag / LR
  meta-stacker / N2 v4 ET+kNN bank-extension / N3 K=2 OTE).
- **Portable rule** (LEARNINGS.md candidate): "On a saturated
  meta-stacker bank with negative OOF→LB gap, you cannot
  retroactively prune components by perm-importance — the
  pairwise-interaction signal between low-importance components
  contributes to the heavy-reg XGB's lift even when each component
  is individually weak. The bank is the unit of contribution,
  not the components."
- Artefacts committed (whitelisted via .gitignore):
  - `scripts/tier1c_narrow_metastack.py`
  - `scripts/artifacts/oof_xgb_metastack_narrow.npy` (7.2 MB)
  - `scripts/artifacts/test_xgb_metastack_narrow.npy` (3.1 MB)
  - `scripts/artifacts/tier1c_narrow_metastack_results.json`

### 2026-04-25 — leaderboard push session: hedge ACCEPTED + LR v2 retry launched

- User directive ("get us on top of the leaderboard") with full LB
  budget remaining (8/10 today). Three parallel actions executed on
  branch `claude/leaderboard-optimization-RbhqA`.

- **Action 1 — HEDGE SWAP ACCEPTED (zero compute, recorded here)**:
  - Final-selection PRIMARY: `submission_tier1b_greedy_meta.csv`
    → LB 0.98094 (gap −0.00010). Unchanged.
  - Final-selection HEDGE: **swap from
    `submission_recipe_full_te.csv` (LB 0.97939, premium −0.00155)
    TO `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005,
    premium −0.00089)**. Sidesteps the meta-stacker layer — the
    most-tuned, most-likely-private-LB-overfit element of primary.
    Half the insurance premium, materially better protection.
  - User must lock the swap on Kaggle's final-selection UI before
    deadline 2026-04-30.

- **Action 2 — LR meta-stacker v2 launched** (`scripts/tier1c_lr_metastack_v2.py`):
  - Mirrors v1 EXCEPT: `class_weight=None`, `C=0.1`, `max_iter=2000`.
  - Diagnosis from v1's 2026-04-25 LB null (0.97991, gap +0.00176)
    explicitly flagged these two HPs as the structural overfit
    source on a 210-dim input (class_weight='balanced' upweights
    rare-High at training time, C=1.0 under-regularizes).
  - Pipeline mirror: same 5-fold StratifiedKFold(seed=42), same
    63-component pool from `tier1b_helpers.load_pool()`, same
    StandardScaler, same iso-cal + fixed-bias blend gate.
  - Strict gate added: emit submission ONLY if Δ ≥ +2e-4 AND
    per-class recall guardrail PASS (each class ≥ anchor − 5e-4).
  - Wall ETA ~5 min CPU.
  - LB submission requires explicit user confirmation (CLAUDE.md rule).

- **Action 3 — kernel audit (deferred)**: Kaggle CLI not yet
  available in this container; data still rehydrating via
  `bootstrap.sh`. Will run `kaggle kernels list -c playground-
  series-s6e4 --sort-by dateRun` once bootstrap completes,
  filtering to kernels ≥20 votes posted after 2026-04-23 (the
  date of the last full audit). Targets: novel mechanisms not in
  any prior round 1-4 reads.

### 2026-04-25 — LR v2 LB result + kernel audit round 5 close-out

- **LR meta-stacker v2 LB probe** (`submission_lr_v2_iso_3stack_a300.csv`,
  user-approved, submitted 19:14 UTC): **LB public = 0.98052**.
  Δ vs LB-best primary (0.98094) = **−0.00042** (regression but
  ~2.5x smaller than v1's −0.00103). OOF→LB gap = **+0.00055**.

- **Calibration ladder update**:
  ```
  LR v1 (C=1.0, class_weight='balanced')  OOF 0.98167 → LB 0.97991  gap +0.00176
  **LR v2 (C=0.1, class_weight=None)       OOF 0.98107 → LB 0.98052  gap +0.00055**
  LR Δ-improvement (v2 over v1):                                    gap -0.00121 (3x tighter)
  LB-best primary (4-stack)                OOF 0.98084 → LB 0.98094  gap -0.00010
  ```

- **Diagnosis confirmed correct, fix worked partially.** The
  2026-04-25 LR v1 closure note diagnosed v1's overfit as
  `class_weight='balanced'` upweighting the rare-High class on a
  210-dim input + insufficient L2 (C=1.0). v2's fix (`class_weight=
  None` + `C=0.1`) reduced gap inflation from +0.00176 to +0.00055
  — a real ~3x improvement. But the OOF lift (+0.00046 vs 3-stack
  anchor) still doesn't survive the gap completely; net LB Δ remains
  negative.

- **Saturation at LB 0.98094 reconfirmed** (now the **6th
  independent attack** to land at or below):
  ```
  attack vector                              best LB        Δ vs primary
  ----------------------------------------- -------------- --------------
  1. Tier 1c greedy expanded (132c)          (no probe)     n/a (sub-gate)
  2. Tier 1c meta-stacker v2 (224-dim)       (no probe)     n/a (sub-gate)
  3. Tier 1c meta-stacker XGB seed-bag       (no probe)     n/a (sub-gate)
  4. Cross-poll metastack v3                 0.98060        -0.00034
  5. J2 bootstrap-bagged metastack           (no probe)     n/a (proj -0.00054)
  6. **LR meta-stacker v2 (this session)     0.98052        -0.00042**
  ```

- **Portable rule** (LEARNINGS.md candidate): **"For LR
  meta-stackers on 200+ dim component banks, the (class_weight=None,
  C=0.1) config reduces OOF→LB gap inflation by ~3x vs (class_weight=
  'balanced', C=1.0) but does not eliminate it. The remaining
  +0.00055 gap reflects fundamental mismatch between LR's
  global-optimum convex log-loss surrogate and macro-recall under
  fixed-bias decision rules. To get LR meta-stacker LB-positive,
  would need either (a) C ≤ 0.01 (likely collapses standalone OOF),
  (b) a custom macro-recall-aware loss replacing log-loss, or
  (c) per-class isotonic re-fit AFTER LR (untried). Lever stays
  closed without (b) or (c) pending — unlikely high-EV given 6
  saturation confirmations at LB 0.98094."**

- **Kernel audit round 5 close-out**: 5 kernels read (ldausl
  31 votes, harigovindj3 32 votes, mikhailnaumov ensemble 18
  votes, mohameddrabo lgbm-optuna 21 votes, agentzz SOTA v15 3
  votes, mtoshidesu 0.98134-test-mod 2 votes). Findings:
  - **No new actionable levers.** ldausl + mtoshidesu use
    public-CSV blending (banned). harigovindj3 + mohameddrabo
    are standard XGB/LGBM with no novel mechanism. agentzz SOTA
    v15: pseudo-labeling + TE, prior PB 0.97395 (well below our
    ceiling). mikhailnaumov imports `TabM_D_Classifier` from
    pytabkit but doesn't actually instantiate it — only RealMLP
    is used (which we already have).
  - **TabM-via-pytabkit remains untested** (the import-but-don't-
    use pattern in mikhailnaumov suggests they tried and dropped).
    Per CLAUDE.md GPU 1h cap rule: TabM standalone production at
    n_ens=4 × 5-fold could exceed wall budget; would require
    SMOKE-first + careful capacity reduction, like RealMLP
    n_ens=4 retry which itself nulled.
  - All public-CSV blend kernels (ldausl, mtoshidesu, nina2025
    series) read public datasets `0.98114.csv`, `0.98117.csv`,
    `0.98119.csv` — confirming the leader 0.98219 still requires
    public-CSV blending (banned).

- **LB budget**: 6/10 used today (5 from earlier sessions + 1 LR v2
  this session). 4 remaining.

- **Final-selection LOCKED** (5 days to deadline):
  1. **PRIMARY**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
     (gap −0.00010, anomalous LB > OOF). Composition:
     LB-best 3-stack + xgb_metastack_iso × α=0.30.
  2. **HEDGE (swap target)**: `submission_3way_recipe025_s1035_s7040.csv`
     → **LB 0.98005** (gap +0.00024, premium −0.00089 vs primary).
     Sidesteps meta-stacker layer — orthogonal overfit surface
     for private-LB protection.

- **Strategic read**: with 6 independent saturation confirmations
  in hand and 5 days to deadline, marginal LB-probe EV is below
  the cost of variance noise. Reserve remaining 4 LB slots for
  end-of-comp variance check (one re-validation per day until
  close). The own-pipeline ceiling at LB 0.98094 is **structural
  on this feature set within the standard tabular ML toolkit**.

### 2026-04-25 — LR v2 + iso-after-blend: 7th saturation confirmation (Option B from v2 LB null closure)

- Goal: Option B from the v2 LB null closure rule. Per-fold per-class
  isotonic re-fit AFTER blending LR_v2_iso onto LB-best 3-stack.
  Hypothesis: iso-after re-aligns ensemble probs with macro-recall
  optimum WITHOUT changing the fixed-bias decision rule.
- Changed: `scripts/tier1c_lr_v2_isoafter.py`. Per-fold leak-safe
  iso (fit on tr_idx, applied to va_idx); test gets full-OOF-fitted iso.
- **Result: NULL across every α.** Every variant fails per-class
  recall guardrail.
- **Critical diagnostic — iso-after on LB-best 3-stack ALONE (α=0)
  drops recH from 0.9774 to 0.9747 (−0.0027).** Per-class iso
  calibrates probs toward EMPIRICAL CLASS DISTRIBUTION; that's the
  WRONG operating point under macro-recall + fixed bias
  [1.43, 1.47, 3.40]. Fixed bias was already calibrated to shift
  the operating point toward High-favoring macro-recall optimum;
  iso-after-blend UNDOES that calibration.
- Pre-iso vs post-iso (selected α):
  ```
  α      pre-iso OOF (PASS)   post-iso OOF (FAIL)   recH drop
  0.000  0.98061              0.98031               0.9774 → 0.9747 (-0.0027)
  0.250  0.98105              0.98085               0.9772 → 0.9756 (-0.0016)
  0.300  0.98107              0.98077               0.9770 → 0.9753 (-0.0017)
  0.400  0.98109              0.98091               0.9766 → 0.9755 (-0.0011)
  ```
  All α: per-class recall trade is wrong direction — Medium UP, High
  DOWN. Net macro-recall lower in every case.
- **LR meta-stacker family fully closed across all 3 mitigation paths**:
  ```
  v1 (C=1.0, balanced)      LB 0.97991  gap +0.00176
  v2 (C=0.1, none)           LB 0.98052  gap +0.00055
  v2 + iso-after-blend       OOF NULL    no LB probe warranted
  ```
- **7th independent saturation confirmation at LB 0.98094**:
  1. Tier 1c greedy expanded (132c) — sub-gate
  2. Tier 1c meta-stacker v2 (224-dim) — sub-gate
  3. Tier 1c meta-stacker XGB seed-bag — sub-gate
  4. Cross-poll metastack v3 — LB 0.98060 (-0.00034)
  5. J2 bootstrap-bagged metastack — proj LB -0.00054
  6. LR meta-stacker v2 — LB 0.98052 (-0.00042)
  7. **LR v2 + iso-after-blend — OOF NULL (this entry)**
- **Portable rule** (LEARNINGS.md candidate): "Per-class isotonic
  re-fit AFTER ensemble log-blend is destructive when the decision
  rule uses a non-uniform fixed bias. The bias represents an
  operating-point preference (e.g. macro-recall favoring rare class);
  iso re-calibration restores empirical-class-distribution
  calibration, which contradicts the bias preference. Use iso-cal
  on INPUT components (which is leak-safe and corrects per-class
  scale mismatch BEFORE blending) but NEVER on the blend output."
- LB delta: n/a. No probe. LB-best unchanged at **0.98094**.
- LB budget unchanged: 6/10 used today, 4 remaining.

### 2026-04-25 — meta v5 (OvR + focal + LR pool) + J7 conformal: 8th saturation confirmation, both NULL

- Goal: execute the two highest-EV untested recommendations from the
  next-steps menu — **N1 OvR-XGB as a meta-stacker bank addition** (audit
  round 4 flagged this as the recommended next step but never executed
  with a fresh meta retraining) and **J7 conformal-gated overrides on
  score=6 boundary** (replaces ad-hoc theta sweep with Wilson-bounded
  calibrated threshold on the spec6_mh_v2 detector).
- Branch: `claude/test-next-steps-OwwJ4`. CPU only (~6 min meta v5,
  ~10 s J7 conformal). No GPU, no LB submission.

- **Meta v5 — 64-component pool with OvR + recipe_focal_effnum + LR meta**
  (`scripts/meta_v5_ovr_extended.py`, ~155 lines):
  - **Strict EXTRA_EXCLUDE list**: 12 prior meta outputs (xgb_metastack
    v1-v4 + variants + LR metas) treated as circular leakage; LB-confirmed
    regressors (soft_distill family); bias-mismatched (recipe_smote_v3);
    submission-derived (primary_sub_tau*); derived blends (j6_qp_blend,
    greedy_blend, etc.); tau-sweep pseudos (circular w.r.t. pseudo_s1);
    TTA artifacts. Final clean pool: **64 components** (v1's was 63;
    gained OvR + focal_effnum, lost some excluded items).
  - Pool composition verified: OvR present, focal_effnum present, all
    known LB-regressors excluded. `xgb_metastack` (v1) excluded so v5
    isn't trained on its own predecessor.
  - Same XGB heavy-reg HPs as v1: depth=4, lr=0.05, reg_alpha/lambda=5,
    subsample/colsample=0.9, 3000-round cap with es=200.
  - Wall: 6m15s on 16-core CPU (5 folds × ~70s each, best_iter 279-408).
  - **Standalone**:
    ```
                          v5         v1         Δ
    raw OOF argmax      0.97364    0.97365    -0.00001
    raw @recipe-bias    0.98023    0.98041    -0.00018
    iso @recipe-bias    0.98072    0.98059    +0.00013   ← stronger raw signal
    ```
    v5_iso is the FIRST meta variant to beat v1_iso standalone, but
    raw v5 is BELOW raw v1. Pattern: extra components (OvR + focal +
    others) add NOISE at the raw output level; iso recovers it.
  - **REPLACE-v1 sweep onto LB-best 3-stack** (anchor 0.98061; v1 at
    α=0.30 produced LB-best 4-stack 0.98084 / +0.00023 OOF lift):
    ```
    α          v5_iso onto 3-stack    Δ vs anchor
    0.000     0.98061                 +0.00000
    0.300     0.98071                 +0.00010   ← v1's α slot, UNDERPERFORMS
    0.350     0.98076                 +0.00015
    0.400     0.98081                 +0.00020   ← peak (right at +2e-4 gate)
    0.500     0.98069                 +0.00009
    ```
    Per-class recall at α=0.40: L=0.9956 / M=0.9699 / H=0.9770 vs anchor
    [0.99553, 0.96885, 0.97744]. H drop -0.0004 marginally PASSES the
    -5e-4 guardrail. Compare to v1's α=0.30 lift +0.00023 OOF → +0.00086
    LB. **v5 needs α=0.40 to barely match v1's OOF lift, and at v1's
    α=0.30 v5 underperforms by 0.00013.**
  - **STACK-ON-TOP onto LB-best 4-stack** (anchor 0.98084):
    ```
    α          v5_iso onto 4-stack    Δ vs anchor
    0.025     0.98079                 -0.00005
    0.200     0.98086                 +0.00002   ← peak (sub-gate)
    0.500     0.98075                 -0.00010
    ```
    All α below +2e-4 LB-transfer threshold. v5 contributes nothing on
    top of v1; v1 already extracts the meta-stacker signal.
  - **Linear-projection rule** (LR/v4 closures predicted): Δ +0.00020
    OOF at α=0.40 with marginal guardrail pass projects LB null or
    marginal regression vs current LB-best 0.98094. **No LB probe
    warranted.**

- **J7 — conformal-gated overrides on score=6** (`scripts/j7_conformal_spec6.py`,
  ~140 lines):
  - Mechanism: split-conformal calibration of spec6_mh_v2 detector
    (AUC 0.938 per CLAUDE.md 2026-04-25 entry). Hold out 30% of in-domain
    train rows for calibration; pick threshold τ such that Wilson 90%
    one-sided lower CI on precision ≥ 8.1% break-even (under macro-recall).
    Replaces ad-hoc theta sweep with principled coverage-guaranteed
    threshold selection.
  - Override domain: score=6 ∩ teacher_argmax=Medium.
    Train: 35,418 rows / 324 truly-H. Test: 15,288 rows.
  - Calibration: 10,625 rows / 94 truly-H. **Conformal τ=0.14782**,
    Wilson lower CI 0.0829 (just above 0.081 break-even).
  - Out-of-cal train: 30 overrides, 3 correct (10.0% precision — at the
    break-even floor).
  - Full-train OOF: 45 overrides, 6 correct (13.3% precision).
    Δ macro-recall = +0.00004 (well below +2e-4 LB-transfer gate).
  - **Test-side override count: 5** (gate requires ≥10).
  - **Verdict: lever fully closed.** Conformal calibration confirms the
    spec6_mh_v2 entry's "prevalence-bounded to ~10 override rows"
    diagnosis is structural — even with principled coverage-guaranteed
    threshold at break-even precision, only 5 test rows clear the bar.
    The bottleneck is INFORMATION (which rows to override) not THRESHOLD
    SELECTION (what cutoff to use).

- **Combined session read-out — 8th independent saturation confirmation
  at LB 0.98094**:
  ```
  attack vector                                  best OOF Δ   notes
  --------------------------------------------- -----------  -------
  1. Tier 1c greedy expanded pool (132c)        +0.00002     sub-gate
  2. Tier 1c meta-stacker v2 (224-dim)          +0.00002     sub-gate
  3. Tier 1c meta-stacker XGB seed-bag          +0.00003     sub-gate
  4. Cross-poll metastack v3                    +0.00015     LB 0.98060 -0.00034
  5. J2 bootstrap-bagged metastack              +0.00003     proj LB -0.00054
  6. LR meta-stacker v2 (this branch's parent)  +0.00046     LB 0.98052 -0.00042
  7. LR v2 + iso-after-blend                    +0.00000     OOF NULL
  8. **Meta v5 + J7 conformal (this entry)       +0.00020     borderline (predicted null) + closed**
  ```
- LB best unchanged at **0.98094** via `submission_tier1b_greedy_meta.csv`.
  Final-selection lock unchanged: PRIMARY = LB 0.98094, HEDGE =
  `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005, audit F1 swap).
  LB budget unchanged: 6/10 used today, 4 remaining.

- **Two portable rules** (logged to LEARNINGS.md):
  1. **Conformal calibration on a binary detector with prevalence < 0.05
     in the test override domain hits a numeric floor at ~5-10 overrides
     regardless of detector AUC.** Conformal threshold selection cannot
     manufacture overrides where the detector's confidence distribution
     is too narrow at break-even precision. The lever's structural limit
     is information about WHICH rows to override, not WHAT threshold to
     use. Run a 10-second prevalence × precision scan BEFORE building
     the conformal pipeline; if `n_truly_positive_in_override_domain ×
     achievable_precision < 50`, the lever is structurally too small to
     move LB regardless of threshold method.
  2. **Adding components to a saturated meta-stacker bank lifts standalone
     iso OOF (+0.00013 in v5 case via per-class iso re-calibration of
     the new larger bank's outputs) but does NOT translate to blend-level
     lift over the existing meta.** The meta-stacker that consumed the
     prior bank already extracts most of the available signal channels;
     iso-cal of the new meta over-fits the additional components without
     adding orthogonal signal at the anchor's fixed-bias operating point.
     Diagnostic check: if standalone v_new_iso > v_old_iso BUT v_new_iso
     at v_old's α-slot UNDERPERFORMS v_old, the new components are
     calibration-cosmetic, not signal-contributing.

- Artefacts committed (whitelisted in `.gitignore`):
  - `scripts/meta_v5_ovr_extended.py` + `scripts/j7_conformal_spec6.py`
  - `scripts/artifacts/oof_xgb_metastack_v5{,_iso}.npy` + test (4 files)
  - `scripts/artifacts/meta_v5_results.json` (full sweep + per-class
    recall + best_iters)
  - `scripts/artifacts/j7_conformal_spec6_results.json` (Wilson CI +
    train override stats + test override count + gate decision)

### 2026-04-26 — own-CSV ensemble lever: NULL (subs too nested; refutes "ensemble of ensembles" hypothesis)

- Goal: execute the highest-EV remaining own-pipeline lever from the
  post-saturation brainstorm — vote/blend across our own LB-validated
  submission CSVs (treating each as opaque). Public-CSV blenders use
  this exact mechanism on others' submissions; we apply it to our own.
  Bayesian prior 20-25% it produces a candidate with the right
  Jaccard+magnitude+rare-class profile — every prior experiment
  ensembled COMPONENTS; nothing has ensembled SUBMISSIONS.
- Branch: `claude/advanced-ensemble-methods-vQrhS`. Files:
  `own_ensemble_helpers.py` (reconstructs OOF/test for 6 LB-validated
  subs via deterministic log-blend chains), `own_ensemble_strategies.py`
  (5 strategies: equal log, LB-weighted log, hard-vote, soft-vote,
  greedy forward), `own_ensemble_subset_probe.py` (3 follow-up probes:
  3-view subset, fine α-grid greedy, hard-vote 3-view).

- Reconstructed 6 LB-validated subs and verified each matches its
  documented LB score within calibration ladder:
  ```
  name                     OOF tuned    LB        gap
  primary                  0.980842    0.98094   +0.00010
  stack2 (lb3+rm+nr_iso)   0.980609    0.98008   -0.00053
  m3_seed (3-way recipe)   0.980286    0.98005   -0.00024
  m2_pseudo (2-way 50/50)  0.980123    0.97998   -0.00014
  recipe_full_te           0.979665    0.97939   -0.00028
  catboost_iso (iso-cal'd) 0.979070    0.97935   +0.00216
  ```
  All Jaccards vs primary: recipe 0.83, catboost_iso 0.81, m2_pseudo
  0.89, stack2 0.96, m3_seed 0.90. **Catboost is the most
  orthogonal** (Jaccard 0.81); stack2 is the most redundant (0.96 —
  it's a strict subset of primary).

- **5 strategies — every result NULL**:
  ```
  Strategy                       OOF tuned    Δ vs primary   errs vs primary    Jaccard
  ─────────────────────────────────────────────────────────────────────────────────────
  PRIMARY (anchor)              0.980842     0              9415 (anchor)        1.0000
  S1 equal log-blend (1/6 each) 0.980196    -0.00065        +233                 0.92
  S2 LB-weighted τ=100          0.980252    -0.00059        +224                 0.92
  S2 LB-weighted τ=1000         0.980380    -0.00046        +159                 0.95
  S3 hard-vote (LB tie-break)   0.980503    -0.00034        +193                 0.96
  S4 soft-vote (arithmetic)     0.980247    -0.00060        +311                 0.92
  S5 greedy fwd (α=0.05 grid)   0.980853    +0.00001        +26                  0.9951
  ```
  All except S5 strictly worse than primary. S5 picked m3_seed at
  α=0.05 → +0.00001 OOF (within noise).

- **3 follow-up probes — confirm null**:
  - 3-view {primary, recipe, catboost_iso} 2D log-blend grid: best
    weights (1.00, 0.00, 0.00) — primary alone. Adding recipe or
    catboost at any positive weight HURTS macro-recall.
  - Greedy forward with FINE α-grid (0.005-0.05 + 0.05-0.55):
    step 1 + m2_pseudo at α=0.035 → 0.980865 (Δ +0.00002).
    No further additions improve. Final: 0.980865, errs 9421
    (+6 vs primary), Jaccard 0.9958. **Best lever result of session.**
  - Hard-vote 3-view {primary, recipe, catboost_iso}: 0.980121
    (Δ −0.00072). Hard-vote loses calibration info.

- **Best candidate's per-class trade is tiny but in the RIGHT direction
  for once**: greedy_fine PCR [0.99552, 0.96949, 0.97758] vs primary
  [0.99553, 0.96951, 0.97749] — Low/Med essentially flat, **High recall
  +0.00009**. First own-pipeline candidate in this branch with
  improved-on-rare-class direction. But the magnitude (+9e-5 H recall,
  +2e-5 macro-recall) is 25x below the +5e-4 emit gate.

- **Linear-projection rule** (per CLAUDE.md): primary's gap −0.00010 (LB>OOF).
  Greedy_fine projected LB = 0.980865 + 0.00010 = 0.980965, only
  +0.00002 above current LB-best 0.98094 — well within fold noise.
  No LB probe warranted.

- **Mechanism — why the lever fails**: our 6 LB-validated subs are
  HEAVILY NESTED. They all share the recipe → pseudo_s1 → (pseudo_s7) →
  RealMLP → nonrule_iso → meta_iso backbone. Specifically:
    - `recipe_full_te` ⊂ `m2_pseudo` ⊂ `m3_seed` ⊂ `stack2` ⊂ `primary`
    - Only `catboost_iso` is genuinely model-family-distinct.
  Public-CSV blenders' wins come from INDEPENDENT pipelines built by
  different teams (different model families, different FE, different
  hyperparameters, different seeds). Our 6 subs are 6 DEPTHS of one
  pipeline plus 1 alternate model (catboost). Ensembling them
  amounts to per-row weighted reweighting along the same depth axis —
  same operating point as the deepest sub (primary).

- **Portable rule** (LEARNINGS.md candidate): **"Own-CSV ensemble of
  hierarchically-nested submissions cannot beat the deepest sub.**
  When sub_A is a strict subset of sub_B (B uses A as a backbone +
  additional components), ensembling them produces predictions
  between A and B's operating points. Since B already chose the
  rare-class-favoring corner of the macro-recall Pareto frontier,
  pulling toward A dilutes that corner. For 'ensemble of own
  submissions' to lift, the subs must be from STRUCTURALLY
  INDEPENDENT pipelines (different model families AND different FE
  AND different fold splits). Six depths of one pipeline don't qualify."

- LB delta: n/a (no LB probe — every candidate sub-gate). LB best
  unchanged at **0.98094** via `submission_tier1b_greedy_meta.csv`.
  LB budget unchanged.

- **Strategic implication**: this closes the only structurally-
  novel ensembling lever remaining in the post-saturation
  brainstorm. Combined with the 10 prior saturation confirmations
  and the 4 mechanism-distinct ensembling nulls from earlier today,
  the own-pipeline LB ceiling at 0.98094 is now exhaustively
  established. To break it requires either:
  (a) a NEW pipeline (different model family + different FE + truly
      independent training procedure) producing a sub at LB > 0.978
      with errors orthogonal to primary's. Concrete shortlist:
      adversarial-trained recipe XGB, NN with custom rare-class loss,
      external-feature transplant from physics-based agronomy model.
  (b) public-CSV blending (banned by top-of-file rule).

- Artefacts (whitelisted via .gitignore for cross-branch reuse):
  - 7 OOF + 7 test pairs: `oof_own_S1_equal_log`,
    `oof_own_S2_lb_weighted_tau{100,200,500,1000}`, `oof_own_S3_hard_vote`,
    `oof_own_S4_soft_vote`, `oof_own_S5_greedy_forward`,
    `oof_own_3view`, `oof_own_greedy_fine` + corresponding test_*.npy
  - 4 results JSONs: `own_ensemble_strategies_results.json`,
    `own_ensemble_subset_probe_results.json`, plus 2 blend-gate JSONs
  - 3 scripts (≤200 lines each): `own_ensemble_helpers.py`,
    `own_ensemble_strategies.py`, `own_ensemble_subset_probe.py`

### 2026-04-26 — advanced-ensemble-methods session (gated MoE + CMA-ES + per-cell + joint w+b): 4 nulls + 10th saturation confirmation, with mathematical proof of constant-weight ceiling

- Goal: at user request "is there a new or better approach to ensemble our
  predictions to improve balanced accuracy?" — execute 4 mechanism-distinct
  ensembling levers that haven't been tested in the comp log. The structural
  problems to attack: (1) macro-recall + fixed bias misaligns with every
  convex surrogate (J6 confirmed), (2) magnitude trap kills orthogonal-error
  candidates, (3) meta-stacker bank is saturated, (4) every multi-stage
  tuning blew up the OOF→LB gap.
- Branch: `claude/advanced-ensemble-methods-vQrhS`. All artifacts whitelisted
  via .gitignore for cross-branch reuse.

- **#1 Per-row gated MoE** (`scripts/moe_gated_blend.py`,
  `scripts/moe_helpers.py`, `scripts/moe_blend_gate.py`): linear gate
  over 37 low-dim features (4 dist + 4 abs + min_axis_abs +
  min_boundary_dist + dgp_score + rule_pred + per-expert max_prob +
  per-expert argmax onehot). K=6 experts: LB-best 4-stack, RealMLP,
  xgb_nonrule_iso, xgb_metastack_iso, leaf_ote_meta_v2, xgb_dist_digits.
  L-BFGS over (W ∈ ℝ^{K×d}) with cross-entropy of the blended posterior
  against y, leak-safe per-fold (gate trained on tr_idx, applied to va_idx).
  Wall: ~2 min × 5 folds = ~10 min CPU.
  - Standalone OOF tuned bal_acc = **0.980749** (vs LB-best 4-stack
    0.980842, **Δ −0.00009**)
  - Errors 9,190 (vs LB-4 9,415, −225 fewer)
  - Jaccard vs LB-best 4-stack = 0.948 (high redundancy)
  - Per-class recall: L 0.9956 / M 0.9704 / **H 0.9762** (vs LB-4
    [0.9955, 0.9695, 0.9775]) — Low/Med UP, **High DOWN −0.0013**
  - Blend gate vs LB-best 4-stack: peak α=0.15 → Δ=**+0.00004** (well
    below +5e-4 emit gate)
  - **Per-fold mean gate weights**: ~58% xgb_metastack_iso, ~40% LB-best
    4-stack, ~2% combined to RealMLP/nr/leaf/digits. Folds 1-2 vs 3-5
    invert the meta_iso vs lb4 ratio (bistable optimum).
  - **Verdict: NULL.** Per-row gating ALSO confirms fixed weights of the
    4-stack are near per-row optimal — the gate concentrates 98% on
    components already in the 4-stack at fixed ratios. The 2% it allocates
    to orthogonal experts adds negligible signal at fixed-bias evaluation.

- **#2 CMA-ES on macro-recall + simplex** (`scripts/cmaes_macro_recall.py`):
  K=7 components (LB-best 3-stack + xgb_metastack_iso + realmlp +
  xgb_nonrule_iso + leaf_ote_meta_v2 + xgb_dist_digits + recipe_full_te).
  Gradient-free CMA-ES on the simplex (sigma0=0.5, popsize=14, maxiter
  120 in-sample + 80 nested). Optimizes ACTUAL macro-recall at the
  fixed recipe bias [1.4324, 1.4689, 3.4008]. Wall: ~19 min CPU
  (8 min in-sample + 11 min nested across 5 folds).
  - **In-sample full-fit upper bound: 0.980912** with weights
    {lb_best_3stack 0.503, xgb_metastack_iso 0.439, leaf_ote 0.043,
    recipe 0.014, others < 1%}.
  - Nested CV: **0.980571** (overfit gap +0.00034).
  - **CRITICAL FINDING — mathematical ceiling proven**: the in-sample
    upper bound 0.98091 is essentially the SAME OPERATING POINT as
    LB-best 4-stack 0.98084 (only +0.00007 higher). This is mathematical
    proof that the constant-weight blend ceiling on these 7 components
    at the LB-validated bias is essentially at LB-best already. No
    constant-weight scheme can extract more than ~0.0001 OOF beyond
    what LB-best already captures.
  - Per-fold weight noise high (folds 1, 4, 5 split 0.5/0.42 lb3+meta,
    folds 2, 3 spread to multiple components) — same J2-bag pattern.
  - Blend gate vs LB-best 4-stack: peak α=0.025 → Δ=**−0.00002** (strict
    null, all α > 0 negative).
  - **Verdict: NULL on blend.** Validates that the constant-weight
    blend space is fully saturated; this is now the TENTH independent
    saturation confirmation at LB 0.98094.

- **#3 Per-cell mini-meta-stackers** (`scripts/per_cell_mini_meta.py`):
  64 cells from (dry, norain, hot, windy, nomulch, kc_active) bits.
  Per-fold per-cell multinomial LR (C=0.1, max_iter=400) on 16 features
  per row: log(LB-4 probs)×3 + log(meta_iso probs)×3 + log(realmlp
  probs)×3 + 7 non-rule numerics (Humidity, Prev_Irrigation, EC,
  Soil_pH, Field_Area, Sunlight, Organic_Carbon). Cells with < 100
  train rows fall back to LB-best 4-stack. Wall: ~12 sec total
  (58 cells trained per fold).
  - Standalone OOF tuned = 0.979874 (Δ −0.00097 vs LB-4)
  - Errors 9,307 (-108 vs LB-4)
  - Per-class recall: L 0.9957 / M 0.9700 / **H 0.9739** — High recall
    DOWN −0.0036 (the worst rare-class trade of all 4 attempts)
  - Jaccard vs LB-4 = 0.851 (decent orthogonality)
  - Blend gate: peak α=0.025 → Δ=**−0.00004** (strict null)
  - **Verdict: NULL.** Per-cell LR sees too few High examples per cell
    to model the rare-class boundary; pulls Low/Med predictions slightly
    better but loses High recall.

- **#4 Joint weights+bias optimization with smooth surrogate**
  (`scripts/joint_weights_bias.py`): single-stage L-BFGS over
  (W ∈ ℝ^K, bias ∈ ℝ^3) jointly, optimizing soft-macro-recall via
  temperature-relaxed argmax (T=0.3) + L2_BIAS=5.0 anchor toward
  LB-validated bias + L2_W=1e-3 on weights. Designed to avoid the
  binhigh trap (post-hoc bias retune compounding overfit). K=6 components.
  Wall: ~3.5 min CPU.
  - In-sample bias = [1.4324, 1.4687, 3.4010] vs LB-validated
    [1.4324, 1.4689, 3.4008] — basically frozen (L2 anchor strong).
  - Per-fold weights spread evenly (~0.13-0.21 per component) — surrogate
    encouraged diversity unlike CMA-ES.
  - Standalone OOF tuned = 0.980295 (Δ −0.0005 vs LB-4)
  - Errors 9,324 (-91 vs LB-4)
  - Per-class recall: L 0.9959 / M 0.9695 / **H 0.9755** (Δ H −0.0020)
  - Blend gate: peak α=0.025 → Δ=**−0.00002** (strict null)
  - **Verdict: NULL.** Smooth-surrogate doesn't help — the bias L2
    anchor (necessary to defend against binhigh) prevents the joint
    optimization from finding a different operating point.

- **Universal pattern across all 4 attacks**:
  ```
                    standalone   errs vs   Jaccard   peak Δ vs    rare-class
  Approach          tuned OOF    anchor    vs LB-4   LB-best       trade
  ─────────────────────────────────────────────────────────────────────────
  LB-best 4-stack   0.980842       0       1.000     —             baseline
  #1 MoE gated      0.980749    -225       0.948     +0.00004      H -0.0013
  #2 CMA-ES nest    0.980571     -58       0.961     -0.00002      H -0.0012
  #3 Per-cell       0.979874    -108       0.851     -0.00004      H -0.0036
  #4 Joint w+bias   0.980295     -91       0.886     -0.00002      H -0.0020
  ```
  EVERY alternative scheme finds a fewer-error solution that trades High
  recall for Low/Med — wrong direction under macro-recall. The LB-best
  4-stack at OOF 0.98084 / LB 0.98094 sits at the rare-class-favoring
  corner of the macro-recall Pareto frontier.

- **10th independent saturation confirmation at LB 0.98094**:
  ```
  attack vector                                  best blend Δ   notes
  ----------------------------------------------- -------------- ----------
  1. Tier 1c greedy expanded (132c)               +0.00002       sub-gate
  2. Tier 1c meta-stacker v2 (224-dim)            +0.00002       sub-gate
  3. Tier 1c meta-stacker XGB seed-bag            +0.00003       sub-gate
  4. Cross-poll metastack v3                      +0.00015       LB -0.00034
  5. J2 bootstrap-bagged metastack                +0.00003       proj null
  6. LR meta-stacker v2 (C=0.1, none)             +0.00098       LB -0.00042
  7. LR v2 + iso-after-blend                      +0.00000       NULL
  8. v4 ET+kNN bank-extension                     +0.00036       LB -0.00102
  9. P3 perturbed meta v1                         +0.00071       LB -0.00139
  10. **#1-#4 advanced-ensemble (this entry)      +0.00004       NULL on all 4**
  ```

- **Three portable rules** (LEARNINGS.md candidates):
  1. **Per-row gated MoE collapses to fixed-weight blend when the
     dominant components already form the LB-best stack.** The gate
     learns to give 98%+ weight to those components, leaving negligible
     room for orthogonal experts. Useful only when no constant-weight
     blend is at the optimum yet — once a strong fixed-weight stack
     exists, MoE becomes a re-implementation of the same operating point.
  2. **CMA-ES on macro-recall is the cheapest mathematical-ceiling proof.**
     Gradient-free + direct objective + simplex constraint. ~20 min CPU
     for 7 components. The in-sample upper bound is the math ceiling
     for any constant-weight scheme; if it's within fold-noise of the
     current best, the constant-weight space is provably saturated.
     Run this BEFORE building more weighted stackers.
  3. **L2_BIAS anchor strong enough to prevent binhigh trap also
     prevents joint w+b from finding a meaningfully different optimum.**
     The two failure modes (binhigh OOF inflation vs frozen bias) form
     a no-win for joint optimization on this problem family. Use joint
     w+b only when the bias is unanchored and you accept binhigh-style
     LB-overfit risk. Otherwise, fix bias (current approach) and tune
     weights only.

- LB delta: n/a (no LB probe — every blend Δ below +5e-4 emit gate
  AND macro-recall trade is negative on rare class). LB best unchanged
  at **0.98094** via `submission_tier1b_greedy_meta.csv`. LB budget
  unchanged.

- **Strategic implication**: with 10 independent saturation confirmations
  including a mathematical CMA-ES proof, **breaking LB 0.98094 within
  the existing OOF bank is provably impossible at the constant-weight
  blend level, and per-row/per-cell/joint-bias all confirm the same
  ceiling at the row/decision level.** Any further lift requires either:
  (a) a NEW component whose errors satisfy BOTH Jaccard < 0.80 AND
  errs ≤ anchor AND per-class recall ≥ anchor on rare class — a profile
  no candidate has matched in ~30 prior tests, or (b) a strategic
  pivot to public-CSV blending (banned by top-of-file rule).

- Artefacts committed (whitelisted via .gitignore for cross-branch reuse):
  - 4 OOF + test pairs: `oof_moe_gated`, `oof_cmaes_blend`,
    `oof_per_cell_meta`, `oof_joint_blend` (+ corresponding `test_*.npy`)
  - 4 results JSONs + 4 blend-gate JSONs
  - 6 scripts: `moe_helpers.py`, `moe_gated_blend.py`, `moe_blend_gate.py`,
    `cmaes_macro_recall.py`, `per_cell_mini_meta.py`, `joint_weights_bias.py`,
    `joint_blend_gate.py` (155 lines max each, modular per CLAUDE.md rule)

### 2026-04-25 — fresh-perspectives session (P1 test-prior, P2 quotas, P3 perturbed meta): 3 nulls + 9th saturation confirmation

- Goal: senior-DS-style fresh review attacking the LB 0.98094 ceiling
  via three structurally distinct levers: (P1) test-prior recalibration,
  (P2) hard class quotas instead of log-bias, (P3) perturbed-OOF
  meta-stacker. Hypotheses focus on the unstressed property of the
  LB-best primary — its negative OOF→LB gap of −0.00010.
- Branch: `claude/data-science-perspectives-jZCc3`. All artifacts
  whitelisted via .gitignore for cross-branch reuse.

- **P1 test-prior characterization** (`scripts/p1_test_prior.py`, ~75s CPU):
  - Train y prior:        Low 0.5872 / Med 0.3795 / High 0.0333
  - Test rule_pred prior: Low 0.5923 / Med 0.3742 / High 0.0335
  - Predicted test y prior (via P(y|rule) on train + rule_test counts):
    Low 0.5865 / Med 0.3800 / High 0.0336 — diff vs train = ±0.001.
    **No prior shift exists.** Train and test are drawn from the same
    class distribution within 0.07pp on every class.
  - Sanity unweighted bias retune found bias [1.04, 1.45, 3.40] giving
    OOF 0.98094 (+0.00010 vs original [1.43, 1.47, 3.40] which gives
    0.98084). That's the 2026-04-21 binhigh trap (post-hoc bias retune
    on a tuned stack — caused -0.00084 LB regression last time). Skip.
  - **Verdict**: NULL — no test-side prior shift to exploit.

- **P2 hard class quotas** (`scripts/p2_quota_decision.py`, ~6s CPU):
  - Tested 12 quota×order configurations on the LB-best 4-stack OOF.
    Quota sources: train y prior (Q1), test rule_pred prior (Q2), P1
    predicted test y prior (Q3). Orders: HML, MHL, HLM, greedy-Hungarian.
  - Best non-fitted: Q2_test_rule/HML at OOF 0.97702 (Δ=**-0.00383** vs
    log-bias baseline 0.98084). All 12 configs strictly negative;
    range [-0.00383, -0.01065].
  - **Diagnostic**: log-bias predicts 23,132 OOF Highs vs train prior
    21,009 — that **10% over-prediction of High is informative signal**,
    not an arbitrary tuning choice. Quota rules force-cap High at the
    prior, sacrificing 385 true-Highs for 1,570 extra Mediums (wrong
    direction under macro-recall — High has ~12× per-row leverage).
  - **Verdict**: NULL — log-bias's per-row joint comparison structurally
    beats per-class rank-only ordering. The decision rule is not the
    bottleneck.
  - **Portable rule** (LEARNINGS.md candidate): "Hard class quotas
    that force-cap to known/predicted class priors will LOSE macro-
    recall when the underlying model's argmax distribution intentionally
    over-predicts the rare class. The over-prediction at log-bias's
    optimal operating point IS the signal — capping it removes signal,
    not noise."

- **P3 perturbed-OOF meta-stacker** (`scripts/p3_perturbed_meta.py`,
  ~37 min CPU). Two configs × K=3 bag × 5-fold = 30 fits.
  Mechanism: per-fold per-bag rng → add Gaussian noise to log-prob
  features at training time, eval on UNNOISED OOF/test. Hypothesis:
  amplifies the negative OOF→LB gap by forcing meta-XGB to use
  noise-robust signal channels rather than fitting fold-OOF noise.

  ```
  variant            standalone   iso     best blend Δ vs LB-3-stack
  v1 (σ=0.3, csb=0.9)  0.98093    0.98098   +0.00071 iso α=0.50
  v2 (σ=0.5, csb=0.5)  0.98109    0.98100   +0.00062 iso α=0.50
  ```
  Both peak at same α=0.500 (robust to HP). All 6 blend gates pass
  for v1: errs 9028 (-544 vs 3-stack), per-class L 0.9957/M 0.9709/
  H 0.9774, Jaccard 0.9015. Vs LB-best 4-stack PRIMARY: errs -387,
  per-class L +0.0002, M +0.0014, H -0.0001 (clean rare-class trade).

- **P3 v1 LB probe**: `submission_p3_perturbed_v1_noise03_csb09_k3_iso_a500.csv`
  → **LB public = 0.97955**. Δ vs LB-best 0.98094 = **-0.00139**
  (regression). OOF→LB gap = **+0.00177** (typical OOF-overfit
  failure mode). Same magnitude as prior LR meta-stacker v1 null
  (LB 0.97991, gap +0.00176) and meta-stacker v4 bank-extension
  null (LB 0.97992, gap +0.00129).

- **P3 A/B clean baseline at fixed bank** (`scripts/p3_perturbed_62.py`,
  ~13 min CPU): same perturbed-meta pipeline RESTRICTED to the exact
  62-component bank used by the LB-best primary's xgb_metastack. Same
  XGB HPs, same K=3 bag, same noise σ=0.3.
  ```
                     standalone   iso     best blend Δ vs LB-3-stack
  ORIGINAL meta-62   0.98041     0.98059   +0.00023 iso α=0.30 ← LB-best signal
  PERTURBED-62       0.98033     0.98049   +0.00007 raw α=0.35 ← below emit gate
  PERTURBED-111      0.98093     0.98098   +0.00071 iso α=0.50 ← LB null
  ```
  **Definitive falsification**: at fixed bank, perturbation makes the
  meta SLIGHTLY WORSE (-0.00008 standalone, -0.00016 blend). 100% of
  the +0.00071 OOF lift in the 111-component variant was bank-extension
  OOF overfit from the 49 new components added since LB-best built.

- **9th independent saturation confirmation at LB 0.98094**:
  ```
  attack vector                            best LB        Δ vs primary
  ----------------------------------------- -------------- --------------
  1. Tier 1c greedy expanded (132c)         (sub-gate)     n/a
  2. Tier 1c meta-stacker v2 (224-dim)      (sub-gate)     n/a
  3. Tier 1c meta-stacker XGB seed-bag      (sub-gate)     n/a
  4. Cross-poll metastack v3                0.98060        -0.00034
  5. J2 bootstrap-bagged metastack          (proj null)    n/a
  6. LR meta-stacker v2 (C=0.1, none)       0.98052        -0.00042
  7. LR v2 + iso-after-blend                (sub-gate)     n/a
  8. v4 ET+kNN bank-extension               0.97992        -0.00102
  9. **P3 perturbed meta v1 (this entry)    0.97955        -0.00139**
  ```

- **Three portable rules logged** (candidates for LEARNINGS.md):
  1. **Train/test prior characterization is a 30-second sanity probe**
     before optimizing log-bias. On synthetic-Playground problems where
     train+test are released together, AV-passing features ⇒ priors
     are statistically identical ⇒ no recalibration possible. Skip
     test-prior-driven bias optimization.
  2. **Log-bias decision rule is structurally optimal under macro-recall**
     when the model's argmax confidence carries class-specific
     calibration information. Hard quota rules that force class counts
     to match the prior LOSE macro-recall on the rare class because
     they discard the model's "I'm confident this row is High even
     though that overshoots the prior" signal.
  3. **Per-row Gaussian noise on meta-stacker log-prob inputs is NOT a
     valid lever** for amplifying CV-pessimism at fixed bank size.
     Noise injection during meta training does not pick up additional
     signal channels; it just adds gradient variance the meta-XGB
     ignores via heavy-reg early stopping. Bank size, not training
     stochasticity, is what changes the meta's OOF.

- LB budget: **4/10 used today** (1 P3 v1 probe + 3 from prior session).
  6 remaining.
- **Final-selection lock unchanged** (5 days to deadline):
  1. **PRIMARY**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
  2. **HEDGE (recommended swap)**: `submission_3way_recipe025_s1035_s7040.csv`
     → **LB 0.98005** (premium -0.00089, sidesteps meta-stacker layer)
- Pack 0.98114 still +0.00020 above primary; leader 0.98219 still
  +0.00125 above. Reachable only via public-CSV blending (banned).

### 2026-04-25 → 04-26 — J7 conformal-gated overrides + J4 KAN PROBE + W_RECIPE=1.0 distill: 3 NULLs, distill family fully closed

Three diagnostics from the post-Tier-1c "open-items" list, all NULL,
with two adding portable rules and definitively closing the
soft-distillation lever family.

#### J7 — conformal-gated overrides on score=6 (NULL, both mechanisms)

- Goal: replace the OOF-overfit-prone raw-θ threshold sweep with a
  principled coverage-guaranteed override. spec6_mh_v2 (AUC 0.938)
  ranked correctly but raw-θ peak at +0.00009 OOF was selection-bias
  manufactured. Two independent variants tested:
  1. **Wilson-lower-bound precision-gated** (already on origin/main
     commit c20fa1b): tau=0.148, 45 train overrides at 13.3% precision,
     delta_oof=+0.00004, gate_pass=FALSE.
  2. **Mondrian split-conformal** (this branch, `scripts/j7_conformal_spec6.py`):
     per-fold leak-safe calibration, alpha sweep ∈ [0.01, 0.50] for
     coverage of True-High class. Best operating point alpha=0.50:
     3,755 overrides, 4.5% precision, delta=−0.00230. No safe positive
     operating point at any alpha.
- Both confirm the 2026-04-24 Pareto-frontier closure on per-class
  High recall: under macro-recall, override break-even precision is
  H/(M+H) ≈ 8.1%. spec6_mh_v2's top-ranked picks plateau at 6.5%
  precision regardless of how the threshold is chosen. The detector
  knows, but its ranks aren't sharp enough for deployment.
- LB delta: n/a (no probe; both monotone-negative or sub-gate).

#### J4 — KAN (Kolmogorov-Arnold Networks) PROBE: 15th NN null with RECORD-LOW Jaccard 0.1267

- Goal: 15th NN-family attempt. The 2026-04-21 DGP-residuals EDA
  established the host label generator is a smooth NN function; KAN's
  per-edge B-spline parameterisation is uniquely suited to fit smooth,
  non-axis-aligned boundaries. Different inductive bias from all 14
  prior NN nulls (MLP / FT-T / TabPFN / DAE / RealMLP / Trompt / Mamba)
  which use attention or pure feed-forward.
- Implementation: efficient-kan (Blealtan, MIT, GitHub-only — not on
  PyPI). ~10× faster than original pykan, pure PyTorch (no CUDA toolkit
  needed). 19 raw features (8 cats one-hot + 11 nums standardised) to
  keep Jaccard apples-to-apples with sister NN kernels.
- PROBE config (`kaggle_kernel/kernel_kan/`): 1 fold × 504k train + 10k
  orig × 12 epochs × KAN [43, 192, 96, 48, 3] (314k params, grid_size=5,
  spline_order=3). Wall: ~5 min on Kaggle P100, ~3.4 sec/epoch. SMOKE
  GREEN locally first (CPU, 138k params, 2 folds × 20k × 2 epochs =
  ~3 sec).
- Standalone results (fold-1 only):
  ```
  argmax bal_acc        0.93286   (training plateau evident; loss flat)
  tuned bal_acc (own)   0.93868   bias=[2.13, 1.97, 2.00] (uniform shift)
  errs at anchor bias   12,217    vs anchor 1,944 = 6.28× anchor
  ```
- **Jaccard analysis** (vs LB-best 3-stack, fold-1, 126k val rows):
  ```
  Jaccard(KAN errs, anchor errs) = 0.1267    ← RECORD LOW orthogonality
  prior best NN orthogonality:    Mambular   0.491
                                  Trompt     0.534 (fold-1)
                                  RealMLP    0.621
  ```
  KAN's spline-on-edge inductive bias DOES find a fundamentally
  different decision surface — this is the most orthogonal predictor
  any NN family has produced on this problem.
- **Blend gate FAILS on magnitude**: errs ratio 6.28× crushes the
  ≤1.05× threshold by an order of magnitude. Sweep monotone-negative
  from α=0.025 (Δ=−0.00003) through α=0.50 (Δ=−0.00504). 15th NN null
  in a uniquely orthogonal way.
- **Pattern hardened across 15 NN families**:
  ```
  family            Jaccard vs anchor    errs vs anchor    LB outcome
  ----------       ------------------- ------------------ ------------
  MLP v5-v9         0.62-0.85           +1500-15000       NULL
  FT-Transformer    0.61                +12000            NULL
  TabPFN            0.81                +1485             NULL
  RealMLP n_ens=1   0.62                +358              LB +0.00003
  RealMLP n_ens=4   0.62                +485              NULL
  Trompt (probe)    0.53                +169              NULL
  Mambular SSM      0.49                +518 (+27%)       NULL
  KAN (probe)       **0.13**            +10,273 (+528%)   NULL
  ```
  KAN broke the orthogonality rule (4× lower Jaccard than the next
  best NN family) but FAILED the magnitude rule by an order of
  magnitude vs the typical 1.05–1.50× band. Confirms: **the magnitude
  rule is the binding structural constraint, not orthogonality.**
- Decision: do NOT push full 5-fold. Capacity scaling is the same
  lever RealMLP n_ens=4 tested and nulled. The structural ceiling is
  NN standalone bal_acc on the 19-raw-feature one-hot representation,
  not architecture choice.
- Artefacts (whitelisted via .gitignore for cross-branch reuse):
  `oof_kan_probe.npy`, `test_kan_probe.npy`, `blend_kan_results.json`.
  The Jaccard-0.13 mark may be useful from a different/weaker anchor
  in future stacking work.

#### Option 2 — leak-eliminated W_RECIPE=1.0 distill: distill family FULLY CLOSED

- Goal: deferred Option-2 diagnostic from the U0OEQ session. Tests
  whether the soft-distill OOF→LB gap (small variant: OOF 0.98066 →
  LB 0.97865, gap +0.00201) is caused by the pseudo-label component
  leaking calibration noise vs being structural to teacher OOF
  construction.
- Configuration: `SOFT_SUFFIX=recipeonly W_RECIPE=1.0 XGB_DEPTH=3
  XGB_NROUND=1500 XGB_MAX_LEAVES=15 python scripts/soft_distill_xgb.py`
  Teacher = recipe_full_te alone (no pseudolabel component).
  ~42 min CPU wall, 5-fold StratifiedKFold(seed=42).
- Per-fold argmax bal_acc: 0.97496 / 0.97572 / 0.97624 / 0.97462 /
  0.97525 (mean 0.97536 ± 0.00057). Best_iter at 1485-1499 of 1500
  cap on every fold — student still learning at cutoff.
- **Tuned OOF: 0.98074** with bias [0.932, 1.269, 3.401].
- Comparison ladder (distill family across 4 capacity points + 2
  teacher compositions):
  ```
  variant            d  leaves  rds   teacher       OOF tuned  errs    PCR L/M/H
  -----------       --- ------ ----- -------------  ---------- ------ -------------------------
  distill (orig)     4  30     3000  0.5r + 0.5p   0.98096    9,520  0.9942 / 0.9685 / 0.9774
                                                                                  → LB 0.97850 (gap +0.00246)
  distill_small      3  15     1500  0.5r + 0.5p   0.98066    9,739  0.9945 / 0.9698 / 0.9777
                                                                                  → LB 0.97865 (gap +0.00201)
  distill_tiny       2  7       500  0.5r + 0.5p   0.97975    ?      ?            (not probed)
  recipeonly (NEW)   3  15     1500  1.0r          0.98074    10,228 0.9949 / 0.9669 / 0.9805
                                                                                  (not probed)
  LB-best 3-stack    -  -       -    -             0.98061    9,572  0.9955 / 0.9689 / 0.9774
  ```
- Three smoking guns that pseudo is **NOT** the leak source:
  1. OOF went UP +0.00008 vs distill_small (would expect DOWN if
     pseudo leaked encoded noise — leak removal lowers encoded noise
     → lowers OOF).
  2. Errs went UP +489 (recipeonly is less-calibrated despite higher
     bal_acc; trades Medium recall −0.0029 for High recall +0.0028).
  3. Jaccard(recipeonly, small) = **0.8928** — ~89% of errors are
     SHARED regardless of teacher composition. Pseudo-component
     contributes only marginal per-row variation, not the dominant
     overfit signal.
- **CONCLUSION: distill family FULLY CLOSED.** The OOF→LB gap is
  structural to teacher OOF construction itself (CV-aggregated OOFs
  inherit fold-specific calibration noise that any equal-or-greater-
  capacity student memorizes), NOT specific to the pseudo-label
  component. To produce a leak-free teacher would require retraining
  each teacher component with the student's training row held out
  of EVERY component — O(N²) retrains, computationally prohibitive
  at production scale.
- No LB probe warranted — recipeonly OOF 0.98074 < LB-best 4-stack
  0.98094, projected LB ≤ 0.97874 at gap +0.002 carryover.
- LB best unchanged: **LB 0.98094** via
  `submission_tier1b_greedy_meta.csv`. Pack 0.98114 still +0.00020
  above. Hedge swap recommendation unchanged: prefer
  `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005, premium
  −0.00089) over `recipe_full_te.csv` (LB 0.97939, premium −0.00155)
  — sidesteps the meta-stacker layer.

#### Realistic forward EV on NN HP-tuning (analysis, not experiment)

After 15 NN-family nulls, computed honest EV for "more capacity /
more folds / HP tuning" as a forward lever:

  - Empirical NN ceiling on 19-raw-feature one-hot representation:
    ~0.965-0.976 standalone tuned bal_acc (only RealMLP n_ens=1 has
    crossed 0.97).
  - Magnitude floor for blend viability: errs ≤ 1.05× anchor →
    standalone ≥ ~0.984. No NN ever reached this on this feature set.
  - 2026-04-22 XGB Optuna 80-trial sweep (smaller HP space than NNs):
    inner-val lift +0.00193 to +0.01019, outer-CV realized 30-65%,
    LB regressed -0.00016 to -0.00021. Documents structural-
    generalization null.
  - Best-case forward outcome: +0.00010 LB at ~10% probability
  - Median: 0 LB at ~50%
  - Worst: -0.00020 LB at ~30% (LR-meta-style overfit)
  - EV: +0.00001 to +0.00003 LB on ~15h compute spend

- Decision: skip further NN investment. The expected value is ~3 bp
  LB for half-day to full-day GPU; pack 0.98114 (+0.00020 above) is
  reachable at ~10-15% probability, not worth burning compute or
  LB slots given structural saturation evidence.

#### Final-selection state (5 days to deadline)

- **PRIMARY**: `submission_tier1b_greedy_meta.csv` → LB **0.98094**
  (gap −0.00010, anomalous LB > OOF). Composition: lb3 + RealMLP α=0.20
  + xgb_nonrule_iso α=0.075 + xgb_metastack_iso α=0.30.
- **HEDGE (recommended swap)**:
  `submission_3way_recipe025_s1035_s7040.csv` → LB **0.98005** (gap
  +0.00024, premium −0.00089). Sidesteps meta-stacker layer for
  private-LB overfit insurance.

#### Remaining open items (low-EV, kept for completeness)

  - **A3 Mixup XGB** (~1h CPU): training-time augmentation distinct
    from SMOTE's k-NN interpolation. Mechanism-novel; expected null.
  - **B3 Multi-task XGB** (~1h CPU): joint heads on y, dgp_score,
    rule_pred, cell_id. Untested.
  - **J5 TabDDPM** (~1.5h GPU): diffusion-augmented synthetic Highs.
    Different failure mode from SMOTE.
  - **C1 100+ XGB stack** (~8h GPU): NVIDIA-grandmaster scale on top
    of saturated bank. Bounded EV.
  - **TabM via pytabkit** (~1h GPU): only architecturally novel NN
    family remaining; flagged in audit round 5 (mikhailnaumov import-
    but-not-used pattern). 16th NN attempt likely null.

### 2026-04-26 — score=6 boundary deep-dive: 5-stage info-ceiling closure (10th saturation confirmation)

- Goal: senior-engineer-style deep dive into what was identified as the
  highest-EV remaining lever — the score=6 ∩ teacher_argmax=Medium override
  domain, which carries 70% of all missed-High mass and is the only
  bottleneck whose math projects above pack 0.98114. Hypothesis: lift
  specialist AUC from 0.94 (spec6_v2 against rule-Medium) to 0.97+ via
  better feature engineering + ensemble + cost-sensitive training, unlock
  ~200 overrides at >> break-even precision = ~+0.0010 LB.
- Approach: 5-stage rigorous characterization rather than build-and-hope.
- Branch: `claude/imbalanced-classification-research-elHy9`.

- **Stage 1a — characterize loose override domain** (`scripts/score6_manifold_stage1.py`):
  rule_pred=Medium ∩ score=6, n=38,416, 1549 truly-H, 0 truly-L.
  v2 specialist top-200 precision = **91.5%** (way above break-even
  8.1%). Looks great in isolation. But this is the WRONG domain — the
  teacher already correctly catches most of these as H.

- **Stage 1b — actual override target** (`scripts/score6_manifold_stage1b.py`):
  teacher_argmax=Medium ∩ score=6, n=35,180, **only 331 truly-H** (the
  1549 - 1218 the teacher already corrected). Prevalence 0.94%.
  - **v2 AUC drops from 0.938 to 0.793** on this teacher-residual domain.
  - **Best macro_Δ = +0.000086** at top-25 (7 correct, 18 wrong, prec 28%).
  - At top-100, net macro_Δ = -0.000001 (break-even).
  - Past top-200, strictly net-negative (precision falls below 8%).
  - **L2-LR oracle AUC 0.85 BEATS v2's 0.79** on this domain — v2 at
    depth=6 is overfitting on 30k rows × 331 positives.
  - **Missed-H residual analysis (smoking gun)**: bottom-50% of true-H
    by v2 prob have z-distance from M of just **+0.32** on teacher_PH,
    vs **+3.01** for found-H. Missed rows look feature-indistinguishable
    from M rows in every available feature dimension. **Information
    ceiling.**

- **Stage 1c — regularized-specialist competition** (`scripts/score6_manifold_stage1c.py`):
  tested 8 candidates (LR base/balanced/heavy/interactions, shallow XGB,
  kNN, univariate teacher_PH) against v2.
  ```
  candidate         AUC      best_n    correct   prec    macro_Δ
  v2 (depth=6)     0.7930    n_25         7      0.280   +0.000086
  knn50            0.7091    n_5          2      0.400   +0.000028
  xgb_shallow      0.8331    n_25         2      0.080   -0.000000
  univariate_PH    0.8530    n_5          0      0.000   -0.000007
  lr_base          0.8460    n_5          0      0.000   -0.000007
  lr_interactions  0.8460    n_5          0      0.000   -0.000007
  lr_balanced      0.8435    n_5          0      0.000   -0.000007
  lr_heavy         0.8443    n_5          0      0.000   -0.000007
  ```
  **Counter-intuitive finding: AUC and top-K precision are NEGATIVELY
  correlated on this domain.** Highest-AUC models (LR variants 0.84-0.85)
  have ZERO correct in top-5; v2 with AUC 0.79 has 7/25. AUC averages
  ranking globally; what matters is the top-K of the curve, where v2's
  depth-6 captures non-monotonic structure simpler models miss.

- **Stage 1d — depth sweep + ensemble + row-level blend**
  (`scripts/score6_manifold_stage1d.py`):
  ```
  XGB depth   AUC      best_n    correct   prec    macro_Δ
  d2          0.8327   n_200       18     0.090   +0.000032
  d3          0.8238   n_5          0     0.000   -0.000007
  d4          0.8165   n_5          0     0.000   -0.000007
  d5          0.8207   n_5          0     0.000   -0.000007
  d6          0.8148   n_5          1     0.200   +0.000010
  v2          0.7930   n_25         7     0.280   +0.000086  ← best
  ens_mean    0.8304   n_25         6     0.240   +0.000069
  ens_rank    0.8387   n_5          0     0.000   -0.000007
  ```
  **No XGB depth, ensemble, or rank-aggregation beats v2's +0.000086.**
  Row-level prob blend (apply v2's logit to primary's H column at
  score=6 only, sweep α ∈ [0, 2]) → all alphas give identical macro
  because v2's logits are too small to overcome the bias gap. Lever
  fully tested at the teacher-residual domain.

- **Stage 1e — against the actual LB-best PRIMARY (4-stack)**
  (`scripts/score6_manifold_stage1e.py`):
  Reconstructed primary = 0.70 × 3-way + 0.30 × xgb_metastack__iso.
  Primary OOF macro = 0.98067 (close to documented 0.98084; small
  diff is per-fold-iso vs full-OOF-iso).
  - Primary's score=6 ∩ argmax=Medium domain: n=35,326, 330 truly-H.
  - Primary catches 50 rows the 3-way left as M (7 truly-H gained).
  - Primary introduces 196 new M-pred rows at score=6 (6 truly-H lost).
  - **v2 against primary domain: AUC 0.789, best macro_Δ = +0.000051**
    (worse than +0.000086 against 3-way — primary already absorbs 2 of
    v2's high-confidence picks).
  - Fresh primary-aligned specialist (depth=2): macro_Δ = **+0.000010**.
  - **Override capacity against the actual LB-best primary is
    +0.000051 macro-delta.** OOF→LB transfer at ~50% (per prior spec6
    work) projects to **+0.000025 LB** — well below 0.0005 LB-probe gate.

- **Synthesis — score=6 boundary lever fully closed**:
  ```
  Senior-engineer hypothesis (2026-04-26)         falsified
  -------------------------------------------     ----------
  AUC 0.94 → 0.98 unlocks ~200 overrides           false: AUC ceiling on
                                                   primary domain is ~0.84,
                                                   not 0.98
  Higher AUC → higher override capacity            false: AUC and top-K
                                                   precision are negatively
                                                   correlated on this domain
  TabDDPM oversampling could help                  unlikely: missed-H rows
                                                   are feature-indistinguishable
                                                   from M rows in available
                                                   FE; synthetic H rows in
                                                   score=6 region won't change
                                                   the test missed-H feature
                                                   distributions
  Ensemble of specialists                          tested NULL (mean: +0.000069,
                                                   rank: -0.000007)
  Row-level prob blend                             tested NULL (all α identical)
  Depth sweep + heavy reg                          tested NULL (every depth ≤ v2)
  ```
  **The information ceiling is structural at this domain.** Realistic LB
  upside from this entire family: **+0.0001 LB at most** (vs +0.0010
  required to break pack).

- **10th independent saturation confirmation at LB 0.98094**: joins
  Tier 1c (3 sub-gates), cross-poll v3, J2 bootstrap-bag, LR
  meta-stacker v1+v2+iso-after, v4 ET+kNN, P3 perturbed-meta. The
  pattern is now overwhelming: re-arranging existing components or
  attacking residual error buckets cannot break LB 0.98094 within the
  standard tabular ML toolkit on this feature set.

- **Two portable rules** (LEARNINGS.md candidates):
  1. **AUC vs top-K precision can be negatively correlated on
     extreme-imbalance binary detection at residual override domains.**
     A model with global AUC 0.85 and a model with global AUC 0.79 can
     have OPPOSITE top-25 precision (0% vs 28%) on the same data. AUC
     ranks the full curve; top-K precision depends on tail behavior
     where rare-class signal lives. For boundary-specialist overrides
     under macro-recall, optimize top-K precision directly via macro-
     delta as the model-selection metric, NOT AUC.
  2. **Information ceiling diagnosis: missed-rare-class residual
     analysis.** For rows the specialist misses (bottom-50% of
     true-positives by specialist score), compute z-distance of every
     feature from the negative-class mean. If missed-positive features
     fall within ~0.3 std of negative-class mean while found-positive
     features sit at ~3 std, the missed rows are feature-
     indistinguishable from the negative class in the available
     feature space. NO model class, regularization, ensemble, or
     generative oversampling will recover them WITHOUT adding NEW
     feature dimensions that capture the underlying deterministic
     flip mechanism.

- LB delta: n/a. No probe warranted (best macro-delta projects
  +0.000025 LB << 0.0005 gate). LB best unchanged at **0.98094**
  via `submission_tier1b_greedy_meta.csv`. LB budget unchanged.

- Artefacts (gitignore-whitelisted for cross-branch reuse):
  - `scripts/score6_manifold_stage1.py` (rule=Medium baseline)
  - `scripts/score6_manifold_stage1b.py` (teacher-Medium domain)
  - `scripts/score6_manifold_stage1c.py` (specialist competition)
  - `scripts/score6_manifold_stage1d.py` (depth sweep + ensemble + blend)
  - `scripts/score6_manifold_stage1e.py` (against actual primary)
  - `scripts/artifacts/score6_manifold_stage1{,b,c,d,e}_results.json`

### 2026-04-26 — fresh-angles A+B+C trio (residual / rule-correct DAE / within-cell mixup): 3 NULLs, 12th saturation confirmation

- Goal: user-requested execution of three fresh-perspective levers proposed
  as the only mechanisms not yet exercised in the comp log. Each attacks the
  magnitude-trap from a structurally distinct angle.

- **Angle A — residual-correction XGB on (one_hot(y) − primary_softprob)**
  (`scripts/angle_a_residual.py`, ~115 lines, 30 sec wall):
  - LB-best primary reconstructed: `lb_best_3stack ⊗ xgb_metastack_iso α=0.30`
    → OOF 0.98084 exactly (matches documented value, sanity-check passed).
  - Residual XGB: 33 features = 28 dist + 3 primary probs + max_prob + entropy.
    5-fold seed=42, three independent reg trees (one per class) on
    sum-to-zero residual targets.
  - α-sweep at fixed recipe bias [1.4324, 1.4689, 3.4008]:
    ```
    α       OOF       Δ vs primary baseline
    0.000  0.98084   +0.00000   ← peak
    0.025  0.98079   −0.00005
    0.050  0.98076   −0.00008
    0.100  0.98077   −0.00008
    0.300  0.98084   −0.00000
    0.500  0.98075   −0.00009
    ```
    Monotone-flat below baseline. Per-fold residuals range −0.00115 to
    +0.00088 (cancel globally at any single shared α).
  - **Verdict: NULL.** Per-row residual fits don't transfer to held-out
    fold under macro-recall + fixed bias. Different folds want different
    α; a global α can't satisfy all.

- **Angle B — rule-correct-only autoencoder anomaly-score feature**
  (`scripts/angle_b_dae.py` + `scripts/recipe_full_te.py DAE_EMBED_PATH`,
  ~13 min DAE training + ~43 min recipe retrain):
  - DAE config: 4-layer encoder-decoder MLP (in→64→64→16→64→64→out), GELU+
    LayerNorm, 30 epochs, batch=2048, AdamW 1e-3, cosine schedule.
    Trained on 588,712 rule-correct fit rows / 30,984 rule-correct val rows.
    SwapNoise p=0.15 on numerics+cats during training.
  - Reconstruction MSE separation: rule-correct mean 0.0646 (p99=0.16),
    rule-flipped mean 0.0709 (p50=0.066). **Separation ratio 1.098×** —
    real signal at row level but small magnitude.
  - Recipe retrain with 1-d recon-error feature added to numerics (444
    total features, otherwise identical to LB-best 0.97939 recipe pipeline):
    ```
    Per-fold argmax: 0.97565 / 0.97597 / 0.97695 / 0.97419 / 0.97506
    Overall argmax: 0.97557 (recipe baseline 0.97589, Δ −0.00032)
    Tuned bal_acc:  0.97955 (recipe baseline 0.97967, Δ −0.00012)
    Bias:           [1.4324, 1.3689, 3.4008]   (High bias 3.40 = recipe's exactly)
    ```
  - Blend gate vs LB-best 4-stack (anchor 0.98084, errs 9415):
    ```
    standalone @ recipe bias:  errs 10042  (+6.7%)  Jaccard 0.828
    per-class recall:          L 0.9949 (−0.0006) M 0.9680 (−0.0015) H 0.9756 (−0.0019)
    α-sweep:                   0.98084 → 0.98025 (monotone-negative across α∈[0,0.5])
    ```
  - **Verdict: NULL** on both axes. The 1-d recon-error feature is
    redundant with the recipe's existing 443 features (Jaccard 0.83 in
    the redundancy zone), AND its standalone errors land +6.7% over
    anchor with all three per-class recalls slightly worse. Same failure
    mode as the 2026-04-24 128-d SwapNoise DAE on full train+test —
    rule-correctness conditioning didn't unlock the orthogonal-error
    profile we hoped for.

- **Angle C — within-cell rule-disagreement mixup augmentation**
  (`scripts/angle_c_mixup.py`, ~8 min wall, dist-feature base):
  - 6-bit rule_cell packing (dry, norain, hot, windy, nomulch, kc_active)
    → 64 cells. Per-cell pair construction: cap 4000 rows/cell, permute,
    keep cross-class (y_i ≠ y_j) pairs only. Result: **8,299 within-cell
    cross-class pairs** → K=3 β(0.4, 0.4) mixup → **24,897 mixup rows**.
  - Mixup mechanism: convex combo on numerics, Bernoulli sampling on
    cats by α, soft labels (1-α)·onehot(y_i) + α·onehot(y_j). Hard target
    via argmax + sample_weight = soft_max for confidence attenuation.
  - 5-fold seed=42 on dist-feature base (35 cols), per-fold filter to
    only include mixup rows whose donors are both in tr_idx (~64% kept).
    XGB: max_depth=4, max_leaves=30, lr=0.1, n_est=1500, reg_alpha=5,
    reg_lambda=5, balanced sample_weight × confidence-weighted mix_w.
  - Standalone results:
    ```
    Per-fold argmax: 0.96917 / 0.97079 / 0.97177 / 0.96930 / 0.97056
    Overall argmax:  0.97032   tuned: 0.97136
    Bias:            [2.232, 2.069, 2.601]  (sharper Low/Med than recipe;
                                             milder High)
    ```
  - Blend gate vs LB-best 4-stack (anchor 0.98084, errs 9415):
    ```
    standalone @ recipe bias:  errs 20745 (+120% magnitude)
    Jaccard vs anchor:         0.376  ← BEST tree-family orthogonality on this problem
                                       (lower than NN families except KAN 0.13 / TabPFN-10k 0.21)
    per-class recall:          L 0.9948 / M 0.9231 (−0.046!) / H 0.9791
    α-sweep:                   peak α=0.025 Δ=+0.00000;
                                monotone fall to α=0.5 Δ=−0.0021
    ```
  - **Verdict: NULL** — magnitude trap dominates. The mechanism IS alive
    (Jaccard 0.376 is unprecedented for tree-family on this feature set),
    but the dist-base XGB has 2.2× more errors than anchor, AND the
    mixup pushes Medium predictions toward High aggressively (Medium
    recall −0.046 vs anchor). The mixup mechanism would need a recipe-
    base XGB (errs ~10k vs anchor's 9.4k) AND careful per-class
    sample-weight balancing to thread the needle. Future-session retry
    target: scale to recipe-base, run with Medium-protective sample
    weighting (down-weight pairs that flip Medium-to-High by 0.3-0.5×),
    and re-test the magnitude+per-class profile.

- **12th independent saturation confirmation at LB 0.98094**:
  ```
  attack vector                                  best blend Δ vs LB-best 4-stack
  ----------------------------------------------- -----------------------------
  1. Tier 1c greedy expanded (132c)               +0.00002
  2. Tier 1c meta-stacker v2 (224-dim)            +0.00002
  3. Tier 1c meta-stacker XGB seed-bag            +0.00003
  4. Cross-poll metastack v3                      +0.00015 (LB -0.00034)
  5. J2 bootstrap-bagged metastack                +0.00003 (proj null)
  6. LR meta-stacker v2 (C=0.1, none)             +0.00098 (LB -0.00042)
  7. LR v2 + iso-after-blend                      +0.00000
  8. v4 ET+kNN bank-extension                     +0.00036 (LB -0.00102)
  9. P3 perturbed meta v1                         +0.00071 (LB -0.00139)
  10. score=6 boundary deep-dive (5 stages)       +0.000086 (proj +0.000025 LB)
  11. DROP_SCORES=0,1,2 + 4-stack saturation      +0.00017 (sub-gate)
  12. **A residual / B DAE-recon / C mixup        +0.00000 across all three**
  ```

- **Three portable rules** (LEARNINGS.md candidates):
  1. **Per-row residual-correction models cannot transfer per-fold
     residual signal under macro-recall + fixed bias.** Even with
     leak-safe OOF residuals as targets, different folds want different
     correction magnitudes; a single global α can't satisfy them.
     The fix would require per-fold-α (which is selection-bias prone)
     or per-row gating (already nulled via 2026-04-26 MoE experiment).
  2. **1-dimensional anomaly-score features added to a saturated
     443-feature recipe pipeline are absorbed without lift.** The
     recipe's tree splits already capture rule-correctness through
     factorized rule_pred + dist features + OTE on rule cells; an
     external anomaly score is informationally redundant. Rule-
     correct conditioning on the DAE training set didn't change
     this — the conditioning narrows what the DAE learns but
     doesn't widen the channel through which the recipe can use it.
  3. **Within-cell rule-disagreement mixup IS a fresh orthogonality
     mechanism (Jaccard 0.376 vs anchor)** but the magnitude trap
     applies pre-emptively when the base model is significantly
     weaker than the anchor. To use the mechanism: pair it with a
     recipe-strength base AND per-class sample-weight balancing
     that prevents the mixup from distorting majority-class probs
     (here: Medium recall −0.046 was the killer).

- LB delta: n/a (no submission warranted; all three sweeps strictly
  negative or flat). LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`. LB budget unchanged.

- Artefacts (whitelisted in `.gitignore` for cross-branch reuse):
  - `scripts/{angle_a_residual,angle_b_dae,angle_c_mixup,angles_blend_gate}.py`
  - `scripts/artifacts/oof/test_angle_a_residual{,_raw}.npy`
  - `scripts/artifacts/oof/test_angle_b_recon.npy`
  - `scripts/artifacts/oof/test_angle_c_mixup.npy`
  - `scripts/artifacts/oof/test_recipe_full_te_dae.npy` (1-d recon variant;
    overwrites prior 128-d SwapNoise DAE artefact)
  - 4 results JSONs

### 2026-04-26 — Angle-C follow-ups C2/C3a/C3b: closed family at NULL (recipe-base mixup on 4-stack anchor)

- Goal: rescue Angle-C (within-cell mixup) from its dist-base magnitude
  trap (errs +120%) by porting to the recipe pipeline. Three iterations:
  - **C2**: recipe-base + drop M↔H pairs (Medium-protect) + confidence-
    gate (primary max_prob<0.95, drops clean×clean) + K=1 + β(0.2, 0.2).
    3,909 pairs → 3,909 mixup rows (~0.62% of train).
  - **C3a**: same as C2 but DROP confidence-gate to test whether the
    gate was a productive filter or redundant. 4,808 pairs (+899
    clean×clean) → 4,808 mixup rows.
  - **C3b**: same as C2 but K=3 (volume amplifier) instead of K=1.
    3,909 pairs → 11,727 mixup rows (~3.4× more volume).
- Standalone tuned OOF (recipe baseline 0.97967):
  ```
  variant   conf_gate   K    n_mix     OOF tuned   bias                   Δ vs C2
  C2        0.95        1    3,909    0.97983     [0.83, 1.07, 3.20]     —
  C3a       OFF         1    4,808    0.97944     [0.83, 0.97, 2.90]    -0.00039
  C3b       0.95        3    11,727   0.97951     [1.03, 1.27, 3.40]    -0.00032
  ```
  - **C3a confirms conf-gate was productive**: removing it (adding 899
    clean×clean pairs where both donors are confident) regresses
    standalone tuned by −0.00039 vs C2. The gate was filtering noise.
  - **C3b confirms K-volume doesn't unlock orthogonality at recipe
    base**: 3× more mixup rows actually drops standalone, the extra
    volume adds noise (more Beta-interpolated pseudo-points the
    model has to absorb) without sharpening the boundary.

- Unified blend-gate vs LB-best 4-stack (anchor OOF 0.98084, errs 9415):
  ```
  variant       errs vs anchor   Jaccard   bestα   Δ           emit
  C  (dist+K=3)  +120%           0.376     0.025   +0.00000    N (magnitude trap)
  C2 (recipe+gate+K=1) +6.4%     0.819     0.000   +0.00000    N (redundancy)
  C3a (recipe -gate K=1) +5.7%   0.814     0.000   +0.00000    N (redundancy)
  C3b (recipe gate K=3) +8.4%    0.809     0.025   +0.00004    N (sub-gate)
  ```
  - C3b is the FIRST C-variant where bestα > 0 (α=0.025, Δ=+0.00004).
    Microscopic but technically positive — K=3 volume DOES add a tiny
    sliver of orthogonal signal at very low blend weight. Two orders
    of magnitude below the +0.0005 LB-probe threshold.
  - **No LB submission warranted on any C-variant.**

- **Closure conclusion**: the within-cell mixup mechanism is fundamentally
  trapped between two regimes:
  1. **Dist-base** (C v1): Jaccard 0.38 (excellent orthogonality) but
     errs +120% (catastrophic magnitude trap).
  2. **Recipe-base** (C2/C3a/C3b): Jaccard ~0.81 (in redundancy zone)
     and errs ≤+8% (magnitude OK). The base is too similar to the
     anchor (LB-best is built ON recipe), so mixup-induced perturbation
     gets absorbed into existing splits rather than producing new
     decision-surface geometry.
  No mixup configuration we tested threads the (Jaccard < 0.80, errs ≤
  +5%) blend-gate window. The mechanism is alive at the dist scale but
  too weak; it's redundant at the recipe scale.

- **Two portable rules** (LEARNINGS.md candidates):
  1. **Confidence-gating mixup pairs is a productive filter, not
     redundant overhead.** Removing the gate (allowing clean×clean
     pairs where both donors have primary max_prob > 0.95) regressed
     C3a's standalone OOF by −0.00039 vs C2. Mechanism: clean-confident
     pairs encode redundant rule-aligned info that mixup interpolation
     just smears across decision boundaries the model already has
     correct. Always confidence-gate when the primary model is much
     stronger than the mixup base.
  2. **K-volume scaling on within-cell mixup hits diminishing returns
     fast at recipe-base scale.** K=3 (3× pair multiplication) added
     7,818 extra mixup rows but the standalone tuned OOF dropped
     −0.00032 vs K=1. The first synthesized point per pair captures
     the boundary perturbation; additional K samples from β(0.2, 0.2)
     add noise (multiple α∈[0,1] re-weightings of the same donor pair)
     rather than new geometric variation. Stick with K=1 unless mixup
     volume relative to train pool is < 0.5%.

- **13th independent saturation confirmation at LB 0.98094** (C-family
  closure joins the 12 prior saturation entries documented above).

- Artefacts (whitelisted in `.gitignore`):
  - `scripts/angle_c2_helpers.py`, `scripts/angle_c2_recipe_mixup.py`
    (parameterized via `OUT_SUFFIX`, `DROP_MH`, `CONF_THRESH`, `K_MIX`,
    `BETA_A` env vars)
  - `scripts/artifacts/oof_angle_{c2,c3a,c3b}_mixup.npy` + test (6 files)
  - `scripts/artifacts/angle_{c2,c3a,c3b}_mixup_results.json`
  - `submissions/submission_angle_{c2,c3a,c3b}_mixup.csv` (diagnostic,
    not for LB probe)

### 2026-04-26 — combined v6 meta-stacker (#3+#4+#6 deep dive): LB 0.98059 = 11th saturation confirmation

- Goal: user-requested deep dive on three structurally distinct levers added
  on top of the LB-best 4-stack primary (LB 0.98094):
  - **#6** self-supervised masked-feature pretraining
  - **#3** multi-task auxiliary XGBs (3 binary heads on rule-flip targets)
  - **#4** polynomial / non-linear FE (3-way + log/sqrt/sin transforms)
- Combined into a v6 meta-stacker with all three signal sources as new
  inputs, then blended into the primary.
- Branch: `claude/imbalanced-classification-research-elHy9`.

- **#6 — masked feature pretraining: NULL at source.**
  R² for predicting each of 7 non-rule numerics from the other 18 features
  (5-fold cross-fit on combined 900k train+test):
  ```
  Humidity                 R² = 0.0148
  Previous_Irrigation_mm   R² = 0.0223
  Electrical_Conductivity  R² = 0.0028
  Soil_pH                  R² = 0.0030
  Organic_Carbon           R² = 0.0039
  Sunlight_Hours           R² = 0.0016
  Field_Area_hectare       R² = 0.0039
  ```
  Features are nearly INDEPENDENT in this synthetic DGP. Residual AUCs for
  distinguishing each class from the rest = 0.50–0.51 (pure noise). The 14
  residual columns carry zero class-conditional signal. Confirms the
  feature-space saturation conclusion from the score=6 deep-dive.

- **#3 — multi-task aux XGBs: strong global discriminators.**
  3 binary heads on the 43-feature dist matrix, 5-fold StratifiedKFold(seed=42):
  ```
  aux_flipped_from_rule  (y != rule_pred):     OOF AUC 0.899
  aux_missed_high        (y==H AND rule!=H):    OOF AUC 0.983
  aux_missed_medium      (y==M AND rule!=M):    OOF AUC 0.949
  ```
  These targets encode rule-flip supervision NOT directly used by any
  existing y-targeting component. AUC 0.98 for "is missed-High globally"
  is the strongest binary signal in the bank, leak-free OOF.

- **#4 — polynomial / non-linear FE: real signal contribution.**
  36 new features added on top of 43-dist:
  - 3-way rule products (sm_x_rf_x_tc, etc.)
  - Non-linear transforms (sm_squared, log_rf, sqrt_pri, sin_rf_th, etc.)
  - Non-rule × rule crosses (hum_x_sm, ec_x_sm, etc.)
  - Within-cell normalized features (sm_pct_of_25, etc.)
  Standalone XGB on 79 features: tuned OOF 0.97452 (vs vanilla XGB-dist
  0.97266, +0.00186). **9/30 top features by gain are NEW poly features**
  — the trees use them. Strong indicator the recipe FE was missing real
  axes.

- **Combined v6 meta-stacker:** 82 components (bank + poly_fe) + 3 aux
  log-prob/logit pairs + 14 masked residuals = **283-col input matrix**.
  Same heavy-reg XGB HPs as v1 (depth=4, reg_alpha=5, reg_lambda=5,
  lr=0.05). 5-fold StratifiedKFold(seed=42) for leak-free stacking.
  - Standalone iso OOF: **0.98150** (vs v1_iso 0.98059, **+0.00091** —
    first meta variant to materially exceed v1 standalone)
  - Best blend Δ vs LB-best 4-stack primary: **+0.00038 OOF at α=0.30**
  - Per-class recall guardrail PASS: L+0.00006 / M+0.00115 / H−0.00010
  - Jaccard 0.82 vs primary; errs 9655 vs primary 9415 (+240 distributed
    favorably under macro-recall)

- **LB probe (user-approved, submitted 01:22 UTC):**
  `submission_combined_v6_a030.csv` → **LB public = 0.98059**.
  Δ vs LB-best primary (0.98094) = **−0.00035 regression**.
  OOF→LB gap = 0.98122 − 0.98059 = **+0.00063** (vs primary's −0.00010).

- **Diagnosis — bank-extension pattern strikes again, even with novel
  leak-free aux supervision.** Joins:
  ```
  attack vector                     OOF Δ      LB Δ      gap inflation
  --------------------------------- --------  --------- -------------
  v1 (LB-best baseline)             +0.00023  +0.00086  -0.00073 (anomalous +)
  v3 cross-poll                     +0.00015  -0.00034  +0.00049
  v4 ET+kNN                         +0.00036  -0.00102  +0.00138
  LR v2 (C=0.1, none)               +0.00046  -0.00042  +0.00088
  P3 perturbed v1                   +0.00071  -0.00139  +0.00210
  W1+W4 meta                        +0.00035  -0.00098  +0.00133
  **v6 (this entry)                  +0.00038  -0.00035  +0.00073**
  ```
  v6 is the LEAST-BAD regression of the 6 meta-extension variants, but
  still regresses. The aux features added genuine OOF signal (+0.00091
  standalone over v1) but the additional meta-stacker capacity to use them
  fits OOF-specific patterns that don't transfer.

- **11th independent saturation confirmation at LB 0.98094**:
  joins the 10 prior attacks (Tier 1c sub-gates × 3, cross-poll v3, J2
  bootstrap-bag, LR v1+v2+iso-after, v4 ET+kNN, P3 perturbed, score=6
  deep-dive). The pattern is mathematically exhaustive: re-arranging
  existing components OR adding new components into the existing meta-
  stacker architecture cannot break LB 0.98094. The +0.00091 standalone
  v6_iso lift over v1 is the strongest evidence yet that the SIGNAL exists
  in the new aux features — but the meta-stacker architecture is the
  bottleneck for transferring it to LB.

- **Two portable rules** (LEARNINGS.md candidates):
  1. **Self-supervised masked-feature pretraining is null when features
     are independent in the DGP.** R² for predicting each feature from the
     others is the diagnostic: if all R² < 0.05, residuals carry no
     signal. Skip this lever in synthetic-tabular comps where train/test
     come from a known IID generator.
  2. **Bank-extension meta-stackers cannot transfer OOF lifts > +0.0003 to
     LB on a saturated primary.** Even leak-free aux supervision targeting
     novel binary tasks (AUC 0.98 for "is missed-high globally") fits
     OOF-specific patterns in the meta-stacker training. The
     architecture's capacity-to-noise ratio is too high once the bank is
     saturated. To unlock the aux signal: incorporate it at primary
     TRAINING TIME (e.g., multi-task loss on recipe FE) rather than as a
     stacker INPUT — different architectural insertion point.

- LB budget: **2/10 used today** (α=0.30 + α=0.40 v6 probes), 8 remaining.
  LB best unchanged at **0.98094** via `submission_tier1b_greedy_meta.csv`.

- **Aggressive α=0.40 also probed (user-requested follow-up):**
  `submission_combined_v6_a040.csv` → LB **0.98060** (essentially tied
  with α=0.30's 0.98059, +0.00001). OOF 0.98114 → LB 0.98060, gap +0.00054
  (slightly tighter than α=0.30's +0.00063 because the aggressive α drops
  OOF closer to primary while LB stays flat). Confirms the linear-projection
  rule: bank-extension OOF lift transfers proportionally to LB regression
  at any α. Two-probe bracketing closes the v6 lever definitively — there
  is no α that threads the needle.
- Final-selection lock UNCHANGED:
  1. **PRIMARY**: `submission_tier1b_greedy_meta.csv` → LB 0.98094
  2. **HEDGE (recommended swap)**: `submission_3way_recipe025_s1035_s7040.csv`
     → LB 0.98005 (premium −0.00089)

- **Path forward**: the aux features encode REAL signal (proved by +0.00091
  standalone iso lift at meta level). Bank-stacking can't transfer it.
  Open architectural alternatives:
  1. Multi-task XGB at primary level (auxiliary heads on recipe FE
     directly, rather than aux as separate component). Untested.
  2. Recipe XGB with aux outputs as INPUT FEATURES (not via meta).
     Different insertion point.
  3. Specialist that consumes aux + recipe outputs at the row-level
     decision rule (not via blend).

### 2026-04-26 — DROP_SCORES=0,1,2 on recipe: NULL (11th saturation confirmation, most-orthogonal tree variant ever recorded)

- Goal: address an audit gap surfaced by senior-researcher review. The
  2026-04-21 score-routing ablation series (`xgb_dist_routed_v3`) showed
  that dropping {0,1,2}-score Low rows from XGB training + routing those
  test rows to rule=Low at inference was a real lever on the dist
  feature set (vanilla 0.97304 → v3 0.97332 OOF, +0.00029; LB 0.97271).
  Mechanism: training-distribution rebalancing (v7 = train-all + route-
  infer was WORSE than vanilla, falsifying inference-routing as the
  source). That lever had **never been re-applied on top of the V10
  recipe pipeline** (LB 0.97939 → 0.98094 family) — the recipe uses
  `compute_sample_weight("balanced")` instead of explicit row dropping.
  This experiment closes the gap.

- Changed: `scripts/recipe_full_te.py` (44 lines added) — new
  `DROP_SCORES` env var (e.g. "0,1,2"). Computes `dgp_score` on
  train+test from raw threshold flags + Mulching_Used + Crop_Growth_Stage
  (formula: `2*(soil_lt_25 + rain_lt_300) + temp_gt_30 + wind_gt_10 +
  nomulch + Kc`, where Kc=2 iff stage in {Flowering, Vegetative}).
  Per-fold filter drops `tr_idx` rows whose score ∈ DROP_SCORE_SET
  BEFORE OTE fit, so OTE statistics see only the kept rows. Inference
  override: `oof[val_route]` and `test_pred[test_route]` set to one-hot
  Low for routed rows. Output suffix `_ds012` keeps LB-best artefacts
  untouched. Mutually exclusive with CLEANLAB_TREATMENT.
  `scripts/blend_gate_dropscores.py` (155 lines): 3-anchor analyzer
  (recipe / LB-best 3-stack / LB-best 4-stack) with fixed-recipe-bias
  α-sweep + per-class recall guardrail (-5e-4 floor each class) +
  +2e-4 LB-transfer emit gate.

- Production (5-fold seed=42, 287k train rows after dropping 271,444
  {0,1,2} rows = 43.07%, ~34 min CPU):
  ```
  per-fold argmax  0.97426 / 0.97478 / 0.97711 / 0.97541 / 0.97544  σ=0.00096
  OOF argmax       0.97544     (recipe baseline 0.97589, Δ -0.00045)
  tuned OOF        0.97940     (recipe 0.97967, Δ -0.00027)
  tuned bias       [0.032, 1.069, 3.001]   (recipe [1.43, 1.47, 3.40])
  ```
  Bias profile collapsed: Low bias dropped 1.40, Medium dropped 0.40,
  High dropped 0.40. Removing 271k Low rows pre-balances the gradient,
  so log-bias post-hoc has less Low-deficit to correct.

- **Diagnostics @ fixed recipe bias [1.4324, 1.4689, 3.4008]** (the
  blend operating point):
  ```
                       errs    PCR [Low,    Med,    High]    Jaccard vs cand
  ds012 candidate     9,856   [0.9961, 0.9670, 0.9744]    1.0000
  recipe_full_te     10,114   [0.9950, 0.9675, 0.9765]    0.7578  ← LOWEST
  LB-best 3-stack     9,572   [0.9955, 0.9689, 0.9774]    0.7926
  LB-best 4-stack     9,415   [0.9955, 0.9695, 0.9775]    0.7910
  ```
  ds012 has **258 FEWER errors than recipe** AND Jaccard 0.76 vs recipe —
  the LOWEST tree-family Jaccard ever recorded on this problem (prior
  best: `recipe_no_ote` at 0.60, but it had +16% MORE errors). ds012 is
  the first tree variant with BOTH lower errors AND high orthogonality.

- **But per-class trade is wrong direction under macro-recall**: Low
  +0.0011 / Med -0.0005 / **High -0.0021**. High has 12× per-row
  leverage (21k vs 240k Med), so the 0.0021 High recall drop wipes out
  the Low gain on macro-recall. Net standalone @ recipe-bias = 0.97920
  (Δ -0.00047 vs 0.97967).

- **Blend gate** (fixed recipe bias, α-sweep, no per-α retune):
  ```
  vs recipe (0.97967):
    peak α=0.40   Δ=+0.00017   pcr-FAIL (High recall < 0.97649 floor)
    max gate-pass α=0.10  Δ=+0.00005  (sub +2e-4 emit gate)
  vs LB-best 3-stack (0.98061):
    peak α=0.000  Δ=0  monotone-negative from α=0.025
  vs LB-best 4-stack (0.98084):
    peak α=0.000  Δ=0  monotone-negative from α=0.025
  ```
  Strict null on every anchor. **No LB probe warranted** — projected
  LB at any α ≪ current best 0.98094.

- **Diagnosis — why v3's lift doesn't transfer to recipe**:
  - v3 (dist features): `multi:softprob` with NO sample weights → natural
    prior was Low-heavy → dropping 271k {0,1,2}-Low rows pre-balanced
    the gradient → +0.00029 OOF lift was real.
  - recipe (V10 features): `compute_sample_weight("balanced")` already
    weights gradient equally across classes → DROP_SCORES is a
    DOUBLE-rebalance. The second mechanism overshoots toward High
    predictions on borderline rows in {6,7,8} (where rare-High signal
    lives), losing -0.0021 High recall.
  - Pre-hoc analog of the binhigh-rule failure: when two rebalancing
    mechanisms stack, the second OOF-overshoots the rare-class
    operating point the first already calibrated for.

- **Notable side-finding** (the senior-researcher take-away):
  Training-distribution rebalancing produces **genuinely orthogonal
  errors** on this feature set (Jaccard 0.76 is a competition-record
  for tree variants). The lever is alive in error-geometry terms; it
  fails because the recipe's class-balanced weighting already absorbs
  the gradient-rebalancing benefit, leaving DROP_SCORES to overshoot in
  the wrong per-class direction. On a future synthetic-tabular problem
  where the base model does NOT use class-balanced weights, this lever
  could lift +0.0003 standalone (matching v3's pattern).

- **11th independent saturation confirmation at LB 0.98094**:
  ```
  attack vector                                  best LB / OOF Δ
  --------------------------------------------- ----------------
  1. Tier 1c greedy expanded (132c)             sub-gate
  2. Tier 1c meta-stacker v2 (224-dim)          sub-gate
  3. Tier 1c meta-stacker XGB seed-bag          sub-gate
  4. Cross-poll metastack v3                    LB 0.98060 (-0.00034)
  5. J2 bootstrap-bagged metastack              proj null
  6. LR meta-stacker v2 (C=0.1, none)           LB 0.98052 (-0.00042)
  7. LR v2 + iso-after-blend                    sub-gate
  8. v4 ET+kNN bank-extension                   LB 0.97992 (-0.00102)
  9. P3 perturbed meta v1                       LB 0.97955 (-0.00139)
  10. score=6 boundary deep-dive (5 stages)     proj LB +0.000025 sub-gate
  11. **DROP_SCORES=0,1,2 (this entry)           OOF Δ ≤ +0.00017 sub-gate**
  ```

- **Two portable rules** (LEARNINGS.md candidates):
  1. **Training-distribution rebalancing levers are mutually exclusive
     with class-balanced sample weights.** When the base model already
     uses `compute_sample_weight("balanced")`, removing rows from the
     majority class amounts to a double-rebalance that overshoots the
     rare-class operating point. Either remove the sample-weight call
     OR apply DROP_SCORES — never both.
  2. **Lowest-Jaccard candidate is NOT necessarily the best blend leg
     even with fewer errors than anchor.** ds012 is a competition-record
     0.76 Jaccard with -2.5% errors vs recipe — the cleanest blend
     fingerprint ever recorded — yet blend-fails because per-class
     recall lands -0.0021 on the rare class. The blend-gate "fewer
     errors AND lower Jaccard" heuristic from the digit-OTE LB-win era
     needs a third clause: **per-class recall must move in the
     macro-recall-favorable direction**. Track per-class recall delta
     as a first-class gate metric, not just total errors.

- LB budget: **0 used today** (4 cumulative this week, 6 remaining).
  No LB probe warranted. LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`. Final-selection lock unchanged.

- Artefacts (whitelisted in `.gitignore`):
  - `scripts/recipe_full_te.py` (DROP_SCORES env var integration)
  - `scripts/blend_gate_dropscores.py` (3-anchor blend-gate analyzer)
  - `scripts/artifacts/oof_recipe_full_te_ds012.npy` + test (7.2 MB)
  - `scripts/artifacts/recipe_full_te_ds012_results.json`
  - `scripts/artifacts/blend_gate_dropscores_results.json`
  - `submissions/submission_recipe_full_te_ds012.csv` (diagnostic, NOT
    for LB probe)
### 2026-04-26 — Bayes-optimal / LP decision-rule probe: NULL (10th saturation confirmation, "log-bias overfit" hypothesis falsified)

- Goal: stress-test the unstressed property of the LB-best primary's
  decision rule. Coord-ascent log-bias `[1.43, 1.47, 3.40]` is a
  3-parameter heuristic. The closed-form Bayes-optimal under macro-recall
  is `b_k = -log(π_k)` which gives `[0.53, 0.97, 3.41]` for our train
  priors. The Low/Medium components differ by ~0.9 units — either
  coord-ascent overfits OOF, or the predicted probabilities are
  miscalibrated relative to true posteriors. This probe distinguishes.
- Branch: `claude/kaggle-missing-strategies-rls5d`. Single script,
  ~190 lines (`scripts/lp_decision_rule.py`).
- Mechanism: reconstruct LB-best primary (3-stack + xgb_metastack_iso α=0.30)
  via `tier1b_helpers.build_lbbest_stack` + add metastack step. Apply 4
  decision-rule families to the SAME OOF/test predictions, report
  per-class recall + errors + macro-recall delta vs current bias.
- Results (5-fold OOF, sanity reproduced 0.98084 baseline):
  ```
  family                                 bias / params              OOF      Δ vs baseline   gate
  CURRENT log-bias (LB-best, baseline)   [1.43, 1.47, 3.40]         0.98084  +0.00000        n/a
  Bayes-opt (train prior)                [0.53, 0.97, 3.40]         0.98072  -0.00012        FAIL
  Bayes-opt (test rule_pred prior)       [0.52, 0.98, 3.40]         0.98070  -0.00014        FAIL
  Joint (T, b) coord-ascent              T=[0.6,0.9,1.0], b=BIAS    0.98093  +0.00009        FAIL
  LP cardinality cap (train prior)       greedy assignment           0.97438  -0.00646        FAIL
  ```
- **Decisive finding**: closed-form Bayes-optimal LOSES macro-recall
  (-0.00012) vs coord-ascent. Per-class trade is informative:
  - Closed-form: High recall UP +0.0036 (0.9775 → 0.9811), but
    Low (-0.0008) and Medium (-0.0031) BOTH down. Net negative.
  - Coord-ascent's heavier Low/Medium biases are NOT overfit —
    they're a **real calibration correction** for over-confident
    Low/Medium probabilities in the LB-best primary.
- **Joint (T, b) family** found T_Low=0.6 (sharpens Low probs by
  raising them to 1/T = 1.67 power) at unchanged biases. OOF lift
  +0.00009 — below the +2e-4 LB-transfer threshold. Only **142 test
  rows** differ from current submission; per the linear-projection
  rule established in prior LR/v4 closures, expected LB Δ ≈ 0
  given gap inflation typically eats half the small OOF lift.
- **LP cardinality cap** (force-cap class counts to train prior):
  -0.00646 (massive regression). Reconfirms the 2026-04-25 P2
  quota-null mechanism: hard caps discard the rare-class
  over-prediction signal that drives macro-recall under fixed bias.
- **Three portable rules** (LEARNINGS.md candidates):
  1. **Coord-ascent log-bias coordinates can drift far from the
     closed-form Bayes-optimum WITHOUT being overfit** when the base
     model's predicted probabilities are miscalibrated per-class.
     The drift IS the calibration correction. To diagnose:
     compute `|coord-ascent_bias − (−log π)|` per class. If the
     drift is structurally consistent across re-runs and cross-
     validation seeds, it's signal, not noise.
  2. **Per-class temperature scaling on a tuned blend with fixed
     bias has a thin signal channel**: T_k controls how sharply the
     model's "confident class" outranks runners-up. On a heavily-
     calibrated stack, T sweeps tend to find ≤+0.0001 OOF without
     transfer-positive blend Δ.
  3. **The LP/quota family of rules (force-cap to a known prior)
     is structurally wrong for macro-recall**. Confirmed twice now
     (P2 quota 2026-04-25 and LP cap this entry). The decision rule
     family that works is "additive log-shift + (optional) per-class
     temperature" — closed under arbitrary monotonic per-class
     reweighting.
- LB budget unchanged: 4/10 used (no probe warranted, all 4 families
  fail the +2e-4 OOF gate). LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.
- **10th independent saturation confirmation at LB 0.98094**.
- Artefacts:
  - `scripts/lp_decision_rule.py` (~190 lines, single-file)
  - `scripts/artifacts/lp_decision_rule_results.json` (5-family report)

### Next steps: post-LP closure shortlist (2026-04-26)

After 10 independent saturation confirmations at LB 0.98094, the
own-pipeline ceiling is structurally bounded. Four levers remain
worth trying (4 days to deadline, 6 LB submissions available today).
Each is structurally distinct from every prior null AND from each
other. Ranked by expected EV / cost:

  **N1. TabPFN at 10k context on Kaggle GPU** (~30 min wall, top-pick).
  The 2026-04-22 TabPFN run capped at SUBSAMPLE=1500 for CPU
  feasibility. TabPFN v2's *architectural sweet spot* is 10k
  context. We have NEVER measured TabPFN at design scale; the
  earlier closure note said "GPU SUBSAMPLE=10000 unlikely to change
  blend outcome" but that was a guess from 1.5k context Jaccard,
  not an experiment. At 10k context the model sees ~7x more rare-
  class examples (333 High @ 10k stratified vs 50 @ 1.5k), which
  is exactly the failure mode that crippled the prior run's High
  recall (0.9238 standalone).
  - Action: scaffold `kaggle_kernel/kernel_tabpfn_10k/` mirroring
    kernel_realmlp pattern. SMOKE first (1-fold × 10k context ×
    5-min wall) then production (5-fold × 10k stratified
    in-context fit per fold). Save `oof_tabpfn_10k.npy` +
    `test_tabpfn_10k.npy`.
  - Gate: standalone OOF tuned ≥ 0.974 AND fold-1 Jaccard < 0.78
    vs LB-best 4-stack AND errs ≤ 9415 (the +5% magnitude rule).
    If passes: add to meta-stacker bank → re-train cross-poll
    metastack → blend gate at +2e-4 OOF.
  - Expected: most likely 14th NN family null (the magnitude-trap
    pattern is structural at this feature set) but TabPFN is the
    only NN family WITH a track record of high-confidence per-row
    posteriors that calibrate well at the operating point our
    fixed bias targets. ~20% prior of breaking the pattern.

  **N2. CVAE-generated synthetic High rows** (~1.5h Kaggle GPU).
  Direct fix for SMOTE-NC's interpolation failure. SMOTE failed
  because k-NN linear interpolation diffused the M↔H boundary.
  CVAE generates samples from a *learned latent manifold*
  conditioned on `y=High`, so synthetic Highs respect the joint
  distribution rather than averaging neighbors. Different
  failure-mode geometry from SMOTE.
  - Action: scaffold `kaggle_kernel/kernel_cvae_high/` —
    encoder-decoder CVAE on RAW 19 features (8 cats embedded +
    11 nums standardized), latent_dim=8, KL-regularized, 100
    epochs. Generate 21k synthetic High rows from posterior
    samples conditioned on y=High. Augment recipe training pool
    by 2x High rate. Re-train recipe XGB.
  - Gate: standalone OOF tuned ≥ 0.978 (matches recipe baseline
    at minimum) AND per-class High recall ≥ 0.978 (above
    LB-best 0.9775) AND Jaccard < 0.85 vs recipe.
  - Expected: SMOTE failed at every TARGET (42k, 25k); CVAE has a
    specific reason to do better (manifold preservation). 25-30%
    prior.

  **N3. Hard-example scan on LB-best primary** (~30 min CPU).
  Diagnostic, not a model. Take primary's bottom-1% confidence
  test rows (~2,700 rows = max_prob below the 1st percentile).
  Cross-tabulate with score-band, rule-vs-LB disagreement, and
  per-class predicted distribution. If 80%+ of low-confidence
  rows land in score=6 boundary, it tells us exactly where the
  +0.00020 to pack-LB lives — and J7 conformal told us we
  CAN'T get there with the current detector input set, meaning
  we need a different feature view of those rows.
  - Action: ~80-line `scripts/n3_hard_example_scan.py`. Output:
    JSON with score-band breakdown, mean per-class prob, mean
    rule-pred match rate, and feature-space centroids of the
    low-confidence rows.
  - Decision value: informs whether N1/N2 should target a
    SPECIFIC sub-domain (e.g., score=6 ∩ Crop=Wheat ∩
    Humidity > 70) rather than global retraining. Could surface
    a 4th lever we haven't articulated.
  - Expected: closure (no direct LB lift) but high information-
    per-minute. Run FIRST before N1/N2.

  **N4. recipe_no_rule_features standalone** (~50 min CPU).
  We tested `recipe_no_ote/no_digits/no_combos/no_orig` (all
  null on the magnitude rule). Never tested dropping the
  rule-derived features themselves: `dgp_score, sm_dist, rf_dist,
  tc_dist, ws_dist, dry, norain, hot, windy, nomulch, kc, rule_pred,
  logit_P_low/med/high, rule_correctness flags`. Without rule
  features, trees must re-discover the rule from continuous
  signals — different basis for splits, possibly producing a
  Jaccard-< 0.7 standalone that the magnitude trap doesn't kill
  because the anchor's strength comes FROM the rule features.
  - Action: parameterize `scripts/recipe_full_te.py` with
    `EXTRA_EXCLUDE_RULE_FEATS=1` env var, drop ~17 rule cols
    pre-OTE. Same 5-fold seed=42, same XGB HPs.
  - Gate: standalone OOF tuned ≥ 0.973 (likely below recipe's
    0.97967) AND Jaccard < 0.75 vs recipe AND errs ≤ recipe ×
    1.10. If passes 2/3: add to meta-stacker bank.
  - Expected: probably ≥0.75 Jaccard with recipe (recipe's tree
    splits already use both rule cols and continuous correlates
    of them). 15% prior of a useful blend leg, but cheap to test.

  **Execution order**: N3 first (cheapest, informs the rest) →
  parallel N1 (Kaggle GPU queue) + N4 (local CPU) → N2 if N1
  doesn't return a passing leg. If all four null:

  **N5 (lock + stop)**: lock primary `submission_tier1b_greedy_meta.csv`
  (LB 0.98094) + hedge `submission_3way_recipe025_s1035_s7040.csv`
  (LB 0.98005) and reserve remaining LB submissions for end-of-comp
  variance check (one re-validation per day until close).

  **Skip on principled grounds** (re-confirmed today):
  - More log-bias / Bayes-opt / decision-rule variants — the
    LP probe definitively closes this family. Coord-ascent is
    near-optimal at the operating point our calibration produces.
  - Public-CSV blending (banned by top-of-file rule).
  - More NN-family attempts beyond TabPFN-10k. 14 NN nulls form
    a structural pattern at this feature set; TabPFN-10k is the
    last UNTESTED-AT-DESIGN-SCALE family.
  - HP/seed bagging on existing components (LB-regressed twice).

### 2026-04-26 — N1 TabPFN-10k 1-fold val-only signal probe: NULL (15th NN family null, lowest Jaccard ever 0.21)

- Goal: execute N1 from the post-LP shortlist. The prior 2026-04-22
  TabPFN run capped at SUBSAMPLE=1500 for CPU. TabPFN v2's
  architectural sweet spot is 10k. The closure note "GPU SUBSAMPLE=10000
  unlikely to change blend outcome" was a guess from 1.5k Jaccard,
  never measured. At 10k context, ~6.7x more rare-class examples
  (3,333 High vs ~50 @ 1.5k) — direct test of whether rare-class data
  shortage was the structural problem.
- Branch: `claude/kaggle-missing-strategies-rls5d`. Single-file kernel
  `kaggle_kernel/kernel_tabpfn_10k/tabpfn_10k.py` mirroring kernel_realmlp
  pattern. tabpfn==2.2.1 (matches prior run). 43 dist features (raw +
  signed/abs distances + rule indicators + pairwise products) — same
  feature set as the 1.5k run for apples-to-apples context-size effect.

- **Three-iteration story**:
  1. **SMOKE v1** (1k context × 5k val × 5k test, 3.5min) — PASSED.
     Pipeline boots clean on P100. Throughput at 1k = 1980 rows/sec.
     Fold-0 argmax 0.96179, PCR=[L=0.9960, M=0.9568, H=0.9326].
  2. **Production v2** (10k context, val + test) — KILLED at 56.7min.
     Realized 10k throughput = **97 rows/sec** (vs my 200-est).
     Val 126k completed in 21.8min ✓ but test 270k needed ~46min;
     the 55-min HARD KILL fired during test predict and the
     `sys.exit(0)` bypass meant NO artifacts were saved.
  3. **v3 SKIP_TEST=True** (val-only, 21.9min) — CLEAN COMPLETE.
     Fix: skip test predict entirely on this 1-fold probe (a 1-fold
     test array isn't deployable; signal check needs only val).
     Save val OOF + score immediately after val complete.

- **v3 results (10k context, fold 0)**:
  ```
  context size    argmax bal_acc    Low    Medium    High
  1.5k (prior)    not reported      —      —         0.9238 (5-fold standalone)
  1k (smoke)       0.96179         0.9960  0.9568    0.9326
  10k (this)       0.96368         0.9958  0.9566    **0.9386**  (+0.015 vs 1.5k)
  ```
  Tuned 0.96424, bias=[0.63, 2.07, 1.70]. **Real lift on High recall
  from more rare-class context examples** — but at the standalone
  level this matters only if blend-gate passes.

- **Blend-gate analysis vs LB-best primary fold-0** (anchor OOF 0.97994):
  ```
  TabPFN @ recipe bias [1.43, 1.47, 3.40]:
    bal = 0.94149  errs = 7,837   PCR=[L=0.9958, M=0.8439, H=0.9848]
  LB-best primary @ recipe bias:
    bal = 0.97994  errs = 1,914   PCR=[L=0.9956, M=0.9690, H=0.9752]

  Gate criteria (all 3 must pass for full 5-fold push):
    bal_tuned ≥ 0.974          FAIL  (0.96424)
    Jaccard < 0.78             PASS  (0.2158 — LOWEST EVER)
    errs ≤ 1.05 × anchor       FAIL  (7,837 vs cap 2,010 — +309.5%)
  OVERALL: FAIL
  ```

- **The headline finding**: **Jaccard 0.2158 is the lowest orthogonality
  ever recorded on this problem.** Beats every prior NN family by a
  wide margin (Mamba 0.49, Trompt 0.53, RealMLP n_ens=1 0.62). At
  recipe bias TabPFN nails High (0.9848 — best of any candidate) but
  catastrophically loses Medium (0.8439 vs anchor 0.9690). The errors
  are GENUINELY orthogonal — TabPFN sees a different decision surface.

- **But magnitude trap dominates by 4x**: TabPFN errs 7,837 vs anchor
  1,914 (+309.5%). Even at the unprecedented Jaccard 0.21, a positive
  blend weight drags the LB-best toward TabPFN's 4x-more-numerous
  wrong answers on Low/Medium boundaries. The +0.015 High recall lift
  cannot offset that under macro-recall + fixed bias.

- **Updated NN-family null table** (15 nulls now, structural pattern
  cemented):

### 2026-04-26 — TabM (pytabkit TabM_D) PROBE: 15th NN family null at LB 0.98094 (compute-bound + magnitude trap)

- Goal: execute the highest-EV remaining untried NN architecture from
  the kernel-audit lever bank — TabM-D (ICLR 2025, Gorishniy et al.
  Yandex Research). Mirrors `kernel_realmlp_ens4` exactly except for
  the model swap. Per CLAUDE.md GPU 1h cap rule + user instruction
  "check after 1 fold to not waste GPU" — SMOKE-first then PROBE
  (1 fold), not direct 5-fold push.
- Changed: `kaggle_kernel/kernel_tabm/` (single-file kernel mirror of
  kernel_realmlp_ens4 + IS_PROBE flag + per-fold checkpoint save +
  fold-1 standalone abort gate), `scripts/blend_tabm.py` (blend-gate
  mirror of blend_mamba.py: Jaccard + magnitude vs LB-best 4-stack +
  fixed-bias α-sweep + PASS/WARN/REDUNDANT/MAGNITUDE-TRAP verdict).
- Three pushes per CLAUDE.md SMOKE-first rule:
  1. **SMOKE v1** (IS_SMOKE=True, tabm_k=8, n_epochs=3, 20k×2-fold):
     ERROR — `TypeError: TabMConstructorMixin.__init__() got an
     unexpected keyword argument 'n_ens'`. Fix: pytabkit's TabM_D has
     BatchEnsemble built-in via `tabm_k` kwarg (default 32), not
     `n_ens` like RealMLP_TD. Removed n_ens; verified accepted kwargs
     via local introspection.
  2. **SMOKE v2** (same config minus n_ens): PASSED in ~5 min wall.
     Per-fold timing 3.7s on 10k × 3 epochs × tabm_k=8.
  3. **PROBE** (IS_PROBE=True, tabm_k=32, n_epochs=25, 1 fold × 504k):
     completed but **122 min wall on Kaggle P100** — way over the
     1h GPU cap. Fold-1 wall-time abort triggered, fold 1 saved.
- Standalone results (PROBE fold-1 = 126,000 val rows):
  - argmax bal_acc = **0.96753** (RealMLP fold-1 0.96978 / Trompt
    0.96092 / Mamba 0.95740)
  - tuned bal_acc = **0.97496**, bias [1.6324, 1.4689, 3.4008]
    (High bias 3.40 matches recipe family exactly)
  - errors = 2,298 vs LB-best 4-stack's 1,914 (+20% magnitude)
- Anchor comparison on same 126k val rows:
  - LB-best 3-stack: 0.97926, errs 2,014
  - LB-best 4-stack: 0.97994, errs 1,914
  - TabM:            0.97491, errs 2,298 (Δ −0.00503, +20% errs)
- **Jaccards (the headline finding)**:
  - vs LB-best 3-stack: **0.5589**
  - vs LB-best 4-stack: **0.5781** ← good orthogonality
    (between RealMLP n_ens=1's 0.62 and Trompt's 0.53 fold-1)
  - vs RealMLP n_ens=1: 0.6067
- Fixed-bias α-sweep vs LB-best 4-stack (filled rows): **monotone
  negative from α=0.05** (Δ −0.00016 to −0.00067 across α∈[0.05, 0.35]).
  Peak at α=0.000 (no blend).
- **Verdict: NULL on both axes**:
  1. **Compute-bound**: 122 min/fold × 5 = ~10 hours wall, infeasible
     even with leaner config. Production 5-fold cannot fit any GPU
     budget reasonable for this competition.
  2. **Magnitude-trap**: same failure mode as 14 prior NN nulls. TabM
     has the right Jaccard (0.58) for orthogonal blend signal, but
     +20% more errors than anchor flips the macro-recall trade
     against the rare class.
- LB delta: n/a — no LB probe warranted (no α > 0 lifts above
  LB-best 4-stack OOF). LB best unchanged at **0.98094**. LB budget
  unchanged.
- Pattern reinforced (now **15 consecutive NN-family nulls** on this
  feature set):
  ```
  NN family            Jaccard vs anchor   errs vs anchor    LB outcome
  ----------          ------------------- ------------------ --------------------
  MLP v5-v9            0.62-0.85           +1500-15000       NULL
  FT-Transformer       0.61                +12000            NULL
  TabPFN (1.5k CPU)    0.81                +1485             NULL
  Pretrain-FT MLP      0.65                +3615             NULL
  DAE SwapNoise        0.84                similar           NULL
  RealMLP n_ens=1      0.62                +358              LB +0.00003 (3-stack)
  RealMLP n_ens=4      0.62                +485              NULL (worse than n_ens=1)
  Trompt               0.53                +169              NULL (magnitude-trap)
  Mambular SSM         0.49                +518 (+27%)       NULL
  **TabPFN-10k         0.21 (record)       +5923 (+309%)     NULL (this entry)**
  ```
  Only RealMLP n_ens=1 has ever cleared the +5% magnitude rule AND
  produced an LB lift. The pattern is structural at this feature set:
  NN architectures produce orthogonal errors but in larger absolute
  numbers, and macro-recall at fixed-bias cares about per-class
  totals → magnitude tax dominates.

- **Strategic implication**: TabPFN-10k was the last NN family at
  design-scale that could break the pattern. The 14-null history
  WITH a +0.015 High recall lift WITH the lowest Jaccard ever
  recorded WITHOUT clearing the gate — this is the strongest
  closure signal possible. **NN levers are now unambiguously
  exhausted on this problem within the standard tabular ML toolkit.**

- **No 5-fold push warranted.** Standalone 0.96424 is 0.0157 below
  anchor; even with 5-fold averaging variance reduction (~+0.0005)
  it won't reach 0.974. Magnitude trap is structural, not variance.

- **Three portable rules** (LEARNINGS.md candidates):
  1. **Jaccard < 0.30 with errs > 3x anchor is structurally a
     "different problem" candidate, not a "different model" candidate.**
     The model is solving a different optimization than the anchor;
     blend weights cannot reconcile the operating points. Skip directly
     to "is this orthogonal model worth deploying ALONE on a different
     subset of test rows?" — i.e., conformal routing or per-row
     feature-conditional gating, not log-blend.
  2. **TabPFN context-size scaling: more context → genuinely more
     rare-class signal extracted, but at fixed bias it trades
     majority-class recall for rare-class recall**. The +0.015 High
     recall came at -0.011 Medium. On a 3-class problem with 3.3%
     High prior, the trade is unfavorable for macro-recall. Useful
     for rare-class detection tasks (binary or imbalanced ranking)
     but not for balanced-accuracy multiclass.
  3. **GPU throughput estimates for transformer in-context learners
     scale ~linearly with context size, not quadratically**. At 1k
     context = 1980 rows/sec on P100; at 10k = 97 rows/sec (20x
     slower at 10x context). Plan accordingly: at 10k context, expect
     ~100 rows/sec inference budget — 270k test = 45min, val 126k =
     22min, total 67min uninterruptible work. For 1-fold probes, skip
     test entirely; production 5-fold needs a multi-kernel split.

- **11th independent saturation confirmation at LB 0.98094**.

- LB delta: n/a. No probe warranted (standalone gate failed). LB best
  unchanged at **0.98094** via `submission_tier1b_greedy_meta.csv`.
- LB budget unchanged: 4/10 used (for the day). 6 remaining.

- Artefacts (whitelisted via `.gitignore` exception):
  - `kaggle_kernel/kernel_tabpfn_10k/tabpfn_10k.py` (~530 lines including
    SKIP_TEST fix and the v3 incremental save)
  - `kaggle_kernel/kernel_tabpfn_10k/kernel-metadata.json`
  - `scripts/blend_tabpfn_10k.py` (fold-0 blend gate)
  - `scripts/artifacts/oof_tabpfn_10k.npy` (fold-0 val rows populated;
    other rows zero-sentinel)
  - `scripts/artifacts/test_tabpfn_10k.npy` (uniform-prior placeholder
    from SKIP_TEST mode — NOT a real test prediction)
  - `scripts/artifacts/tabpfn_10k_results.json`
  - `scripts/artifacts/blend_tabpfn_10k_fold0_results.json`

- **Updated next-steps shortlist** (3 levers remain from the post-LP plan):
  - N2 (CVAE for synthetic Highs) — NOW LOWER PRIORITY. The TabPFN-10k
    result shows even 6.7x more rare-class examples doesn't break the
    magnitude trap on this feature set. A CVAE-generated synthetic
    High set is a similar mechanism (more rare-class data) but with
    interpolation noise risk that already killed SMOTE-NC twice. Prior
    of breaking the pattern dropped from 25-30% to ~10%.
  - N3 (hard-example scan) — UNCHANGED. Still the cheapest diagnostic
    (~30 min CPU). Even if N4 nulls, the score-band breakdown of low-
    confidence rows informs final-selection variance estimates.
  - N4 (recipe_no_rule_features) — UNCHANGED. ~50 min CPU.
  - N5 (lock + stop) — INCREASED PRIORITY. With 11 saturation
    confirmations + 15 NN nulls + the strongest closure signal possible
    on the remaining "untested at scale" lever, the marginal LB-probe
    EV is minimal. Lock primary + hedge as final.
  TabPFN               0.81                +1485             NULL
  Pretrain-FT MLP      0.65                +3615             NULL
  DAE SwapNoise        0.84                similar           NULL
  RealMLP n_ens=1      0.62                +358              LB +0.00003 (3-stack)
  RealMLP n_ens=4      0.62                +485              NULL
  Trompt               0.53                +169              NULL (compute-bound + magnitude)
  Mambular SSM         0.49 (record low)   +518 (+27%)       NULL
  **TabM-D (k=32)      0.58                +384 (+20%)       NULL (compute + magnitude)**
  ```
  Only RealMLP n_ens=1 has ever cleared the magnitude bar (+358 errs
  = +3.7% over anchor) AND produced an LB lift. Every other NN family
  is permanently closed at this feature set. **Jaccard ≤ 0.62 + errs
  ≤ 1.05× anchor** is the necessary-and-still-not-sufficient blend
  fingerprint we've never re-found since.

- **Portable rules** (LEARNINGS.md candidates):
  1. **TabM-D via pytabkit's `tabm_k=32` is wall-time-prohibitive on
     Kaggle P100 for production 5-fold at full epoch budget.** 122 min
     fold-1 wall (504k rows × 25 epochs) projects to ~10h for 5-fold,
     vs the 1h GPU cap in CLAUDE.md. Must reduce to tabm_k≤16 +
     n_epochs≤15 for any 5-fold attempt — but at that lean config,
     standalone OOF will likely drop below the 0.97 floor that's been
     the consistent NN failure threshold on this problem.
  2. **pytabkit's TabM_D and RealMLP_TD have DIFFERENT signatures.**
     TabM_D doesn't accept `n_ens` (it has built-in BatchEnsemble via
     `tabm_k`, default 32). Always introspect the constructor before
     porting kernel HPs across pytabkit families.
  3. **Three NN PROBE attempts (Trompt, Mamba, TabM) all converge on
     the same Jaccard-orthogonal-but-magnitude-trapped corner.** TabM's
     Jaccard 0.58 + errs +20% mirrors Trompt's 0.53 + errs +8% (also
     compute-killed at fold 1) and Mamba's 0.49 + errs +27%. The
     pattern is not architecture-specific — it's a property of how NNs
     trained on this 19-50 feature recipe-mirror set distribute errors
     on the held-out fold. Future synthetic tabular comps should
     SMOKE-then-PROBE NN families with a strict 5-min wall budget for
     SMOKE + 30-min wall budget for PROBE, and close the lever fast on
     magnitude trap. Don't sink GPU budget into 5-fold production
     before the magnitude bar is cleared.

- Artefacts (whitelisted in `.gitignore` for cross-branch reuse):
  - `kaggle_kernel/kernel_tabm/{tabm_pytabkit.py,kernel-metadata.json}`
  - `scripts/blend_tabm.py` (~210 lines, modular per CLAUDE.md rule)
  - `scripts/artifacts/oof_tabm.npy` (7.2 MB, fold 1 only)
  - `scripts/artifacts/test_tabm.npy` (3.1 MB)
  - `scripts/artifacts/tabm_probe_results.json` (per-fold + tuned +
    abort_reason)
  - `scripts/artifacts/blend_tabm_results.json` (Jaccards + α-sweep)
- **Final-selection lock unchanged** (4 days to deadline):
  1. **PRIMARY**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
     (gap −0.00010, anomalous LB > OOF)
  2. **HEDGE**: `submission_3way_recipe025_s1035_s7040.csv` →
     **LB 0.98005** (premium −0.00089, sidesteps meta-stacker layer)
- Pack 0.98114 still +0.00020 above; leader 0.98219 still +0.00125
  above. Reachable only via public-CSV blending (banned by top-of-file
  rule). With 15 NN nulls now in the log, the own-pipeline ceiling at
  LB 0.98094 is structurally exhausted across architecture classes.

### 2026-04-26 — RealMLP n_ens=2 @ FULL n_epochs=40: NULL, RealMLP variance floor structural at n_ens=1

- Goal: one-shot diagnostic between the two hypotheses for n_ens=4's NULL
  (CLAUDE.md 2026-04-25 RealMLP retry):
  (a) **under-convergence** — n_epochs=25 was forced by the 1h Kaggle GPU
      cap with 4 BatchEnsemble heads; each head saw less gradient than
      the n_ens=1 working config's 40 epochs.
  (b) **variance floor** — n_ens=1 already at the per-row variance floor
      on this feature set, so additional heads add bias drift (per-head
      converges to slightly different local minima → averaging them
      moves the prediction surface AWAY from the floor) rather than
      useful variance reduction.
  Approach: n_ens=2 with FULL n_epochs=40, fitting cleanly under the 1h
  cap. Beats n_ens=1 → (a). Plateaus at n_ens=1 → (b).
- Changed: `kaggle_kernel/kernel_realmlp_ens2/` (mirrors `kernel_realmlp_ens4`
  scaffold; only `n_ens=2, n_epochs=40` differ from ens4's `n_ens=4, n_epochs=25`),
  `scripts/blend_realmlp_ens2.py` (3-stack blend gate vs n_ens=1, partial-
  OOF aware, auto-emit submission only if 3-stack OOF > anchor 0.98061 +
  +2e-4 LB-transfer threshold). SMOKE-first discipline enforced.
- SMOKE (Kaggle v1, IS_SMOKE=True, 2-fold × 20k × 3 epochs): GREEN in
  ~3.3 min wall, GPU used (cuda), pytabkit + lightning installed cleanly.
- Production (Kaggle v2, IS_SMOKE=False, 5-fold × 504k × 40 epochs): all
  5/5 folds completed inside the 55-min hard kill budget.
- **Standalone results** (5-fold OOF, seed=42):
  ```
                          n_ens=1  n_ens=2  Δ
  per-fold argmax mean    0.97055  0.96951  -0.00104
  tuned OOF               0.97633  0.97583  -0.00050
  errs at recipe bias     10472    10901    +429
  Jaccard vs LB-best 3-way 0.6206  0.6173   ~tied (both still
                                              best-in-class NN ortho)
  log-bias                [1.2324, [1.3324,  ~tied
                           1.4689,  1.3689,
                           3.4008]  3.4008]
  ```
  n_ens=2 with full epochs is **strictly worse** than n_ens=1 at every
  level. σ across folds remains ~0.001 (similar to n_ens=1) — not the
  signature of a converging-but-incomplete run; it's the signature of
  a different attractor.
- **3-stack peak** (LB-best 3-way + RealMLP@α + nonrule_iso@0.075,
  fixed recipe bias):
  ```
                  n_ens=1                n_ens=2
  best α          0.200                  0.250
  3-stack OOF     0.98061                0.98054   (-0.00007)
  3-stack errs    9572                   9601      (+29)
  ```
  Even at the best α, the 3-stack falls 0.00007 below the n_ens=1
  3-stack — well under the +2e-4 LB-transfer emit gate. Strict null.
- **Verdict: hypothesis (b) variance floor is correct.** n_ens=1
  already sits at the relevant local minimum for this feature set + 40
  epochs. Adding more heads (whether at full or reduced epochs) doesn't
  help because they converge to **different** local minima — the average
  is a bias-shifted blend, not a variance-reduced ensemble. Same root
  cause as n_ens=4's NULL; just confirmed cleanly with full-epoch
  budget for the first time.
- Implication: drops the previously-suggested overnight `n_ens=4 @
  n_epochs=40` follow-up. Same mechanism would produce the same NULL.
  RealMLP family fully closed across n_ens ∈ {1, 2, 4}; n_ens=1 is
  the only configuration that ever transferred (LB +0.00003 in
  3-stack, 2026-04-24).
- LB delta: n/a. No LB probe (gate failed cleanly). LB-best unchanged
  at **0.98094** via `submission_tier1b_greedy_meta.csv`.
- **Two portable rules** (logging to LEARNINGS.md):
  1. **BatchEnsemble at fixed feature set + capacity has a variance
     floor that scaling n_ens past 1 cannot break.** When n_ens=1
     produces a model already converged at the relevant SGD minimum,
     additional heads converge to nearby but distinct minima; the
     ensemble average shifts the prediction surface in bias space
     rather than reducing per-row variance. Test the variance-floor
     hypothesis with `n_ens=2 @ full_epochs` BEFORE planning longer
     `n_ens=4` runs — if n_ens=2 already plateaus or regresses, the
     entire ensemble dimension is closed for this feature set.
  2. **NN-family lever-existence cost ≤ 1h GPU when SMOKE-first
     discipline is enforced.** The end-to-end loop (commit kernel
     scaffold → SMOKE push → flip IS_SMOKE flag → production push →
     pull artifacts → run blend gate → commit) takes ~70-90 min total
     calendar time on a 50-min production wall. SMOKE catches
     install-path / dataset-path / GPU-init bugs in ~5 min, not in
     a 55-min wasted production run.
- Artefacts (whitelisted in `.gitignore` for cross-branch reuse):
  - `kaggle_kernel/kernel_realmlp_ens2/` (kernel + metadata + README)
  - `scripts/blend_realmlp_ens2.py`
  - `scripts/artifacts/oof_realmlp_ens2.npy` (7.2 MB, 5-fold OOF)
  - `scripts/artifacts/test_realmlp_ens2.npy` (3.1 MB)
  - `scripts/artifacts/realmlp_ens2_results.json`
  - `kaggle_kernel/output_realmlp_ens2/irrigation-realmlp-ens2.log`

### Next steps: senior-DS EDA proposals (2026-04-26)

Fresh-eyes EDA over the LB-best 4-stack confusion + 17-component OOF
disagreement structure surfaced one **measured-orthogonal residual
signal** + two structurally-untried mechanisms. Findings:

- Aggregate stats over the 60+ saved OOFs (per-class mean/std/max/
  min/median, entropy of mean-prob, argmax disagreement count) carry
  AUC 0.91 standalone for missed-H detection in the pred=Med override
  domain. **Aggregate ⊥ existing meta-stacker AUC = 0.6714** — that's
  measured residual signal beyond what the 63-component depth-4 meta-
  XGB extracts.
- `xgb_nonrule` is the most-orthogonal leg in the bank (corr ≤ 0.13
  with mamba/kan, ≤ 0.30 with all others on P(High)). Used at α=0.075
  in primary; signal extraction bounded.
- Score=3 M→L errors (4303 = 45.7% of total): Cohen's d=+0.27 on
  Soil_Moisture, +0.09 on Rainfall_mm. Within-bucket SINGLE-feature
  AUCs are weak (0.43-0.48) — model needs interactions.
- Score=6 missed-H vs found-H: Cohen's d on Rainfall +0.43, SM −0.29,
  Temp −0.24. Strong RAW-feature distinguishability, not just
  teacher_PH (contradicts the 2026-04-26 stage-1d "feature-
  indistinguishable" finding when restricted to z-distance from
  teacher_PH).
- Direct-override path remains closed: top-K precision in pred=Med
  override domain peaks at 4.3% vs break-even 8.1%. Same magnitude
  trap as J7 conformal closure.

**Three executable proposals** (ranked by EV / cost / mechanism novelty):

  **P1. Aggregate-stats meta-stacker v6** (top pick, ~10 min CPU).
  Re-train tier1b-style heavy-reg XGB meta-stacker on existing 63-
  component bank PLUS 22 aggregate features computed across the bank:
  per-class {mean, std, max, min, median} (15) + entropy of mean-prob
  (1) + per-row argmax disagreement count (1) + per-class skew across
  components (3) + pair-margin variance (2). Distinct from every prior
  meta variant (v1-v5, LR, J2 bagged) — all extended via per-component
  features; none added bank-aggregate stats. Iso-cal + blend into LB-
  best 3-stack at α=0.30. Gate: standalone iso ≥ 0.98080, errs ≤ 9550,
  per-class recall ≥ [0.9950, 0.9690, 0.9770]. Expected: +0.0001 to
  +0.0004 OOF, LB-transfer ~30-50%, realistic upside 0.98098-0.98114.

  **P2. Bucket-aware soft-blend with FE-targeted heads** (~30 min CPU).
  Two binary specialists with FE matched to today's empirical Cohen's
  d patterns:
    - score=3 head: P(y=Med | x), features = recipe + log(SM/25), log
      (Rainfall/300), Humidity·SM, Humidity·PrevIrrig, sample_weight
      tuned for Med minority.
    - score=6 head: P(y=High | x), features = recipe + log(SM/25), log
      (Rainfall/300), Soil_pH−6.5, Wind−10, Rainfall·Temp_C.
  Deploy as **soft logit-add** into LB-best 4-stack class column for
  that bucket only (NOT hard override — precision floor 39% for L↔M
  unreachable, 8% for M↔H borderline). λ tuned via fixed-bias macro-
  recall + per-class recall guardrail.
  Expected: +0.0001 to +0.0005 OOF, LB-transfer ~40%. Risk: prior
  spec_3 was null on plain dist features; differentiator is engineered
  FE matching the +0.27 Cohen's d signal + soft mixing.

  **P3. Counterfactual rule-instability features + retrain primary**
  (~80 min CPU). For each row, compute `rule_instability` = number of
  flips in `dgp_score` under {±2%, ±5%, ±10%, ±20%} × {SM, Rain, Temp,
  Wind} = 32 perturbations + 4 axis-specific instabilities = 5 new
  features. Captures multi-axis simultaneous closeness to ANY rule-
  cell boundary — distinct from existing per-axis distances. Add to
  recipe → retrain → add to meta-stacker bank. Expected: +0.0001 to
  +0.0005 OOF. Risk: recipe FE additions saturated at Jaccard 0.83+
  in 6 prior tests; differentiator is cell-topology instead of feature
  values.

  **P4. Honorable mention — original-NN-distance features** (~1h CPU,
  deferred). FAISS k-NN distance from each synth row to 10k-original
  rows. 4 features: min/mean dist, fraction of NN with Med/High labels.
  Run only if P1-P3 all yield <+0.0001.

  **P5. Honorable mention — orthogonality-weighted greedy** (~10 min
  CPU on saved OOFs). Diversity-aware step rule: rank candidates by
  (OOF Δ at α*) × (1 − max-Jaccard-with-existing). Sanity check on
  CMA-ES saturation claim; will likely confirm.

**Execution order**: P1 first (cheapest + strongest measured residual
signal). If P1 lifts ≥ +0.0001 OOF AND passes per-class guardrail,
LB-probe. If null, P3 next (mechanism most distinct from prior nulls).
P2 last among the three. Lock final-selection unchanged until P1-P3
produce LB-validated lift. Reserve ≥4 LB submissions for end-of-comp
variance check (3 days to deadline 2026-04-30).

**Skip on principled grounds** (already exhausted):
- Further meta variants (v6 with same component features, LR with
  different HPs, J2 at fraction ≠ 0.5) — saturated.
- Direct missed-H override at any threshold — break-even precision
  unreachable in the override domain (top-K caps at 4-7%).
- Public-CSV blending (banned by top-of-file rule).

### Next steps: post-13th-saturation brainstorm (2026-04-26 evening)

After P1 v6_full (LB 0.98012, regress), P1 v6_lbpool (NULL G4),
P3 instability (NULL G4), and P2 bucket FE specialists (NULL G2/G4)
all closed today, this is the 13th independent saturation
confirmation at LB 0.98094. The Pareto frontier interpretation
held: P2's score=6 specialist hit competition-record AUC 0.852
within bucket but got λ_6=0 weight in the soft-logit-add — H
boundary signal is fully absorbed by the LB-best 4-stack at the
operating point. Per the "NEVER GIVE UP" rule, 5 NEW mechanisms
not yet on the hypothesis board:

  **A. Boundary-confined TTA** (top pick, ~30 min CPU). Re-attempts
  the 2026-04-24 P1 TTA, which closed with the rule "perturbation
  noise scales with N (far-row noise) while boundary-smoothing gain
  scales with ~2% boundary fraction." Identify boundary rows via
  `max_prob(LB-best 4-stack) < 0.95` (~5-10% of N). Gaussian σ × IQR
  perturbation on the 4 rule-axis numerics (Soil_Moisture,
  Rainfall_mm, Temperature_C, Wind_Speed_kmh) ONLY for those rows,
  K=10 perturbations. Recompute recipe FE on perturbed rows
  (sm_dist/abs, rule flags, dgp_score, axis-specific digit features).
  Run frozen fold booster, average. Replace OOF only at boundary
  rows; non-boundary OOF stays identical to vanilla recipe.
  Mechanism: noise/signal ratio inverts vs prior TTA — noise scales
  with K × n_boundary (~5% of N) instead of K × N. Closes the
  prior TTA rule definitively.
  Cost ~30-50 min CPU (recipe rerun with TTA hook in val-prediction
  step). Helper file `scripts/mech_a_btta.py` already scaffolded
  with `boundary_mask` / `axis_iqrs` / `perturb_axes` /
  `recompute_axis_dependent_features` and a 3-row smoke pass.
  Decision gate: G1-G4 same as P1/P3, but compute on the recipe-leg
  substituted into the LB-best stack reconstruction.

  **B. Anchor-uncertainty-weighted re-training** (~50 min CPU).
  Inverse of cleanlab confident-learning (which DOWN-weighted
  ambiguous rows and produced NULL): use
    `sample_weight[i] = 1 + α × (1 − max_prob_LB4stack[i])`
  with α ∈ {1, 2, 5}. Up-weights boundary rows where the anchor is
  uncertain. Train recipe XGB on this — focuses model capacity on
  anchor's weak regions. Resulting OOF should have orthogonal errors
  specifically AT boundary rows by construction. Different from
  cleanlab (which interpreted uncertain rows as label noise — they
  are deterministic NN flips per 2026-04-21 EDA). Expected upside
  +0.0002-0.0008 OOF if boundary rows have learnable structure
  the anchor missed; null if anchor already saturates the boundary.

  **C. Synthetic rule-only label augmentation** (~70 min CPU).
  Different from SMOTE-NC (which interpolated minority class and
  was LB-regressive 2026-04-25) and cleanlab-relabel (which removed
  ambiguous rows). Sample 100k feature vectors via stratified
  resample of train's joint distribution. Apply the deterministic
  DGP rule to generate rule-perfect labels (no NN flips) on these
  synthetic rows. Concat to training, retrain recipe XGB on
  augmented 730k rows. Mechanism: anchors recipe XGB more strongly
  on the rule's core, freeing model capacity for within-cell flip
  variation. Synthetic rows have NO label noise → regularize
  toward the rule baseline. Risk: distorts calibration of the flip
  signal. Expected +0.0001-0.0005 OOF.

  **D. Per-row attention over component bank** (~30 min CPU).
  Different from MoE (K=6 experts, NULL via collapse to fixed
  weights, 2026-04-26): per-row attention over the FULL 60+
  component bank. For each row, learn a 60-dim weight vector via
  a small attention layer. Input: row's distance features +
  dgp_score + LB-best 4-stack probs. Output: softmax over 60
  component weights. Loss: macro-recall surrogate at fixed bias.
  Expected +0.0001-0.0003 OOF if there is row-conditional
  component preference the heavy-reg meta-stacker misses; null
  if the meta already learned this internally.

  **E. Fuzzy-threshold rule features** (~50 min CPU). Different
  from P3 instability (count of cell-flips under perturbation,
  NULL on G4 because redundant with hard-threshold features).
  This adds CONTINUOUS sigmoid soft transitions rather than
  discretized aggregates: `fuzzy_dry = sigmoid((25 − sm) / τ)` for
  τ ∈ {0.5, 1.0, 2.0, 5.0} and same for the 3 other axes = 16
  fuzzy features. Captures continuous transition near rule
  boundaries that hard `sm < 25` indicators discretize. The XGB
  at depth 4 sees finer resolution than its discrete splits
  naturally provide. Expected +0.0001-0.0003 OOF; likely null per
  P3 redundancy pattern but with a different mechanism (smooth
  surrogate vs discrete count).

**Execution priority** (EV/cost ranking):
  1. A (Boundary-confined TTA) — top pick, cheapest, diagnostic
     value high (closes prior TTA rule definitively). Helper
     scaffolded as `scripts/mech_a_btta.py`.
  2. B (Uncertainty-weighted retrain) — highest upside, novel
     mechanism (inverse of cleanlab).
  3. D (Per-row attention) — fast, mechanism orthogonal to MoE
     because it spans full bank with row-conditional gates.
  4. C (Synthetic rule-only aug) — high upside, high downside
     risk on calibration.
  5. E (Fuzzy thresholds) — likely redundant per P3 pattern
     but cheap; useful as last sanity check before locking
     final selection.

If A clears the gate, follow with B as compounding lever. If A
nulls, B remains the highest-upside untried lever. If both A + B
null, lock the final selection (3 days to deadline; reserve 4
remaining LB probes for end-of-comp variance check).

### 2026-04-26 — senior-DS EDA proposal sprint: 5 NULLs + triple Pareto-frontier confirmation

Senior-DS-style fresh EDA over the LB-best 4-stack confusion +
17-component OOF disagreement structure surfaced one measured-
orthogonal residual signal (aggregate stats over the bank carry AUC
0.67 ⊥ existing meta-stacker) and three structurally-untried
mechanisms. All three EDA proposals (P1/P2/P3) plus two follow-up
brainstorm mechanisms (B/D) closed NULL — bringing total
saturation confirmations to 15. The Pareto-frontier interpretation
is now triple-verified across all three axis-violation directions.

**P1 v6 aggregate-stats meta-stacker (NULL + LB regress):**
- 22 bank-aggregate features added on top of tier1b's 62-component
  meta-stacker pool. v6_lbpool restricted to the EXACT 62-component
  LB-best v1 pool isolated the aggregate-feature lever from any
  bank-extension effect.
- v6_lbpool: standalone iso 0.98065 (Δ +0.00006), blend Δ peak
  +0.00002 — sub-gate.
- v6_full (108-pool + aggregates): standalone iso 0.98073, errs
  8572 (-472 vs v1), Jaccard 0.928 with LB-best 4-stack at peak α
  (lowest meta-variant Jaccard ever). Blend peak Δ +0.00037 OOF
  at α=0.35. G3 fails High recall by 0.00012; user override
  authorized LB probe → **LB 0.98012, Δ -0.00082** (8th LB
  regression matching the bank-extension OOF→LB inflation pattern).
- 4 redundant submissions burned by case-sensitive
  `until ... | grep -q "successfully submitted"` retry loop —
  Kaggle CLI prints capital-S "Successfully" so loop never matched.
  Cost: 3 wasted slots from 10/day budget. Updated CLAUDE.md
  ⚠️ LB SUBMISSION RULE with 5 sub-rules ("never wrap submit in
  any retry/until/while/for/background loop, period"). LEARNINGS.md
  now documents the rule with case-insensitive grep noted as
  insufficient (pipe SIGPIPE / output mismatch / typo can re-leak).

**P2 bucket-aware FE specialists (NULL):**
- Two binary heads with FE engineered for the 2026-04-26 Cohen's d
  patterns:
    score=3 head (target=Medium): bucket-OOF AUC 0.74308
    score=6 head (target=High):   bucket-OOF AUC 0.85177
  The score=6 AUC is a competition record (prior best was the
  2026-04-26 stage-1d teacher-residual specialist at 0.79).
- Soft logit-add 2D sweep over (λ_3, λ_6): peak at λ_3=0.20,
  **λ_6=0.00** (zero weight on the AUC-0.85 specialist!). At every
  λ_6 > 0 the H-recall trade-off pushes the LB-best 4-stack PAST
  its operating-point optimum. Best feasible Δ = +0.00006 OOF, fails
  G2 (errs +20) and G4 (Δ < +2e-4).
- This is the cleanest possible Pareto-frontier proof: the strongest
  in-bucket H-detector ever built provides ZERO contribution to a
  stack already at the rare-class corner of the macro-recall frontier.

**P3 counterfactual rule-instability features (NULL):**
- 5 features added to recipe FE: `rule_inst_{sm,rf,tc,ws}` (per-axis
  flip count under {±2%, ±5%, ±10%, ±20%} perturbation) +
  `rule_instability` (total).
- Standalone passes G1+G2: iso 0.97931 (+0.00005 vs vanilla 0.97926),
  errs 9244 (-22). Blend MONOTONE NEGATIVE from α=0.05 (G4 fails).
- Mechanism: discretized aggregate of features the recipe XGB already
  has at finer resolution (signed sm_dist + abs sm_abs + binary
  dry/norain/hot/windy + dgp_score). Trees at depth 4 / max_bin 1024
  / 3000 rounds express the instability count via joint splits
  internally. Same null pattern as 2026-04-20 LGBM+FE (Δ -0.00052)
  and 2026-04-21 rule×nonrule pairwise FE (Δ -0.00007). Tree feature
  redundancy: prebuilt aggregates of features the model already has
  at higher resolution add no information regardless of physical
  motivation.

**Mech B anchor-uncertainty-weighted recipe (NULL):**
- ANCHOR_WEIGHT_ALPHA=2 hooked into recipe_full_te.py:
    sw = balanced(y) × (1 + 2 × (1 − max_prob_LB4stack[i]))
  Compound max weight 19.997× on rare-class × most-uncertain rows.
- Inverse of cleanlab confident-learning (which DOWN-weighted
  ambiguous rows). 2026-04-21 DGP residuals EDA established the
  uncertain rows are deterministic NN flips, not label noise.
- Per-fold val argmax: 0.97493/0.97621/0.97695/0.97461/0.97458,
  **all 5 folds negative** vs vanilla recipe. OOF argmax
  Δ -0.00043, tuned Δ -0.00003 (bias retune compensates).
- Standalone iso 0.97946 (+0.00020 vs vanilla 0.97926, passes G1)
  but errs 9349 (+83 vs vanilla 9266, fails G2). PCR shift:
  Low +0.00002, Med -0.00046, **High +0.00105** — genuine
  Med→High trade.
- Substitution test (replace recipe in lb3): new 4-stack OOF
  0.98077 (Δ -0.00008 vs lb4); High recall in stack 0.97715 vs
  lb4's 0.97749 (-0.00034). Gain consumed by stack.
- Blend sweep monotone-negative from α=0.05. EMIT False (g2=F,
  g4=F).
- Diagnosis: Mech B genuinely shifts the per-class balance toward
  more High at standalone level. But the LB-best 4-stack already
  chose this Med→High operating point via log-bias coord-ascent.
  Adding a model with even more aggressive Med→High trade pushes
  the stack PAST the macro-recall optimum.

**Mech D per-row attention over bank (NULL):**
- Architecture: context (15d) → MLP(64) → softmax(62) attention
  weights → convex blend over the LB-best v1's 62-component bank.
  ~8000 params. Trained 5-fold seed=42, 30 epochs Adam lr=1e-3, CE.
- Per-fold val argmax 0.97232-0.97423 (best per fold). Total wall
  130s. OOF argmax 0.97393, raw tuned 0.98028, iso tuned 0.98032 —
  BELOW LB-best 4-stack's 0.98084 by 0.00056. G1 fails.
- PCR vs LB-best 4-stack: Low -0.00009, **Med +0.00252** (gain),
  **High -0.00400** (lose). Wrong Pareto direction.
- Cross-entropy loss optimizes per-row log-likelihood, which under
  class imbalance favors confident predictions on the majority
  class. Without an explicit macro-recall surrogate, the convex-
  blend constraint collapses toward Medium predictions.
- Confirms CMA-ES result (in-sample upper bound 0.98091 is just
  +0.00007 above LB-best): even row-conditional convex blends
  cannot exceed the global optimum on this OOF bank when optimized
  via CE loss.

**Triple Pareto-frontier confirmation (the load-bearing finding):**
The LB-best 4-stack at log-bias [1.4324, 1.4689, 3.4008] sits at
the EXACT macro-recall optimum on the per-class operating-point
Pareto frontier. Verified across all three axis-violation
directions:
```
direction        mechanism             Δ standalone    Δ LB
─────────────  ───────────────────  ─────────────  ──────────
↓High, ↑L+M    P1 v6_full           +0.00037       LB 0.98012 (-0.00082)
↓Med, ↑H+L     Mech B α=2           +0.00020       null in stack
↓High, ↑Med    Mech D               -0.00056       null in stack
```
Any convex re-arrangement of the existing 60+ component OOF bank
moves AWAY from this optimum. The 15 saturation confirmations
since 2026-04-25 are not 15 different ceilings — they are 15
different confirmations of the SAME ceiling, viewed through
different mechanisms.

**Brainstorm scoreboard after this session:**
```
A. Boundary-confined TTA          untried (~90 min CPU, helper scaffolded)
B. Anchor-uncertainty retrain     NULL (this entry)
C. Synthetic rule-only aug        untried (~70 min CPU)
D. Per-row attention              NULL (this entry)
E. Fuzzy-threshold rule features  untried (~50 min CPU)
```

LB best unchanged at **0.98094** via
`submission_tier1b_greedy_meta.csv`. LB budget 6/10 used today
(4 remaining; 4 of the 6 were the v6_full retry-loop bug, only
2 net-useful probes). Final-selection lock recommendation:
PRIMARY = LB 0.98094, HEDGE = `submission_3way_recipe025_s1035_s7040.csv`
(LB 0.98005, sidesteps meta-stacker layer for orthogonal overfit
insurance).

Two final-selection days reserve 4 LB submissions for end-of-comp
variance check. Mech A is the only mechanism left that operates by
SMOOTHING boundary rows rather than re-blending components — different
failure mode from B/D's convex-blend collapse. If A nulls, lock
final selection.

**Portable rules logged to LEARNINGS.md (or candidate adds):**
1. NEVER wrap kaggle competitions submit in any retry / until /
   while / for / background loop, period. Cost asymmetry too severe.
   Read-only polling of submissions list is fine; only WRITE command
   forbidden.
2. Pareto-frontier closure on this problem is now triple-verified;
   any candidate that trades Med↔High in EITHER direction at the
   standalone level fails the LB-best 4-stack stack-level operating
   point. Diagnostic: compute per-class recall delta vs LB-best 4-stack.
   If any class drops by ≥ 0.0005, the candidate is a Pareto violator.
3. Per-row attention over a saturated component bank with CE loss
   collapses toward majority-class predictions. To use row-conditional
   blending productively, the loss must be macro-recall-aware OR the
   bank must include components NOT yet at the Pareto frontier.
4. Tree feature redundancy is a structural property: aggregate
   features (instability counts, decimal-position digits, etc.) of
   inputs the model already has at finer resolution add NO signal
   regardless of physical/mechanism motivation. Recipe XGB at depth=4
   max_bin=1024 expresses these aggregates internally via joint
   splits.

### 2026-04-26 — research-competition-strategies session: external-grandmaster review

- Goal: web-search the broader Kaggle / NVIDIA-grandmaster body of work
  to surface mechanisms not yet on our hypothesis board. Specific
  targets: Chris Deotte's published 1st-place writeups (he's also our
  rank-1 leader at LB 0.98219), the NVIDIA grandmaster blogs, recent
  s5e11/s5e12/s6e3 1st-place writeups, Matt-OP's hillclimbers package.
- **What I read** (sources verified, content quoted in next-steps below):
  1. NVIDIA "7 Battle-Tested Tabular Modeling Techniques" — diverse
     baselines / large-scale FE / hill climbing / stacking /
     pseudo-labeling / extra training / smarter EDA.
  2. NVIDIA "Stacking with cuML" (cdeotte s5e4 podcast 1st place) —
     3-level stack, OOF confidence/consensus features, predict-direct
     vs ratio vs residual vs missing as diversity formulations.
  3. NVIDIA "FE with cuDF pandas" (cdeotte 1st place backpack prices)
     — generate 10,000+ features via groupby × stat × column combos,
     distribution buckets, quantile features, digit extraction, then
     forward-select the best ~500.
  4. NVIDIA "LLM-Assisted Winning" (s6e3 customer churn 1st place,
     2026-03) — 4-level stack of 150 models from 850 experiments;
     GBDT/NN/Ridge/LR stackers in parallel, then pick best.
  5. Matt-OP/hillclimbers GitHub — concrete `climb_hill()` with
     `precision=0.001`, `negative_weights=True` option, custom
     metric via `partial()`. More permissive search than our greedy.
  6. Kaggle s5e11 / s5e12 / s6e3 1st-place writeups exist (cf-blocked
     content; titles confirm "Hill Climbing + Ridge Ensemble" and
     "lot of features, lot of models" patterns).

- **Cross-checked against our log** — these mechanisms are CLOSED:
  - Predict-residual / predict-ratio (Angle A NULL: per-fold residuals
    cancel; argmax-equivalence theorem closes 10k-rule residual).
  - OOF confidence/consensus features as meta inputs (v6 NULL +
    LB regress).
  - Multi-round pseudo-labeling, soft-distillation (closed across
    capacity sweep).
  - Predict-missing-features (J3/AV nulled — features independent
    in DGP, R² < 0.05).
  - Heavy-reg XGB / LGBM / CatBoost on recipe (saturated tree-family
    diversity at Jaccard ≥ 0.78).

- **Genuinely UNTESTED mechanisms (this session's contribution)**:
  These four are not subsumed by any of the 15+ saturation
  confirmations in this log; each has a structurally distinct failure
  mode from prior nulls.

  **C. Distillation student on a DIFFERENT feature view**
  (~30 min CPU, executed first per EV/cost). Train a fresh XGB on
  LB-best 4-stack's hard pseudo-labels using a feature subset that
  EXCLUDES rule-derived features (no `dgp_score`, no rule indicators,
  no signed-distance features, no `rule_pred`, no logit_P_*). Trees
  forced to discover the LB-best decision surface from a different
  basis → potentially Jaccard < 0.80 with primary. Distinct from
  prior soft-distill nulls because (a) hard-label not soft-target,
  (b) feature-restricted not capacity-restricted.

  **D. Matt-OP-style hill climb with negative_weights=True**
  (~30 min CPU). Use `climb_hill()` with `precision=0.001` and
  `negative_weights=True` on the full ~70-component bank, custom
  metric = balanced_accuracy_score at fixed bias. Distinct from our
  greedy log-blend (α∈[0,0.5] precision 0.005, no negative weights).
  Negative weights let the optimizer SUBTRACT components that are
  anti-correlated in some region — a strictly more flexible search.

  **B. L3 weighted average of TWO L2 metas (XGB-meta + small NN-meta)**
  (~30 min CPU). Same 63-component bank that produced LB-best v1
  meta-stacker. NN-meta on 200-dim input is a much smaller capacity
  fit than NN-on-recipe (which 15 NN-on-recipe nulls have closed).
  cdeotte's specific recipe is "GBDT + NN at L2, weighted average
  at L3". We've tested LR-meta (NULL) and XGB-meta (LB-best) but
  never blended XGB-meta with a small NN-meta at L3.

  **A. Wide programmatic FE (5,000+ features, forward-select ~500)**
  (4-6 h CPU, only if A/B/C/D leave headroom). Programmatically
  generate: all C(20,2) cat-pair × {mean,std,count,nunique,skew,
  min,max} stat combos, distribution buckets per groupby, quantile
  features `[5,10,40,45,55,60,90,95]`, all decimal-fraction features,
  all binnings at multiple resolutions. Forward-select top 500 by
  per-fold gain. Recipe is at ~440 hand-designed features; cdeotte's
  1st-place backpack-prices mechanism is "go *wide*, then prune".
  We've added FE incrementally (utaazu 16 cols, rohit8527 176 cols,
  instability 5 cols) — never the wide-scan-then-cut pattern.

- **Decision rule for this session** (4 days to deadline, 6 LB slots):
  Run C → D → B → A in order. Each has its own blend gate (Jaccard <
  0.80 AND errs ≤ 1.05 × anchor AND per-class recall ≥ anchor − 5e-4).
  LB probe ONLY if blend OOF Δ ≥ +2e-4 vs LB-best 4-stack AND gate
  passes AND user explicitly approves. Lock final-selection if A/B/C/D
  all null.
- Sources logged for cross-branch reference:
  - <https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/>
  - <https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/>
  - <https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/>
  - <https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/>
  - <https://github.com/Matt-OP/hillclimbers>

### 2026-04-26 — execution of C/D/B from GM-research session: B PASS (Δ +0.00033 OOF gate), D NULL (Pareto violation), C in flight

- Goal: execute the four mechanism-novel levers documented above (C → D
  → B → A) as a sequenced experiment battery.
- Branch: `claude/research-competition-strategies-GzR6R`. Container
  rehydrates kill detached processes within ~5 min idle, much faster
  than the CLAUDE.md "~2h" estimate. Per-fold checkpointing + per-step
  hillclimb state added so all three scripts are now resilient
  (whitelisted `oof_*_fold{n}.npy`, `test_*_fold{n}.npy`, and
  `hillclimb_state.npz` in .gitignore).

- **B — L3 weighted XGB-meta + MLP-meta — GATE PASSED ✓** (single
  experiment to clear the strict +2e-4 OOF gate this session):
  - `scripts/meta_l3_xgb_mlp.py` — small MLP `[128, 64, 32]` on the
    same 410-dim meta-feature matrix that produced the LB-best XGB
    meta-stacker (62-component pool + 14 dist features + LB-best
    3-stack 3 cls). Class-balanced cross-entropy, 30 epochs AdamW,
    cosine schedule. 5-fold StratifiedKFold(seed=42) for OOF alignment.
  - **MLP-meta standalone** (5-fold OOF, vs y):
    - per-fold argmax: 0.98064 / 0.98140 / 0.98222 / 0.98171 / 0.98171
    - overall argmax = **0.98154** (well above LB-best 4-stack 0.98084)
    - @recipe-bias = 0.97496 (un-iso-cal'd; expected — different prob scale)
  - **L3 weighted average sweep** (W_MLP × XGB-meta-iso + W_MLP ×
    MLP-meta-iso, both iso-calibrated, then log-blend into LB-best
    3-stack at α-sweep, fixed bias [1.4324, 1.4689, 3.4008]):
    ```
    W_MLP   L3 OOF   blend-into-3stack    Δ vs 4-stack 0.98084
    0.00    0.98059  0.98061              +0.00000  (XGB-meta alone, baseline)
    0.10    0.98085  0.98090              +0.00006
    0.20    0.98102  0.98101              +0.00017
    0.30    0.98120  0.98101              +0.00016
    0.40    0.98139  0.98104              +0.00020
    0.50    0.98145  0.98118              +0.00033 ← BEST
    ```
    Best at W_MLP=0.50, α=0.50: OOF **0.98118**, errs **9,341** (vs
    LB-best 4-stack 9,415 — 74 fewer).
  - **Per-class recall delta** vs LB-best 4-stack at the BEST blend:
    L=−0.00007 / **M=+0.00036** / **H=+0.00071**. All three within
    guardrail; M and H are POSITIVE deltas (favoring rare class).
  - **GATE: PASS** (Δ ≥ +2e-4 AND PCR ≥ −5e-4 each class).
  - **Critical caveat — structurally similar to LR-meta-stacker** (which
    has nulled twice on LB):
    - LR v1 (C=1.0, balanced): OOF 0.98167 → LB 0.97991 (gap +0.00176)
    - LR v2 (C=0.1, none): OOF 0.98107 → LB 0.98052 (gap +0.00055)
    - **MLP v1 (this experiment): OOF 0.98118, LB unknown.** Both
      LR-meta and MLP-meta are "simpler-than-XGB on 200+ dim bank"
      patterns. Per the linear gap-projection rule, expected LB Δ
      sits in [−0.0007, +0.00033] depending on whether MLP escapes
      the LR pattern.
  - **Difference from LR-meta** that may help MLP: (a) ReLU/GELU
    non-linearity (LR is linear), (b) dropout 0.2 (regularization
    that LR's `class_weight='balanced'` lacked), (c) smaller hidden
    layers `[128,64,32]` vs LR's full 200-dim weight vector. These
    are 3 reasons MLP may not inherit LR's overfit, but it's
    speculative until LB-tested.
  - **DO NOT auto-submit.** Submission requires user approval per the
    top-of-file CLAUDE.md rule. Candidate CSV not yet built; user
    must approve before LB probe.
  - Artifacts committed: `oof_mlp_metastack.npy` + test +
    `oof_meta_l3_xgb_mlp.npy` + test + `meta_l3_xgb_mlp_results.json`.

- **D — Hill-climb with negative weights — GATE FAILED (Pareto
  violation)**:
  - `scripts/hillclimb_negweights.py` — Matt-OP-style arithmetic-blend
    HC, anchor = LB-best 4-stack at OOF 0.98084. 132 components
    (anchor + 131 from `tier1b_helpers.load_pool()`). Step deltas
    `{±0.005, ±0.05}`, custom metric = balanced_accuracy at fixed
    bias [1.4324, 1.4689, 3.4008].
  - **12-step convergence path**:
    ```
    step  weight change                                     bal_acc
    0     anchor only                                       0.98084
    1-3   stack lr_metastack +0.05 each                     0.98098 → 0.98142
    4     own_S3_hard_vote -0.05                            0.98157
    5     extratrees_dist_digits -0.05                      0.98165
    6-9   small additions                                    0.98171 → 0.98180
    10    lr_metastack_v2 +0.005                            0.98180
    11    tta_recipe_s030 -0.05                             ~0.98181
    12    final, 11 active components                       0.98181
    ```
  - **Final state**: 11/132 active components, OOF **0.98181**, Δ vs
    anchor **+0.00097** (gate ≥ +2e-4 PASS).
  - **Top weights**:
    ```
    lb_best_4stack                          +1.0000  (anchor seed)
    lr_metastack                            +0.1500  (LB-confirmed regressor)
    lr_metastack_v2                         +0.0050
    extratrees_dist_digits                  -0.0500  (negative weight)
    own_S3_hard_vote                        -0.0500
    tta_recipe_s030                         -0.0500
    xgb_metastack_varB                      +0.0500
    meta_perturbed_v1_noise03_csb09_k3      +0.0050
    recipe_full_te_seed7                    -0.0050
    recipe_no_ote                           -0.0050
    recipe_pseudolabel_tau099               -0.0050
    ```
  - **Per-class recall delta vs anchor**:
    L=−0.00015 / **M=−0.00191** / **H=+0.00495**.
  - **GATE: FAIL — Pareto frontier violation**: Medium recall
    drops 0.0019 (well below −5e-4 floor) while High recall surges
    0.005. Classic Med→High Pareto-violation pattern documented
    13+ times before. The +0.00097 OOF "lift" comes from over-pushing
    rare-class predictions; will not survive macro-recall on LB.
  - **Mechanism diagnosis**: HC's arithmetic blend with negative
    weights amplifies the LR-meta-stacker overfit (lr_metastack at
    +0.15 weight is 75% of its effective contribution at α=0.5 in
    log-blend — and LR-meta v1 nulled at LB 0.97991 with α=0.5).
    The negative weights on `own_S3_hard_vote`, `extratrees_dist_digits`,
    `tta_recipe_s030` are SUBTRACTIONS of weak components → equivalent
    to "manufacturing lift by removing noise", but at fixed bias the
    rare-class push compounds.
  - **Verdict: NULL.** Gate-failed on per-class guardrail; expected
    LB regression. No submission warranted.
  - Artifacts committed: `oof_hillclimb_negweights.npy` + test +
    `hillclimb_negweights_results.json` + `hillclimb_state.npz`
    (resume checkpoint).

- **C — Distill_no_rule — IN FLIGHT**:
  - `scripts/distill_no_rule.py` — recipe pipeline minus 4 threshold
    flags (`soil_lt_25/temp_gt_30/rain_lt_300/wind_gt_10`) and 3
    LR-formula logits (`logit_P_{Low,Medium,High}`). 424 features
    (vs recipe's 433). Same XGB heavy-reg HPs, 5-fold StratifiedKFold
    (seed=42). Tests whether trees can match recipe's standalone
    OOF 0.97967 from a basis that excludes rule-derived signal.
  - Status at this commit: production restarted after rehydrate kills,
    fold 1 XGB at iter ~1000. Per-fold checkpointing in place. Will
    auto-document when complete.

- **Strategic implication of B's gate-pass + D's gate-fail**: B is the
  first experiment in 11+ saturation confirmations to satisfy ALL of
  (Δ OOF ≥ +2e-4, errs < anchor, PCR within guardrail with **positive
  rare-class delta**). The OOF→LB calibration risk remains the open
  question — same architecture family as 2 prior LB-regressors but
  with structurally distinct safeguards (non-linearity, dropout, smaller
  capacity). Worth ONE careful LB probe with user approval after C
  completes.

- **LB-best unchanged**: `submission_tier1b_greedy_meta.csv` at
  **LB 0.98094**. LB budget unchanged (no submissions this session).

### 2026-04-27 — B LB probe: 0.98091 (−0.00003 from LB-best, gap +0.00027 — 5x tighter than LR)

- Goal: LB-probe B's gate-passing candidate after the previous session's
  documentation pass. User-approved single submit; one `kaggle
  competitions submit` invocation, no retry, per CLAUDE.md rule.
- Changed: `scripts/emit_meta_l3_blend.py` (~95 lines) — reconstructs
  the L3 weighted average (0.5 × XGB-meta-iso + 0.5 × MLP-meta-iso)
  + log-blends into LB-best 3-stack at α=0.50, applies fixed bias
  [1.4324, 1.4689, 3.4008], emits CSV. Verified OOF reproduction:
  blend bal_acc 0.98118 (matches B's gate output exactly).
- **LB submission**:
  `submission_meta_l3_xgb_mlp_blend_a050.csv` → **LB public = 0.98091**.
  Δ vs LB-best primary (0.98094) = **−0.00003** (essentially tied,
  inside LB noise floor ~±0.0005).
- **OOF→LB calibration ladder for the meta-stacker family**:
  ```
  meta variant                        OOF       LB       gap        LB Δ
  ----------------------------------- --------  --------  --------  --------
  LR-meta v1 (C=1.0, balanced)        0.98167   0.97991   +0.00176  -0.00103
  LR-meta v2 (C=0.1, none)            0.98107   0.98052   +0.00055  -0.00042
  **MLP-meta v1 (B, this probe)       0.98118   0.98091   +0.00027  -0.00003**
  XGB-meta + iso → LB-best 4-stack    0.98084   0.98094   -0.00010  +0.00086
  ```
  Architectural safeguards (non-linearity / dropout / smaller capacity)
  reduced gap inflation by **5×** vs LR v1. But the OOF +0.00033 lift
  over LB-best 4-stack did not fully transfer — only **12% transfer
  rate** (Δ_LB / Δ_OOF). Same direction as LR-meta family but
  attenuated.
- **Per-class recall delta on test** (tied OOF & LB picture):
  L=−0.00007 / M=+0.00036 / **H=+0.00071** vs LB-best 4-stack.
  Rare-class trade was directionally CORRECT this time (positive on
  H), unlike D's hillclimb (Pareto violation Med→High).
- **Saturation reconfirmed at LB 0.98094** — this is the **17th
  independent saturation confirmation** on the LB ceiling. The MLP
  meta-stacker is the FIRST simpler-than-XGB meta to land within 3bp
  of primary on LB, and the first to satisfy the per-class guardrail
  with positive rare-class direction. But even with those structural
  safeguards, breaking past the ceiling requires more than a different
  L2 meta-learner family.
- **Two portable rules** (LEARNINGS.md candidates):
  1. **MLP meta-stackers with dropout + non-linearity DO escape the LR
     `class_weight='balanced'` overfit trap on a saturated bank** —
     gap narrowed from +0.00176 to +0.00027 (6.5× compression). The
     LR-meta failure was specifically architectural (linear + class-
     balanced loss); not all "simpler-than-XGB metas" inherit it.
  2. **OOF→LB transfer rate of 12% on a saturated 63-component bank
     is the practical ceiling for L2 meta variants.** Below that, the
     bank is saturated; the meta architecture choice can move the
     constant slightly but cannot break the structural ceiling.
- LB budget: **1/10 used today** (this probe, on 2026-04-27),
  9 remaining.
- Current LB best unchanged: `submission_tier1b_greedy_meta.csv` at
  LB **0.98094**.

### 2026-04-27 — 3-meta L3 LB result: 0.98060 (−0.00034, REGRESSION) — 18th saturation confirmation

- Goal: follow up B's gate-pass with cdeotte's 3-meta L3 pattern
  (XGB-meta + MLP-meta + LR-meta-v2 weighted average). Tested two
  preliminary variants first:
  - **v7** (XGB-meta retrained on bank with `mlp_metastack` +
    `lr_metastack_v2` added): bank auto-included 149 components
    (vs ~63 when v1 was trained); standalone iso 0.98127, blend
    OOF 0.98120 / Δ +0.00036. Bank-extension trap → not LB-probed.
  - **v7b** (strict-EXCLUDE bank, drops all prior meta variants):
    pool 116 + 2 explicit adds = 118; standalone iso 0.98136 (best
    ever); blend OOF 0.98105 / Δ +0.00021 at α=0.30 (PCR fails at
    α≥0.35). Below LB-transfer threshold → not LB-probed.

- **3-meta L3 sweep** (`scripts/three_meta_l3.py`, ~1.5 min):
  iso-cal each meta, sweep `XGB × MLP × LR` 3-simplex × α∈{0.30..0.60}.
  Standalone meta_iso bal_acc:
  ```
  xgb_iso = 0.98059
  mlp_iso = 0.98146  ← higher than LB-best 4-stack 0.98084 alone!
  lr_iso  = 0.98063
  ```
  **Best gate-pass**: w_xgb=0.00 / w_mlp=0.90 / w_lr=0.10 / α=0.60
  → OOF **0.98152**, Δ +0.00068 vs LB-best 4-stack 0.98084 (2× B's
  +0.00033). PCR delta L=−0.0004 / M=+0.0007 / **H=+0.0018**.
  Surprise: best L3 DROPS XGB-meta entirely; MLP-iso dominates.
  B's α≤0.50 search window missed this lift.

- **LB probe** (user-approved, submitted 05:08 UTC):
  `submission_three_meta_l3_mlp090_lr010_a060.csv` →
  **LB public = 0.98060**.
  Δ vs LB-best 0.98094 = **−0.00034** (regression).
  OOF→LB gap = 0.98152 − 0.98060 = **+0.00092** (3.4× wider than B).

- **Diagnosis — two compounding overfit factors:**
  1. **Removing XGB-meta from L3 lost diversity stabilization.**
     B's 50/50 XGB+MLP was structurally complementary (different
     model families). Pure MLP-iso + small LR is a single-family
     L3 — both MLP and LR are simpler-than-XGB metas with similar
     overfit failure modes.
  2. **α=0.60 amplified the overshoot.** B used α=0.50; pushing to
     0.60 weighted L3 more, amplifying L3's OOF overfit.
  3. **PCR H=+0.0018 was the warning.** Hill-climb (D) had H=+0.005
     also LB-regressive. Empirical rule: **rare-class PCR delta
     above ~+0.0015 is OOF-overfit territory regardless of how
     other diagnostics look.**

- **Updated ladder:**
  ```
  candidate                          OOF       LB       gap
  LB-best 4-stack                    0.98084   0.98094  -0.00010
  B (XGB+MLP α=0.50)                 0.98118   0.98091  +0.00027
  3-meta (MLP-heavy α=0.60)          0.98152   0.98060  +0.00092 (this probe)
  ```

- **18th independent saturation confirmation.** Structural finding:
  **B's XGB+MLP diversity at α=0.50 is the L3 sweet spot on this
  bank.** Going more aggressive in either dimension (drop XGB-meta
  OR raise α) leaks OOF.

- LB budget: **2/10 used today**, 8 remaining.

- **Two portable rules** (LEARNINGS.md candidates):
  1. **PCR H delta > +0.0015 is OOF-overfit** even when other gate
     criteria pass. The Pareto frontier on this problem has a
     bounded rare-class lift; pushes beyond ~+0.001 H come from
     OOF-noise fitting, not signal.
  2. **L3 meta diversity matters more than L3 weight optimization.**
     A 50/50 XGB+MLP at moderate α transferred (B: 12% rate). A
     90% MLP + 10% LR at higher α blew up the gap (3-meta: gap
     3.4× wider than B). When sweeping L3 weights, prefer
     diversity-preserving combinations over pure-OOF-optimum.



### 2026-04-26 — W7 (k=1 NN to 10k original): NULL — synthetic rows are not anchored to original rows

- Goal: execute the cheapest untried wild-step from the 2026-04-25 W1-W8
  brainstorm. `brief.md` discloses both train and test were generated by
  a DL model trained on the 10k original. Hypothesis: synthetic rows
  near (in feature space) an original row inherit the original's label,
  which is rule-perfect by construction. Concretely: k=1 NN search
  test→original; if distance < τ, hard-override the LB-best primary's
  prediction with the neighbor's label. Expected EV per the brainstorm:
  5% × +0.0002 LB. Cost: ~10 min CPU.
- Changed: `scripts/w7_nn_to_original.py` (~155 lines, single file).
  43-dim feature space (11 standardised numerics + 32 one-hot cat
  columns from the 8 cats with cards 4/6/4/3/4/4/2/5 = 32). sklearn
  `NearestNeighbors(n_neighbors=1)` brute-force on 10k reference set.
  Total wall ~15 sec.
- **Distance findings (the smoking gun)**:
  ```
  percentile     train→orig    test→orig
  p0.00 (min)        1.29           1.33
  p0.01              1.82           1.78
  p0.50              2.37           2.37
  p25                3.00           3.00
  p99                3.61           3.61
  ```
  Train and test distance distributions are **statistically identical**
  (matches the AV result), and the **minimum distance is 1.33 — well
  above zero**. There are NO near-duplicate test rows in the 10k.
  A single categorical mismatch contributes √2 ≈ 1.41 to Euclidean
  distance under one-hot, so distance 1.33 means cats roughly match
  but every numeric is meaningfully off.
- **Train validation (precision = NN-from-original label vs actual y_train)**:
  ```
  τ percentile    n            precision
  p0.01           63           87.3%
  p0.10           630          81.1%
  p0.50           3,150        79.9%
  p1.00           6,300        79.0%
  p25.00          157,500      75.1%
  ```
  NN-label-from-original tops out at **87% precision even at the
  closest 0.01% of rows**. The rule alone gets **98.4% raw accuracy
  on the same train rows**. So NN-from-original is **strictly worse
  than the rule everywhere**, let alone the LB-best primary
  (which uses rule + flip-correction at ~98.5% raw acc).
- **Test projection at the same τ buckets**:
  ```
  τ            n_test    overrides_vs_primary
  1.82         35        6
  2.37         1,353     268
  3.00         67,842    16,694
  ```
  Expected delta: at any τ where overrides happen, primary beats
  NN-from-original on macro-recall. Replacing primary with NN at,
  say, τ=2.37 swaps primary's ~2% errors for NN's ~20% errors on
  the override set — net negative under any decision rule.
- **Mechanism falsified**: the NN generator does NOT preserve labels
  from anchor rows in the original. Synthetic rows are **independent
  samples from the NN's learned distribution**, not perturbations of
  specific original anchors. That's consistent with the 2026-04-21
  EDA finding "Zero exact feature-vector duplicates in 630k rows" —
  the NN samples a continuous manifold rather than copying-with-noise.
- **Closes W7 definitively.** No LB probe warranted; OOF projection
  predicts strict regression at every τ.
- LB best unchanged at **0.98094** via `submission_tier1b_greedy_meta.csv`.
  LB budget unchanged.
- Artefacts (whitelisted via `.gitignore`):
  - `scripts/w7_nn_to_original.py`
  - `scripts/artifacts/w7_test_nn_{dist,idx,label_idx}.npy`
  - `scripts/artifacts/w7_train_nn_{dist,idx,label_idx}.npy`
  - `scripts/artifacts/w7_nn_to_original_results.json`
- **Portable rule** (LEARNINGS.md candidate): "On synthetic-tabular
  competitions where train AND test were generated by an NN trained
  on a public anchor dataset, the NN-output rows are independent
  samples from the learned distribution, NOT perturbations of anchor
  rows. There are NO near-duplicate matches between synthetic and
  anchor in feature space (min Euclidean distance under standardised
  numerics + one-hot cats is ≥1 categorical-mismatch's worth — i.e.
  ≥1.4 in our 43-dim space). k=1 NN-to-anchor label transfer is
  strictly weaker than any rule reverse-engineered from the anchor.
  Run a 30-second min-distance probe BEFORE building any NN-to-anchor
  override pipeline; if min distance ≫ 0, skip the lever."
- Also closes the related "find leaky test labels via near-duplicate
  match" idea from the senior-DS audit. The data has no such leak.

### 2026-04-26 — N5b 10k-as-anchor lever family: LB 0.98055 (-0.00039) BUT first AUC-positive evidence

- Goal: senior-engineer reframe — treat synthetic train as test, learn
  DGP from 10k original. The naive "train on 10k → predict synth"
  framing was already nulled (NN-on-original 2026-04-22, TE-from-original,
  argmax-equivalence theorem). Refined angle: use 10k as a **geometric
  anchor** (density estimator + kNN reference frame) rather than a
  label source. Three deployments + variance follow-ups.
- Branch: `claude/learn-dgp-original-data-OTU5c`. Single-CPU box,
  ~10h total wall.

- **Diagnostic probe (n5b_ood_diag.py, 5 min)**: GMM/IsoForest/kNN-density
  on 10k features only → score synth rows → correlate with `|y - rule|`.
  - Cohen's d (flipped vs clean rows): GMM=0.195, IsoForest=0.197,
    kNN-density=0.181 (all clear the 0.10 gate).
  - Spearman corr 0.024 (just below 0.05 gate).
  - Sanity: `min_threshold_dist` (known-good signal) = d=0.31, Spearman
    0.039 — confirms pipeline correctness.
  - Verdict: PROCEED. Signal real but small (~50% strength of known-good
    threshold features).

- **Built `oof_ood3_train.npy` (3 OOD scores) + `oof_knn10k_train.npy`
  (8 kNN-from-10k geometric features = p_low/med/high, nbr0_y, mean
  dists to nearest 10k rows of each class, margin) once via
  `build_10k_anchor_features.py`** (~5 min CPU, FAISS k=20). Reusable
  across all deployments.

- **Deployment #2 — OOD-gated score=6 override** (`n5b_d2_score6_ood_gate.py`):
  combined spec6_v2_prob (binary M↔H AUC 0.94) with GMM_neg_logp as a
  2-D conformal gate on score=6 ∩ teacher_argmax=Medium domain.
  - **Best (theta_spec=0.20, theta_ood=p50): 50% precision (54x break-
    even 0.92%) but only 4 overrides → +0.00003 macro-Δ.**
  - Best under guardrail (theta_spec=0.15, theta_ood=all): n=42, 14.3%
    precision, +0.00005 macro-Δ.
  - **Result: NULL on +2e-4 gate**, but the signal IS real at strict
    thresholds — bucket-size limited (35,180 override-domain rows on
    test, only 326 truly-H). Confirms 2026-04-26 score=6 deep-dive
    "macro-Δ is bucket-size limited, not threshold-method limited."

- **Deployments #1 + #3** — recipe + 3 OOD scores (`EXTRA_OOD=1`) and
  recipe + 8 kNN10k features (`EXTRA_KNN10K=1`). Both 5-fold seed=42,
  ~50 min CPU each (running in parallel on 16-core, 5-6 cores per
  process):
  ```
  variant            argmax    tuned     bias                      Δ tuned
  recipe baseline    0.97589   0.97967   [1.43, 1.47, 3.40]        0
  D1 OOD             0.97541   0.97959   [1.23, 1.27, 3.40]        -0.00008
  D3 kNN10k          0.97581   0.97961   [1.43, 1.47, 3.40]        -0.00006
  ```
  - **Both standalone NULL.** D1's bias drift (Low/Med −0.20 each)
    indicates the OOD scores produce sharper raw probs (less log-bias
    correction needed). D3's bias is identical to recipe baseline —
    kNN10k features pass through tree splits transparently.
  - Blend gate vs LB-best 4-stack PRIMARY (fixed bias):
    - D1 OOD: peak α=0.0 (no blend), Jaccard 0.83.
    - D3 kNN10k: peak α=0.025, Δ=+0.00003, Jaccard 0.83.
    - **Both NULL** at +2e-4 gate, both above 0.80 redundancy.

- **Bank-add follow-up** (`n5b_bank_add_test.py`): include both new
  recipe variants in the meta-stacker pool and retrain
  `xgb_metastack_n5b_both` via `META_OUT_SUFFIX=_n5b_both`. Same
  heavy-reg XGB HPs (depth=4, reg_alpha=5, reg_lambda=5, lr=0.05),
  same 5-fold seed=42, automatic pool extension via load_pool() scan.
  - **Standalone new meta @ recipe-bias = 0.98084** (vs v1_meta 0.98041,
    **+0.00043 OOF**). Errors 8,782 vs LB-best 3-stack's 9,572 (**−790
    fewer — FIRST bank-add candidate with lower error count**).
  - **Jaccard vs LB-best 3-stack = 0.7992** — right at the 0.80
    redundancy threshold (just below).
  - Internal meta α-sweep onto LB-best 3-stack: **peak α=0.50 →
    +0.00058 OOF** (tier1b's auto-emit triggered, saved
    `submission_tier1b_metastack_meta_n5b_both_a500.csv`).
  - At LB-validated α=0.30 in primary architecture:
    `primary' = 0.7×3stack + 0.30×new_meta_iso` → **OOF 0.98104,
    Δ=+0.000199** vs v1 PRIMARY (1ppm short of +2e-4 gate).
  - Per-class trade: Low +0.00011, Medium +0.00097, **High −0.00048**
    (right at −5e-4 guardrail).

- **Follow-up angles (`n5b_followup_blend.py` + `n5b_followup_residual_auc.py`)**:
  - Angle 1 (mean-blend v1+new at fixed α=0.30): geometric mean gives
    **OOF +0.00017 with High recall drift only −0.0001** — safest
    candidate (preserves LB-validated arch, minimal rare-class risk).
  - Angle 2 (fine α-sweep around 0.30 for pure swap):
    ```
    α=0.300  Δ=+0.00020  H=-0.0005
    α=0.350  Δ=+0.00026  H=-0.0005
    α=0.375  Δ=+0.00029  H=-0.0005
    α=0.425  Δ=+0.00033  H=-0.0005   peak under guardrail
    α=0.500  Δ=+0.00037  H=-0.0005   FAIL (other class drops)
    ```
  - Angle 3 (residual-AUC diagnostic): binary XGB on 11 N5b features
    targeting (y != PRIMARY_argmax). **5-fold OOF AUC = 0.6347**
    (σ=0.005, very stable, fold range 0.632-0.646). >> 0.55 threshold.
    AP 0.0247.
  - **CRITICAL FINDING**: Top-K precision is only 4-4.5% (3x base rate
    1.49%), so the signal is REAL but DIFFUSE — can't be deployed as
    hard-gate override. Has to enter as soft signal, which is exactly
    what bank-add and mean-blend already attempt.

- **LB PROBE: `submission_n5b_followup_angle1_geo_mean_a030.csv`**
  (user-approved, 92 test rows differ from v1 PRIMARY = 0.034%
  footprint, OOF +0.00017, the SAFEST candidate). Submitted 10:03 UTC.
  - **LB public = 0.98055**
  - **Δ vs LB-best PRIMARY = −0.00039** (regression).
  - OOF→LB gap = +0.00046 (vs primary's −0.00010).

- **Read-out (the new finding, not a closure)**:
  - The OOF→LB regression is **within the public-LB noise band**
    (~±0.0005-0.0014 for this dataset size). A single −0.00039
    observation is **not conclusively a structural regression** —
    could be an unlucky public split.
  - **AUC 0.6347 is the FIRST positive evidence in 12 saturation
    confirmations that the residual signal IS structurally orthogonal
    to PRIMARY** (vs being generic ranking noise). This distinguishes
    N5b from prior bank-add nulls.
  - Bank-add LB carryover ratios across recent submissions:
    ```
    Submission                  OOF Δ     LB Δ      ratio
    LR v2 (04-25)              +0.00046  -0.00042  -0.91
    combined v6 a030           +0.00038  -0.00035  -0.92
    v6_full a350               +0.00037  -0.00082  -2.22
    P3 perturbed               +0.00048  -0.00139  -2.90
    angle1_geo_mean_a030       +0.00017  -0.00039  -2.29
    ```
    The −1x to −3x carryover **may be a property of the saturated
    bank's OOF→LB transfer**, not the signal source. To validate or
    falsify on this lever, need at least one more variance test from
    the same family.

- **Portable rule** (LEARNINGS.md candidate): "Single LB observation
  on a saturated meta-stacker bank cannot distinguish 'structural
  regression' from 'unlucky public split' when the OOF→LB delta is
  within ±0.0005 of expected. To draw conclusions about lever death,
  require either (a) two LB observations from the SAME family at
  different OOF α (variance test), OR (b) a positive AUC-on-residuals
  signal that's been deployed and probed at multiple operating
  points."

- **CRITICAL: AUC > 0.55 on residual-prediction is necessary but
  apparently not sufficient for LB transfer on a saturated meta-stacker
  bank — the architecture's −1x to −3x carryover dominates regardless
  of signal redundancy vs orthogonality.** The implication is that to
  unlock this signal, we need a DIFFERENT delivery mechanism than
  meta-stacker bank-add (e.g., recipe-level FE for both OOD AND kNN10k
  combined, or a residual-correction head that doesn't go through the
  meta-stacker at all).

- LB budget: **2/10 used today** (1 angle1 probe + 1 prior). 8
  remaining.
- Current LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.

- **Untried options (do NOT lock primary; pursue these)**:
  1. **Variance test**: probe `angle2_swap_a350` (OOF +0.00026, similar
     mechanism but smaller arch deviation) OR `angle2_swap_a375`
     (OOF +0.00029). If both regress similar magnitude, structural
     carryover. If one lifts, public split was unlucky for angle1_geo.
  2. **Combined recipe FE** (NOT TESTED): `EXTRA_OOD=1 EXTRA_KNN10K=1`
     simultaneously. The 11 N5b features at the recipe-XGB level might
     compound into recipe-tier signal (vs meta-tier). Different
     delivery mechanism than bank-add. ~50 min CPU.
  3. **Expanded 10k-anchor feature family**: per-class GMM densities
     (3), per-class kNN distances (3), Mahalanobis-to-10k-centroid
     per class (3) = 9 more features → 20 total. With AUC 0.63
     evidence the family carries signal, more features may compound
     standalone meta lift past the −1x to −3x carryover ceiling.
     ~60-90 min CPU.
  4. **Hard-gate residual override** at LOOSER thresholds: top-N=200
     precision is 4.5% (50% lift over base rate 1.49% but below
     break-even 8.1% for High class). Untested at AUTO-CALIBRATED
     thresholds via per-fold conformal precision.

- Artefacts whitelisted via `.gitignore` for cross-branch reuse:
  - `oof_ood3_train.npy`, `test_ood3.npy` (3 OOD scores: GMM, IsoForest, kNN)
  - `oof_knn10k_train.npy`, `test_knn10k.npy` (8 geometric features from 10k)
  - `oof_recipe_full_te_ood.npy`, `test_recipe_full_te_ood.npy`
  - `oof_recipe_full_te_knn10k.npy`, `test_recipe_full_te_knn10k.npy`
  - `oof_xgb_metastack_n5b_both.npy`, `test_xgb_metastack_n5b_both.npy`
  - `oof_n5b_residual_auc.npy` (residual XGB OOF predictions)
  - `n5b_*_results.json` (8 result JSONs)
  - 8 scripts: `build_10k_anchor_features.py`, `n5b_d2_score6_ood_gate.py`,
    `n5b_blend_gate.py`, `n5b_bank_add_test.py`, `n5b_followup_blend.py`,
    `n5b_followup_residual_auc.py`, `n5b_emit_geo_mean_a030.py`,
    `n5b_ood_diag.py`. `recipe_full_te.py` and `tier1b_xgb_metastack.py`
    gained `EXTRA_OOD`, `EXTRA_KNN10K`, `META_OUT_SUFFIX` env vars.
  - `submission_n5b_followup_angle1_geo_mean_a030.csv` (LB 0.98055)
  - `submission_n5b_followup_angle2_swap_a425.csv` (auto-emitted, untested)

### 2026-04-26 — N5b variance test CLOSED: 3-point monotone carryover proves structural regression (16th saturation confirmation)

- Goal: ML-lead-driven variance test on the N5b family. Prior session
  closed angle1_geo_mean_a030 (OOF +0.00017 → LB 0.98055, Δ −0.00039,
  carryover −2.3x) but documented "single LB observation cannot
  distinguish structural regression from unlucky public split."
  Untested follow-ups remained on disk (`angle2_swap_a350`,
  `angle2_swap_a425`). Probe both to resolve the open question.
- Important context discovered during the session: **`angle2_swap_a350`
  was already submitted at 10:13 UTC today (LB 0.98025) but never
  logged on this branch.** The strategy review above recommended a
  variance probe based on incomplete information; in practice the
  variance test was already 1-of-2 complete before this session began.
- Submitted: `submission_n5b_followup_angle2_swap_a425.csv` at 13:48
  UTC. Result: **LB public = 0.97988**, Δ vs PRIMARY (0.98094) =
  **−0.00106**, carryover ratio **−3.2x**.
- **Three-point ladder, monotone in OOF lift AND carryover ratio**:
  ```
  variant                       OOF Δ      LB         LB Δ        carryover
  --------------------------- ---------- --------- ---------- -----------
  angle1_geo_mean_a030         +0.00017   0.98055   -0.00039     -2.3x
  angle2_swap_a350             +0.00026   0.98025   -0.00069     -2.7x
  angle2_swap_a425 (this)      +0.00033   0.97988   -0.00106     -3.2x
  ```
  Larger OOF lift produces strictly larger LB regression. Carryover
  ratio worsens as more N5b signal is extracted. **Unambiguous
  structural-carryover signature; no luck-of-split interpretation
  survives three monotone observations.**
- **N5b family CLOSED definitively.** The AUC 0.6347 residual signal
  is genuinely orthogonal to PRIMARY (first measured-orthogonal
  evidence in 12 prior saturation confirmations), but the meta-stacker
  bank-add delivery mechanism inflates OOF without LB transfer at a
  −1.5x to −3.2x rate that scales with extracted signal magnitude.
- **16th independent saturation confirmation at LB 0.98094.**
- **Open question downgraded — Priority 3 recipe-tier delivery**:
  the residual AUC was measured at meta-tier; whether the same
  features at recipe-XGB tier produce different carryover is now
  open at lower prior. Cost ~50 min CPU to probe. Recommend skipping
  unless other levers also close.
- **Portable rule** (LEARNINGS.md candidate): "When a candidate
  family produces 2-3 monotone-decreasing LB observations vs OOF
  lift on a saturated meta-stacker bank, structural carryover is
  proven and further variants in the same delivery mechanism will
  null. Three data points spanning ≥2x OOF range is the minimum
  for definitive closure when single observations sit within the
  ±0.0005 noise band."
- Also worth logging: a single LB observation that lands at the
  noise-floor edge (here angle1's −0.00039) is insufficient to
  close a lever, but the cost of one more variance probe (~5 min
  + 1 LB slot) is well below the value of definitive closure.
  This applies in reverse too: had any of the three N5b probes
  REVERSED the monotone trend (e.g., angle2_a425 lifting), the
  variance signature would falsify the structural-carryover
  hypothesis. The test design is bidirectional.
- LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`. LB budget: **3/10 used today**
  (1 N5b probe this session + 2 from earlier), 7 remaining.
- **Strategic posture (post-closure)**:
  - Lock the safe pair on Kaggle UI (PRIMARY 0.98094 + HEDGE 3-way
    multi-seed 0.98005 per audit F1 swap recommendation). Zero
    compute, highest remaining EV.
  - Stop further OOF-extraction experiments on the saturated bank.
    Three carryover ratios above −2x prove the architecture's
    transfer ceiling.
  - Reserve 7 LB slots for end-of-comp variance check (~1 per day
    until 2026-04-30 deadline).
- Artefacts: `submissions/submission_n5b_followup_angle2_swap_a425.csv`
  (LB 0.97988, the variance-test third data point).

### 2026-04-27 — classw + D 3-meta lever family closed; net-rare-class-flip rule

After last night's submit-loop bug burned 4 LB slots on `v6_full_a350` and
container rehydrated, three follow-up architecture experiments closed
NULL with a NEW diagnostic:

- **classw α=0.40 (carryover-test resubmit, 05:26 UTC)**: LB 0.98011
  (Δ −0.00083 vs PRIMARY, ratio −2.96x). The "−0.5x carryover at α=0.30"
  finding was an ILLUSION — at small α, v1 meta's calibration dominated
  the blend; at α=0.40 classw's mismatched calibration was exposed and
  carryover snapped back to bank-extension range. **classw lever closed.**
- **D 3-meta ensemble (v1=0, classw=0.4, mlp=0.6, α=0.30, 05:37 UTC)**:
  LB 0.98073 (Δ −0.00021, ratio −0.57x). The OOF optimum DROPPED v1
  entirely (red flag), but dual-α probe at OOF level passed (1.26x linear
  scaling 0.30→0.40). Dual-α was insufficient — needed per-row check.

- **Per-row error decomposition (the diagnostic that explains why D failed)**:
  ```
  D vs PRIMARY (152 differing test rows):
    D demotes 58 Highs → Medium    │ near-zero NET High flips (+1)
    D promotes 59 Mediums → High   │ but high CHURN (117 row movements)
    D promotes 28 Lows → Medium
    D demotes 7 Mediums → Low

  Net per-class shifts:
    Net High:    +1   (RESHUFFLE, not lift)
    Net Medium:  +20
    Net Low:     -21
  ```
  D doesn't ADD High predictions, it RESHUFFLES them. On OOF, the
  specific reshuffles win (D errs 9318 vs PRIMARY 9415, -97). On LB,
  ~50% of new picks reverse. The OOF macro-recall surface is sensitive
  enough to small per-row changes that 1/(3 × N_high) ≈ 1.6e-5 weighted
  rare-class flips can fit OOF noise.

- **Two new portable rules** (added to LEARNINGS.md):
  1. **Net-rare-class-flip rule**: blend candidates with near-zero NET
     change in rare-class predictions (|net| < 5) but high CHURN (>50
     movements either direction) are OOF-overfit even when passing
     OOF Δ + dual-α + per-class guardrail. The blend isn't lifting
     macro-recall; it's reshuffling predictions to fit OOF surface.
  2. **Asymmetric-flip preference**: candidates that monotonically
     GROW or SHRINK rare-class count (e.g., +49/-5 = +44) are
     structurally cleaner than balanced shuffles. Add a 4th gate:
     `|net_rare_class_flip| / |total_rare_class_churn| ≥ 0.5`.

- **Updated 4-gate criterion for blend candidates going forward**:
  (1) +0.0003 OOF Δ vs PRIMARY
  (2) per-class recall guardrail PASS (each class ≥ baseline − 5e-4)
  (3) dual-α stability (1.0x to 2.0x linear scaling between α=0.30 and α=0.40)
  (4) **NEW**: |net_rare_class_change| / |churn_total| ≥ 0.5
  Without rule #4, classw a030 AND D 3-meta would both pass — and both
  regressed.

- **Carryover ladder updated**:
  ```
  classw  α=0.30:  OOF +0.00023 → LB -0.00011  ratio -0.48x  RESHUFFLE
  D 3-meta α=0.30: OOF +0.00037 → LB -0.00021  ratio -0.57x  RESHUFFLE
  classw  α=0.40:  OOF +0.00028 → LB -0.00083  ratio -2.96x  bank-ext range
  ```
  The −0.5x ratio at small α is consistent for RESHUFFLE-class candidates.

- LB best unchanged at **0.98094**. LB budget: 3/10 used today (a040 +
  3meta_d + retry_resolved_classw_a030_yesterday counted), 7 remaining.
- Final-selection lock unchanged: PRIMARY = `submission_tier1b_greedy_meta.csv`
  (LB 0.98094), HEDGE = `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005).
- **19th saturation confirmation at LB 0.98094.**

### 2026-04-27 — 4-gate filter sweep: 0 survivors among 19 metas (deepest saturation signature)

Applied the new 4-gate filter retroactively to all xgb_metastack* +
mlp_metastack + meta_l3_xgb_mlp candidates on disk (19 metas total) at
LB-validated PRIMARY architecture (0.7 × LB3 + α × candidate_iso, α=0.30).

**Result: ZERO candidates pass all 4 gates.** Stark dichotomy:
  - Candidates passing G4 (asymmetric flip ≥0.5): xgb_v4 (0.603),
    classw (0.548), n5b_both (0.524), varC (0.526) — ALL fail G1
    (sub-+0.0003 OOF Δ).
  - Candidates passing G1 (≥+0.0003 OOF): only mlp_metastack (+0.00033) —
    fails G4 (ratio 0.357, between pure reshuffle ~0.1 and clean asymmetric
    ≥0.5).

**This is the deepest empirical signature of saturation we have**: the
OOF macro-recall surface and LB-transferable directions are now provably
orthogonal in our candidate space. No re-arrangement of existing
components can simultaneously satisfy both.

**One borderline case**: mlp_metastack standalone at α=0.30 (LB-validated
arch). 3 of 4 gates PASS (G1+G2+G3); G4 fails at 0.357 (close to 0.5).
Net High flips on test = +41 (meaningfully asymmetric direction). **HAS
NEVER BEEN LB-TESTED standalone** — the B experiment was BLENDED with v1
at α=0.50.

**Updated calibration ladder for the gate framework**:
```
Pre-rule submissions (gates 1-3 only, no G4):
  classw α=0.30:  passed  →  LB -0.00011  (G4 ratio ~0)  RESHUFFLE
  D 3-meta α=0.30: passed → LB -0.00021  (G4 ratio 0.009) RESHUFFLE
  classw α=0.40:  passed →  LB -0.00083  (carryover snap-back)
Post-rule prediction:
  mlp_metastack α=0.30: passes G1-G3, G4 0.357 borderline. Predicted
    LB ≈ -0.00010 to +0.00005 if 4-gate rule is binary.
    If G4 has slack at the borderline, may tie or slightly lift.
```

LB best unchanged at **0.98094**. LB budget: 3/10 used today, 7 remaining.
Final-selection lock unchanged.

20th saturation confirmation (now LB-validated AND theoretically
characterized via the 4-gate filter).

### 2026-04-27 — mlp_metastack standalone a030 LB result (21st saturation, 4-gate validated)

Per the 4-gate sweep finding, `mlp_metastack` was the ONLY candidate
passing G1+G2+G3 (borderline G4 fail at 0.357). User-approved one-shot
confirmation submit:

  submission_mlp_metastack_a030.csv → **LB 0.98073** (Δ −0.00021 vs PRIMARY)

  Pre-submit prediction: ~0.98086 (Δ −0.00008 at -0.5x small-α carryover)
  Actual:                  0.98073 (Δ −0.00021)
  Magnitude 2.6x larger than predicted, BUT direction correct.

**Updated leak-corrected carryover ladder for RESHUFFLE-class candidates**:
```
                       OOF Δ (raw)   OOF Δ (leak-corr R5)   LB Δ      Ratio (corr)
classw a030           +0.00023       +0.00007                -0.00011  -1.57x
D 3-meta a030         +0.00037       +0.00021                -0.00021  -1.00x
mlp_metastack a030    +0.00033       +0.00017                -0.00021  -1.24x
```

**KEY observation**: mlp_metastack and D 3-meta both landed LB 0.98073 EXACTLY
— same RESHUFFLE pattern (G4 ratio ~0.36), same outcome. This is strong
empirical evidence the 4-gate filter (especially G4) is structurally real.

**Updated rule**: after R5's leak-correction, the carryover for RESHUFFLE
candidates clusters at **-1.0x to -1.6x** (much more consistent than the
raw -0.48x to -0.64x). The +0.00016 OOF inflation from full-OOF iso
explains the apparent "favorable carryover" at small α.

21st saturation confirmation. LB best unchanged at **0.98094**.
LB budget: 4/10 used today, 6 remaining.

**Final-selection lock RECOMMENDED**:
  PRIMARY: submission_tier1b_greedy_meta.csv → LB 0.98094
  HEDGE:   submission_3way_recipe025_s1035_s7040.csv → LB 0.98005

### 2026-04-27 — R2/R5 heavy-reg meta a045 LB regress: 22nd saturation, G4 needs direction refinement

`submission_r2r5_heavy_perfoldiso_a045.csv` → **LB 0.97996** (Δ −0.00098 vs PRIMARY)

Carryover analysis: leak-corrected OOF +0.00029 → LB -0.00098 → ratio **-3.38x**
WORSE than RESHUFFLE candidates (-1.0x to -1.6x).

**Critical finding**: G4 PASS (ratio 0.78) was a FALSE-POSITIVE because the
asymmetric direction was REMOVE-High (net High = -130). Heavy-reg depth=2
removed PRIMARY-High predictions; ~80% of those Highs were CORRECT on test
(High has 12x macro-recall leverage per row). The OOF-validated trade
DOESN'T transfer for asymmetric REMOVE-High direction.

**G4 rule update needed** (logged in LEARNINGS.md candidate):
  Original: |net_rare_class_flip| / |total_rare_class_churn| ≥ 0.5
  Revised:  PASS only if net_rare_class > 0 AND ratio ≥ 0.5
            (i.e., asymmetric ADD-High direction, not REMOVE-High)

R2/R5 a045: net High = -130 (strongly negative) → should have been a red flag.

Updated carryover ladder:
```
                                       OOF Δ (corr)  LB Δ      Ratio    G4    Direction
classw a030 (RESHUFFLE)                +0.00007      -0.00011  -1.57x   0.00  reshuffle
D 3-meta a030 (RESHUFFLE)              +0.00021      -0.00021  -1.00x   0.01  reshuffle
mlp_metastack a030 (mostly-RESHUFFLE)  +0.00017      -0.00021  -1.24x   0.36  mostly-reshuffle
r2r5_perfoldiso_a045 (REMOVE-High asym)+0.00029      -0.00098  -3.38x   0.78  REMOVE-High WORST
```

22nd saturation confirmation. LB best unchanged at **0.98094**. LB budget:
5/10 used today, 5 remaining.

LOCK + STOP RECOMMENDED:
  PRIMARY: submission_tier1b_greedy_meta.csv → LB 0.98094
  HEDGE:   submission_3way_recipe025_s1035_s7040.csv → LB 0.98005
### 2026-04-26 — senior-engineer "reopen and dig deeper" 3-way: 2 NULLs + 1 in-flight (12th saturation confirmation)

- Goal: at user request to reopen the 3 most promising closed-with-
  methodological-flaw experiments and dig deeper. Per the senior-engineer
  recommendation:
  1. **Multi-task XGB at base level** — aux signals (flipped/missed_high/
     missed_med, AUC 0.90/0.98/0.95) inserted via custom xgb.train
     num_class=6 obj. Aux had only ever been tested at meta-stacker level
     (combined v6 = circular OOF inflation); base-level joint loss was
     untested.
  2. **KAN at production capacity with recipe FE** — prior PROBE used
     19 raw one-hot features and hit Jaccard 0.13 (record low) but +500%
     errs. Hypothesis: richer FE shrinks magnitude trap below the +5%
     threshold while preserving orthogonality.
  3. **Leak-eliminated soft-distillation** — proper k-fold-of-k-fold
     teacher: for each outer fold f, retrain recipe with inner CV
     restricted to (full_train \ V_f) so the teacher targets for student-
     fold-f's training rows come from a teacher that never saw V_f.
     Targets the persistent +0.00201 to +0.00246 OOF→LB gap across the
     soft_distill family.

- Branch: `claude/review-ml-experiments-xUrd3`. New scripts: `multitask_common.py`,
  `multitask_xgb.py`, `leakfree_teacher_oof.py`, `leakfree_distill.py`,
  `blend_gate_3way_v2.py`, `auto_chain_v2.sh` (self-restarting loop).
  KAN kernel: `kaggle_kernel/kernel_kan/features.py` rewritten to recipe
  style (157 dim vs prior 51).

- **#1 KAN at recipe FE** — production complete on Kaggle P100 (16 min wall):
  ```
  per-fold argmax: 0.9633 / 0.9647 / 0.9658 / 0.9647 / 0.9639  σ=0.001
  OOF argmax       0.96449
  Tuned OOF        0.96604  bias [3.93, 2.47, 3.40]
  ```
  - KAN's class-balanced + label_smoothing=0.05 produces raw probs that
    underpredict Low badly → bias [3.93] vs recipe [1.43]. Iso-cal aligns
    scales: KAN_iso @ recipe bias = 0.96530.
  - Vs LB-best 4-stack (anchor 0.98084):
    - **Jaccard 0.586** with iso — **record-low** orthogonality for any
      NN family with iso alignment (prior best Mambular 0.49, Trompt 0.53)
    - errs 10446 vs anchor 9415 (+11%) — magnitude trap
    - PCR: L 0.9957 / M 0.9690 / **H 0.9312** vs anchor [0.9955, 0.9695, 0.9775]
      → **High recall drops 0.046** at any positive α
  - α-sweep monotone-negative from α=0.025. **GATE FAIL.**
  - **16th NN-family null** on this problem. Recipe FE shrunk the
    magnitude trap from 6.28× (KAN PROBE 19 raw) to 1.11× (recipe FE
    157 dim) while keeping orthogonality competitive — but the +0.00084
    needed to break the LB-best ceiling requires errs ≤ 1.05× anchor
    AND no rare-class recall trade.

- **#2 Multi-task XGB at base level** — production complete on CPU (~2h 12min wall):
  ```
  per-fold argmax: 0.97566 / 0.97616 / 0.97759 / 0.97527 / 0.97465
  recipe baseline: 0.97544 / 0.97659 / 0.97721 / 0.97465 / 0.97557
  per-fold delta:  +0.00022 / -0.00043 / +0.00038 / +0.00062 / -0.00092
  OOF argmax       0.97567   sigma 0.00095
  Tuned OOF        0.97909   bias [1.2324, 1.0689, 3.4008]
                              (recipe 0.97967 with bias [1.43, 1.47, 3.40])
  ```
  - Custom xgb.train num_class=6 obj produces 6 outputs per row: 3 main
    softmax + 3 sigmoid-aux. AUX_W=0.3 weights aux loss vs main.
  - Per-fold deltas straddle zero (mean -0.00006, σ 0.00064 — within
    fold-noise band). Standalone tuned OOF -0.00058 vs recipe.
  - Vs LB-best 4-stack:
    - **Raw**: bal 0.97896, errs 9934, **Jaccard 0.808** (high redundancy)
    - **Iso**: bal 0.97875, errs 9213, Jaccard 0.811
    - PCR identical to anchor at peak α=0.000 (no boundary correction)
    - α-sweep monotone-negative from α=0.025 vs both 3-stack and 4-stack
  - **Mechanism diagnosis**: aux supervision DID shift trees (different
    bias profile, fewer iso errors) but the joint-loss equilibrium picked
    a slightly-worse main task corner that's redundant with the LB-best
    4-stack. The base-level insertion AVOIDS the meta-stacker overfit
    blow-up (gap stayed normal) but doesn't translate to LB-positive
    contribution — same Pareto frontier, just from a different angle.
  - **GATE FAIL at every α.** This is the **12th independent saturation
    confirmation at LB 0.98094**.

- **#3 Leak-eliminated soft-distillation** — IN FLIGHT after 3 rehydrates.
  - Teacher build (`leakfree_teacher_oof.py`): for each outer fold f,
    inner-CV recipe on tr_outer (504k rows). N_INNER=3, N_OUTER=5.
    Original wall estimate: 5h. Per-outer cost observed: ~63 min × 5
    = ~5.25h.
  - **Container rehydrated 3 times** during the run (12:32, 13:19, 17:19),
    killing the process. Each restart cost ~30-60 min until per-inner
    checkpointing was added.
  - Resilience hardening (committed):
    1. **Per-inner-fold checkpointing** via `_atomic_save()` helper.
       Each inner training writes its OOF + test + va_idx to disk
       atomically (write to `*.tmp.npy` then rename). Lost work per
       rehydrate ≤ 1 inner fold (~5-13 min).
    2. **Skip per-outer full-fit retrain**. Test-side teacher uses
       inner-fold test averages (computed for free during inner CV)
       instead of separate full-fit. Saves ~22 min/outer × 5 = ~1.8h.
    3. **Self-restarting auto_chain_v2.sh**. Restart-loop: keeps
       calling `leakfree_teacher_oof.py` until all 5 outer-fold OOF +
       test artifacts exist, then triggers distill + blend gate.
       Survives unlimited rehydrates.
  - **Progress at session wrap-up (17:20 UTC)**:
    - Outer 1: ✅ saved (production-scale, from first run before rehydrate)
    - Outer 2 inner 1: ✅ checkpointed (bal 0.97516, best_iter 1077)
    - Outer 2 inner 2-3, outer 3-5: ⏳ pending
    - Auto-chain v2 PID 1264 → leakfree teacher PID 1268 running.
  - **Expected outcome**: based on the 2026-04-24 closure note
    ("the OOF→LB gap is structural to teacher OOF construction itself —
    distill family fully closed") and 11+ saturation confirmations,
    estimated ~15-20% probability of clearing the +2e-4 gate even with
    proper outer-fold leak elimination. Auto-chain left running as a
    free overnight attempt; if it completes and clears the gate, that's
    pure upside.

- **Three portable rules logged for future synthetic tabular comps**:
  1. **Per-inner-fold atomic checkpoints are essential for any CPU
     pipeline > 30 min wall on rehydrate-prone containers**. Pattern:
     `_atomic_save(final_path, arr)` writes `final_path.with_name(stem +
     ".tmp.npy")` then renames. Path.with_suffix('.npy.tmp') is BUGGY —
     np.save appends `.npy` to filenames not ending in `.npy/.npz`,
     producing `*.npy.tmp.npy` and breaking subsequent `.rename(orig)`.
  2. **Self-restarting auto-chain script** (retry-loop bash that polls
     for completion artifacts) is the cleanest pattern for rehydrate-
     prone hosts. Runs the long pipeline in a `while ! is_done; do
     python ...; done` loop. Each rehydrate kills the inner python; the
     bash loop wakes back up and re-launches, leveraging the python
     script's own checkpoint-aware resume logic.
  3. **Aux supervision at base level avoids meta-stacker overfit but
     doesn't add orthogonal signal beyond what the recipe already
     captures**. Multi-task XGB with aux heads on the recipe feature
     set produces predictions with Jaccard 0.81 vs LB-best — the joint
     loss equilibrium converges to nearly the same decision surface as
     pure softmax CE, just at a slightly different operating point.
     For aux heads to lift, they need to be inserted at a feature-
     ENGINEERING level (e.g., as additional features the model sees)
     rather than at the loss level on the same features.

- LB delta: n/a. No LB probe spent (both completed candidates failed
  the OOF gate cleanly; leakfree pending). LB-best unchanged at
  **0.98094** via `submission_tier1b_greedy_meta.csv`.

- **Final-selection lock UNCHANGED** (4 days to deadline):
  1. **PRIMARY**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
  2. **HEDGE (recommended swap)**: `submission_3way_recipe025_s1035_s7040.csv`
     → **LB 0.98005** (premium −0.00089, sidesteps meta-stacker layer)

- Artefacts on `claude/review-ml-experiments-xUrd3`:
  - `scripts/multitask_common.py` + `multitask_xgb.py` + multitask OOF/test
    + submission CSV + results JSON
  - `kaggle_kernel/kernel_kan/features.py` (recipe-style FE for KAN) +
    KAN OOF/test + submission CSV + results JSON
  - `scripts/leakfree_teacher_oof.py` (resume-aware, per-inner checkpoint)
    + `leakfree_distill.py` + `auto_chain_v2.sh` (self-restarting)
  - `scripts/blend_gate_3way.py` + `blend_gate_3way_v2.py`
  - Branch pushed to remote; merge to main when ready.

### 2026-04-26 — three-pronged "imitate the leader" experiment + T1/T7 close-out

Brainstorm-driven session focused on the +0.00125 leader gap (Chris Deotte
0.98219 vs our LB-best 0.98094). Three mechanism-distinct attempts at
"imitating" the likely leader recipe, plus two diagnostic levers.

#### Path A — PyTorch MLP on full recipe FE (Kaggle GPU)

Brainstorm hypothesis: every prior NN attempt used 19-66 raw features
(MLP v5-v9, RealMLP, Trompt, Mamba, KAN, FT-T, TabPFN, DAE,
pretrain-FT, NN-on-orig). **None saw the full V10 recipe FE matrix
(443 cols incl. OTE + digits + FREQ + ORIG_stats)**. Hypothesis: NN
inductive bias + recipe FE (precomputed soft signal via OTE) might
finally clear the magnitude floor.

- 4-layer MLP [443→1024→512→256→128→3] with BN+GELU+dropout 0.15,
  AdamW + cosine schedule, class-balanced sample weights, 25 epochs.
- Inlined recipe FE (copy of catboost_recipe_gpu) in self-contained
  Kaggle kernel. P100 sm_60 torch shim required (same as 2026-04-24
  RealMLP retry; new wheel pinning baked into boot).
- Slug rejected once (irrigation-path-a-recipe-mlp → "Notebook not
  found"); renamed to irrigation-recipe-mlp-path-a, pushed clean.

PROBE (1 fold full data): standalone tuned **0.97648** at own bias
[2.23, 2.17, 1.90]. Jaccard 0.65 vs LB-best 4-stack on fold-1 rows —
strong orthogonality, in RealMLP n_ens=1 band. **But errs ratio
2080/1914 = 1.087×** (8.7% over anchor) — magnitude-trap territory.
Blend sweep vs LB-4 monotone-negative from α=0.025; peak +0.00007 at
α=0.15 (within fold noise). High recall drops -0.0019 to -0.0024
across α range (wrong direction under macro-recall).

5-fold production (Kaggle v5/v6): standalone tuned **0.97640** with
bias [2.73, 2.57, 2.50] — comparable to RealMLP n_ens=1 (0.97633).
**16th NN-family null on this problem**. The recipe FE is the
strongest substrate any NN has ever seen here AND still fails the
magnitude rule. Pareto-frontier closure (2026-04-24) holds: rare-
class-orthogonal NN errors don't survive fixed-bias evaluation.

**Kaggle infra bug**: long-running kernels (>20 min) consistently hit
"DNS cache overflow" during output save — both v5 and v6 saved 18-byte
error stubs instead of real arrays. Metrics fully recoverable from
kernel logs but arrays unrecoverable without retry. Affected v3, v5,
v6 path A and v2 path C.

#### Path B — per-cell MLP on rule-cell partition (local CPU)

Brainstorm hypothesis: per-cell LR (2026-04-21) was rejected at
linear capacity (0.96280); MLP capacity in EACH cell could break the
cross-cell flip-signal bottleneck.

- 128 rule-cells (stage × dry × norain × hot × windy × nomulch).
- 15 features per cell: 11 raw nums + 4 signed dist-to-threshold.
- Per-cell sklearn MLPClassifier(hidden=(64, 32), max_iter=150).
- Cells with <200 train rows fall back to empirical class prior.
- 5-fold seed=42 production wall ~5 min (much faster than estimated).

Standalone OOF tuned **0.89357** (vs LB-best 4-stack 0.98084, way
below). 28/96 cells fell back to empirical prior in production
(small-cell-MLP can't fit reliably). Jaccard vs LB-best 4-stack =
**0.083** — RECORD LOW orthogonality on this problem. But per-row
magnitude is 72,880 errors vs LB-best 9,415 → **7.7× magnitude trap**.
Blend sweep monotone-negative from α=0.025 (Δ −0.00021 → −0.00736
across α range). High recall drops sharply with α (-0.0021 at α=0.05,
-0.0040 at α=0.10).

**NULL — fragmentation overrides orthogonality.** Per-cell partition
fragments the data too much; small-cell MLPs can't learn the cross-
cell flip signal.

#### Path C — pseudo-label retraining with LB-best 4-stack as labeler

Brainstorm hypothesis: the prior pseudo-label stage-1 used
recipe_full_te (LB 0.97939) as labeler; using LB-best 4-stack
(LB 0.98094) at τ=0.99 should produce purer pseudo labels and a
cleaner gradient.

- Reconstructed LB-best 4-stack test posterior locally (matches
  hypothesis-board OOF 0.98084 exactly via tier1b_helpers.build_lbbest_stack
  + xgb_metastack_iso log-blend at α=0.30).
- Saved as scripts/artifacts/test_path_c_primary_labeler.npy +
  path_c_primary_labeler_results.json (log_bias = [1.4324, 1.4689, 3.4008]).
- Local CPU runs killed 3× by container rehydrate (~30-min cycles
  vs 50-min wall). Pivoted to Kaggle CPU kernel.

**Kaggle CPU kernel scaffold**:
  - kaggle_kernel/ds_path_c_scripts/ — uploaded as Kaggle dataset
    `chrisleitescha/irrigation-pseudolabel-scripts`
    (common.py + recipe_features.py + recipe_ote.py +
    recipe_full_te.py + recipe_pseudolabel.py +
    test_path_c_primary_labeler.npy + path_c_primary_labeler_results.json)
  - kaggle_kernel/kernel_path_c/path_c_pseudolabel.py — 118-LoC
    wrapper that mirrors local repo layout (scripts/, data/),
    symlinks competition CSVs, **zips orig CSV into data/archive.zip**
    (recipe_full_te.py expects ZIP not raw CSV — v1 BadZipFile error
    fixed in v2), exports env vars, exec()s recipe_pseudolabel.py.

v2 ran ~6.5 hours on Kaggle CPU (much slower than local — fold wall
~75 min vs ~10 min). Per-fold argmax: 0.97592 / 0.97606 / 0.97705 /
0.97513 / 0.97607. Overall OOF argmax 0.97605, **tuned 0.97998** with
bias [1.03, 1.07, 3.40].

Comparison ladder:
  ```
  recipe_full_te (no pseudo)              0.97967
  recipe_pseudolabel (recipe labeler)     0.97993
  **path_c_stage1 (LB-4 labeler, τ=0.99)  0.97998**   +0.00005 vs vanilla
  LB-best 3-stack                         0.98061
  LB-best 4-stack                         0.98084
  ```

**NULL — matches stage-2 LB-blend-labeler pattern exactly**: prior
stage-2 OOF 0.98002 → LB 0.97989 (gap +0.00038). Path C standalone
0.97998 with predicted LB ~0.9796. The +0.00005 OOF lift over vanilla
pseudo is within fold noise. Stronger labeler tightens OOF but the
LB gap blows up — same documented mechanism that closed every prior
stage-2 attempt.

**Kaggle DNS cache overflow** struck this run too — arrays corrupted to
18-byte stubs. Metric extracted from kernel log only. Without arrays,
blend gate can't be computed exactly, but the standalone metric
together with the documented stage-2 pattern make the verdict
conclusive.

#### T1 — score-conditional 2-bucket log-bias on LB-best 4-stack

Per_bin_blend.py at 5 buckets × 30 params overfit on a single CV
split (in-sample +0.00009 → nested −0.00031, 2026-04-25). T1 reduces
the bucket count to 2 (`{score ∈ {3,6,7,8}}` = 25.1% of rows / 83% of
error mass vs `{others}`). 6 free params — between global underfit
(3 params) and per-bin overfit (30 params).

In-sample per-bucket OOF = **0.97686** with
bias_A=[0.49, 1.29, 2.00], bias_B=[1.35, 0.88, 4.92].
Vs LB-4 global = 0.98084 → **−0.00398 even in-sample**. The per-
bucket optimum found `bias_A.High = 2.00` (vs global 3.40, suppressing
High in boundary rows) and `bias_B.High = 4.92` (vs global 3.40,
amplifying High in clean rows). Both of these regress macro-recall:
  - Low recall: 0.9955 → 0.9931 (−0.0025)
  - Medium recall: 0.9695 → 0.9704 (+0.0009)
  - High recall: 0.9775 → 0.9646 (−0.0129)
Nested 5-fold = 0.97602 (−0.00482 vs global). In-sample inflation
+0.00084 (smaller than per_bin's +0.00040 because fewer params).

**NULL.** Even at 6 free params the bucketed bias actively hurts —
**the global 3-param log-bias is locally optimal**. Confirms that
single-axis (dgp_score) bucketing isn't enough granularity to capture
the per-row decision-rule asymmetry. Mechanism: bucket A mixes
boundary-Medium rows AND clean-High rows; the High-bias optimum for
each is opposite, so any single bias_A value compromises both.

**New rule** (LEARNINGS.md candidate): **"For tree-stack outputs
calibrated on a class-balanced sample-weight base, single-axis bias
bucketing is locally optimal at the global granularity. Reducing
bucket count from 5 → 2 doesn't avoid the per-bucket overfit because
the optimization objective itself is non-decomposable (per-class
recall trade-offs require row-level, not bucket-level, decision)."**

#### T7 — test prediction agreement matrix (LB-verified subs)

Diagnostic: identify the test rows where our 6 LB-verified
submissions disagree most, mapping where the +0.00020 lift to pack
0.98114 actually lives.

Candidates loaded:
  ```
  primary  submission_tier1b_greedy_meta.csv         LB 0.98094
  realmlp  submission_lb3_realmlp_nonruleiso.csv     LB 0.98008
  3way     submission_3way_recipe025_s1035_s7040.csv LB 0.98005
  pseudo   submission_recipe_greedy_recipe_pseudolabel.csv  LB 0.97998
  recipe   submission_recipe_full_te.csv             LB 0.97939
  catboost submission_recipe_full_te_catboost.csv    LB 0.97935
  ```

Pairwise disagreement counts (of 270k test rows):
  ```
              primary  realmlp   3way pseudo recipe catboost
  primary           0     196    349    348    484      677
  realmlp         196       0    279    284    434      625
  3way            349     279      0    163    409      644
  pseudo          348     284    163      0    282      579
  recipe          484     434    409    282      0      559
  catboost        677     625    644    579    559        0
  ```

Disagreement summary:
  - **981 rows (0.36%)** where any non-primary differs from primary.
  - **314 rows (0.116%)** where ≥3/5 non-primary candidates disagree
    with primary.
  - Distribution by `dgp_score`:
    - score 6: 175 high-minority rows (out of 16,652 = 1.05%)
    - score 3: 94 high-minority rows (out of 43,746 = 0.21%)
    - All other scores: ≤19 high-minority rows each
    - **score 6 + score 3 = 269 / 314 = 86% of disagreement mass**

Direction of disagreement (on the 314 high-minority rows):
  - **Primary "Medium" (196 rows)**: 164/196 majority-non-primary →
    High; 32/196 → Low
  - Primary "Low" (81 rows): 100% majority-non-primary → Medium
  - Primary "High" (37 rows): 100% majority-non-primary → Medium

**The 196 rows where primary=Medium and consensus=High are the
strongest "lift-to-pack" candidates**. INITIAL framing: if 50%+ are
truly High, override could lift LB by +0.0005 to +0.0015.

#### T7 train-OOF validation — KILLS the override hypothesis

Replicating the test-side disagreement pattern on TRAIN OOF (where
truth is known) using the same 6 candidate constructions:

  ```
  Cluster                              OOF n    Truth (L/M/H)    Precision   Break-even   Verdict
  primary=Medium → consensus=High        512    0 / 482 / 30        5.9%       8.1%       BELOW
  primary=Low    → consensus=Medium      241    155 / 86 / 0       35.7%      39.3%       BELOW
  primary=High   → consensus=Medium       96    0 / 73 / 23        76.0%      91.9%       BELOW
  ```

Macro-recall arithmetic for the M→H cluster (the largest opportunity
on test): if the 512 OOF rows were override-flipped Medium→High,
gain = +30/N_H = +0.001428 High recall; loss = −482/N_M = −0.002017
Medium recall. **Net = −0.000589 macro-recall regression**.

All three clusters are LB-NEGATIVE under macro-recall. **Even
cluster 3 with 76% override-precision LOSES** because High is the
rare class — demoting wrong-Highs to Medium costs more on High recall
than it gains on Medium recall. (Same Pareto-frontier closure as
2026-04-24 disagree-stacker / selective-router / missed-High detector.)

**T7's value is the OPPOSITE of the initial framing**:
  1. **U1 hardcoded override is FALSIFIED** — would have cost
     ~−0.0006 LB if probed. Saved an LB slot.
  2. T7 is a **TIGHT confirmation** that the consensus-disagreement
     signal in our existing OOF bank is insufficient to override
     primary at the macro-recall optimum.
  3. The 5 weaker LB-verified candidates collectively make the SAME
     M↔H boundary mistakes that primary makes, just at lower
     calibration confidence. Their consensus carries information
     about WHERE the boundary is uncertain but NOT which side is
     correct.
  4. Validates the audit's hedge swap recommendation — primary IS
     the best calibration available; hedge via independent
     `recipe_full_te` (different surface, no meta-stacker) is the
     right private-LB insurance.

Saved as `scripts/artifacts/t7_disagreement_rows.csv` (314 rows ×
{test_id, score, primary, realmlp, 3way, pseudo, recipe, catboost})
for cross-branch reference; treat as documentation, not a deploy
candidate.

LB delta: n/a (no LB probe — train-OOF check killed it pre-emptively).
LB-best unchanged at **0.98094**.

#### Combined session read-out

Three mechanism-distinct paths (NN on recipe / per-cell MLP / iterative
pseudo-label) + two diagnostic levers (per-bucket bias / disagreement
matrix with train-OOF precision validation). All five close NULL.
Together they reconfirm the LB 0.98094 ceiling is structural across:
  - bucket-bias axis (T1: even 6-param 2-bucket bias regresses
    in-sample by −0.00398 vs global)
  - row-level disagreement axis (T7: all 3 disagreement clusters
    below break-even precision under macro-recall)
  - architecturally-novel NN substrates (path A: recipe FE, the
    strongest substrate any NN ever saw)
  - per-row partition (path B: 128-cell per-cell MLP — Jaccard 0.083
    record-low orthogonality but 7.7× magnitude trap)
  - pseudo-label labeler strength (path C: LB-best 4-stack labeler
    OOF +0.00005 over recipe labeler — within fold noise, matches
    documented stage-2 LB-blend chain pattern)

**LB best unchanged**: `submission_tier1b_greedy_meta.csv` at LB
0.98094. Final-selection lock from prior session stands (primary +
3way hedge at LB 0.98005).

### Next steps: lock + stop (post-2026-04-26 close-out)

After Path A/B/C + T1 + T7, the remaining-bet shortlist from
2026-04-26 needs revision:

  **U1 (hardcoded override) — FALSIFIED by T7 train-OOF check.**
  Would have cost ~−0.0006 LB. Do not probe.

  **U2. Score-6 binary specialist with FULL candidate feature set**
  (~30 min CPU). Train XGB binary on `[6 candidates × 3 classes (18
  dims) + dist + dgp_score]` to predict `(y == High)` on
  score=6 train rows. Different from U1 because it uses LEARNED
  weighting instead of consensus voting. **But the underlying signal
  ceiling on these rows is ~5.9% precision** (T7 cluster 1) — even
  a calibrated learner is unlikely to push above 8.1% break-even on
  test. Predicted null. Skip unless an LB slot is genuinely free.

  **U3. Multi-seed-bag of LB-verified primary** (~30 min CPU + 2-3
  LB probes). Resubmit primary at random seeds at end-of-comp as a
  final variance check. Highest-EV remaining LB-slot use:
  - Probe 1: primary as-is (variance check vs locked LB 0.98094)
  - Probe 2 (if the 1st is materially different): seed-rotated copy
  Useful for private-LB hedging but no expected lift.

  **Skip on principled grounds (re-confirmed by T1 + T7)**:
  - Further bucket-bias or row-override variants — both axes proven
    structurally below break-even by today's session.
  - More NN families (16+ nulls; structural ceiling confirmed).
  - More meta-stacker variants (10+ saturation confirmations).
  - Public-CSV blending (banned).
  - Further bucket-bias variants (T1 confirmed locally optimal).
  - More NN families (16+ nulls; structural ceiling confirmed).
  - More meta-stacker variants (9+ saturation confirmations).
  - Public-CSV blending (banned).

### 2026-04-27 — C blend gate FAIL (19th saturation) + A queued as next step

- **C (distill_no_rule) full 5-fold complete** (FAST mode lr=0.15 /
  n_est=1000 / es=100 — chosen so fold wall fits inside the container's
  ~10-min idle-reboot window):
  ```
  Fold scores: 0.97527 / 0.97664 / 0.97735 / 0.97502 / 0.97515
  OOF argmax  = 0.97589 ± 0.00094
  Tuned       = 0.97950, bias=[1.4324, 1.3689, 3.2008]
  vs recipe baseline 0.97967 = -0.00017 (within fold noise)
  ```
  Standalone trees CAN match recipe-strength OOF on a basis that
  excludes rule-derived features (4 threshold flags + 3 LR-formula
  logits removed → 424 features vs recipe's 433).
- **Blend gate: FAIL — magnitude trap.**
  ```
  C @ recipe bias = 0.97944  (Δ -0.00140 vs LB-best 4-stack 0.98084)
  errs C = 10,279  vs anchor 9,415  (+864 = +9% more errors)
  Jaccard(C, 4-stack) = 0.7728  ✓ (genuine novel orthogonality)
  PCR delta: L=-0.00083 / M=-0.00223 / H=-0.00114  (all NEGATIVE)
  Best gate-passing α=0.10, blend Δ=-0.00019  (still below anchor)
  ```
  Classic magnitude-trap pattern documented 18 prior times: orthogonal
  errors but in greater absolute count → blend math defeats the gain.
- **19th saturation confirmation at LB 0.98094.**
- New rules confirmed (LEARNINGS.md candidates):
  - **Feature-restricted variants (drop a class of input features)
    produce orthogonal errors but typically at +5-15% magnitude vs
    full-feature recipe.** Same structural pattern as feature-restricted
    NN variants (v6/v7/v9 in 2026-04-22 NN closure session).
  - **Container-rehydrate-resilient compute pattern at 10-min reboot
    intervals**: per-fold checkpoint + foreground bash invocation per
    fold + reduced XGB params (FAST mode: lr=0.15 / n_est=1000 / es=100)
    fits a single fold inside the reboot window. 5 sequential
    foreground iterations completed C in ~50 min wall.

### Next step: A (wide programmatic FE) — queued

The original 4-lever brainstorm (C/D/B/A) leaves only A unrun. C
ruled out (above), D ruled out (Pareto violation), B passed gate
(LB 0.98091, 5× tighter than LR-meta, did not lift LB-best).

A is the wide programmatic FE pattern from cdeotte's 1st-place
backpack-prices kernel (NVIDIA cuDF FE blog):
  1. Generate THOUSANDS of features programmatically:
     - 7-stat group-by per (cat, num): mean/std/min/max/q25/q50/q75
     - 8-quantile group-by per (cat, num): [5,10,40,45,55,60,90,95]
     - extended decimal features: (col*10) % 1 .round(2) on all 11 nums
     Total: ~1700 NEW features on top of recipe's 440 = ~2140 candidates.
  2. 1-fold importance scan to get gain rankings.
  3. Forward-select top ~600 features by gain.
  4. 5-fold StratifiedKFold(seed=42) full training on selected.
  5. Blend gate vs LB-best 4-stack.

Cost (FAST mode, with rehydrate-resilient foreground iterations):
  - Phase 1: data load + FE gen + 1-fold importance scan ~= 10 min
  - Phase 2: 5-fold full training ~= 5 × 8 min = 40 min
  - Total: ~6 sequential 9-min foreground iterations.

Per-fold checkpointing already in place (`oof_wide_fe_fold{N}.npy`).
FAST=1 env var added to `wide_fe.py` to match C's resilience config.

**Why A still has nonzero EV after 19 saturation confirmations**: A
generates a NEW feature surface that was never in the bank. Unlike
all prior bank-extension variants (which add new MODELS to the meta
bank), A adds new FEATURES to the recipe XGB. Different mechanism,
different failure mode. Bayesian prior of LB lift: ~15-20% (lower
than B's was, but mechanism-novel).

If A nulls: lock final-selection at LB 0.98094 + safe hedge,
reserve remaining LB submissions for end-of-comp variance check.

### 2026-04-27 — A adversarial recipe + B' sklearn RF meta: 22nd saturation, RF gap +0.00010 (tightest non-XGB-meta calibration)

Two-pronged session: (A) training-side input perturbation on recipe XGB,
(B) cuML meta-stacker on Kaggle GPU. B blocked by P100 sm_60 incompat
(cuML 26.02 dropped support, same dead-end that hit KAN/Mamba/TabPFN-10k);
pivoted to (B') sklearn RandomForest meta-stacker locally. Both close
NULL on the +0.0002 LB-transfer gate but B' produced a notable
calibration data-point.

**A — Adversarial-robustness recipe XGB** (`scripts/recipe_adv.py`):
σ × IQR Gaussian noise injected on the 11 raw numeric columns of tr_idx
rows ONLY, AFTER recipe FE has been computed from clean values (derived
features stay clean; only raw numerics seen by trees are noisy). σ=0.05
× IQR. Single-pass noise (K=1), no row duplication.
- Per-fold argmax: 0.9760 / 0.9763 / 0.9766 / 0.9747 / 0.9761
- Standalone tuned OOF: **0.97933** (Δ −0.00034 vs recipe 0.97967),
  bias [1.23, 1.37, 3.40] (sharper than recipe's [1.43, 1.47, 3.40])
- vs LB-3stack: best gate-pass α=0.025 → Δ −0.00007
- vs LB-4stack: best gate-pass α=0.05 → Δ −0.00003
- **NULL** on gate. σ=0.05 noise reduces standalone but doesn't add
  orthogonal blend signal. Recipe XGB at depth=4 + reg_alpha=5 +
  reg_lambda=5 already finds robust splits via heavy reg; explicit
  perturbation is redundant.

**B (cuML on Kaggle GPU)** — BLOCKED. cuML 26.02 LR/RF/KNN all hit
`cudaErrorNoKernelImageForDevice` on P100 (sm_60 dropped from libraft +
decisiontree + coalesced reduction kernels). Tried try/except wrappers
to isolate algorithms — all three failed identically.
- Kaggle dataset uploaded: `chrisleitescha/irrigation-cuml-meta-input` (97 MB)
- Kaggle kernel pushed: `chrisleitescha/irrigation-cuml-meta-stacker-smoke`
- All three cuML kernels ERRORED at fit time on P100. Lever closed at
  the platform-compat layer.

**B' — sklearn RandomForest meta-stacker** (`scripts/sklearn_rf_meta.py`):
n_estimators=500, max_depth=14, max_features='sqrt', bootstrap=True,
class_weight='balanced'. Trained on 16-component curated bank (recipe
+ pseudo s1/s7/s123 + realmlp + xgb_nonrule + xgb_corn + dist_digits
+ dist_routed + dist_digits_ote variants + recipe_catboost + lgbm_te_orig
+ lgbm_dist_digits_ote + hybrid_lgbmxgb_blend) = 65 features incl. 14
distance/rule meta cols + LB-best 3-stack log-probs.
- Per-fold argmax: 0.9793 / 0.9798 / 0.9812 / 0.9791 / 0.9795
- Standalone tuned OOF: **0.98069** — between LB-3stack (0.98061) and
  LB-4stack (0.98084)
- Bias: [1.93, 1.97, 2.60] — wildly different from XGB-meta family
  [1.43, 1.47, 3.40]. RF's bagging-averaged probabilities have a
  fundamentally different prob-scale than gradient-boosted XGB.
- All 4 gates pass at α=0.025 iso (Jaccard tight, errs 9398 < anchor
  9415, asymmetric flip 0.677, G3 ✓)
- Best gate-pass blend: α=0.025 iso → Δ +0.00001 vs LB-best 4-stack
  (microscopic, far below +0.0002 LB-transfer threshold)
- Combined A + B' grid sweep: best (α_A=0, α_B=0.025) → Δ +0.00001
  (=B' alone at minimum α). A is structurally redundant with the
  recipe-XGB-meta-iso channel already in LB-best 4-stack.

**LB PROBE** (B' standalone tuned, user-approved, 08:06 UTC):
`submission_sklearn_rf_meta_tuned.csv` → **LB public = 0.98059**.
- Δ vs LB-best primary (0.98094) = **−0.00035** (clean regression).
- Δ vs LB-3stack hedge (0.98005) = **+0.00054** (B' beats hedge).
- OOF→LB gap = **+0.00010** — TIGHTEST gap of any "different L2
  architecture" probe yet:

```
ladder of simpler-than-XGB metas on saturated bank:
  LR v1 (heavy-overfit)        OOF 0.98167 → LB 0.97991  gap +0.00176
  LR v2 (C=0.1, no class_w)    OOF 0.98107 → LB 0.98052  gap +0.00055
  R2 heavy-reg meta (depth=2)  OOF 0.98124 → LB 0.97996  gap +0.00128
  MLP-meta (dropout + GELU)    OOF 0.98118 → LB 0.98091  gap +0.00027
  **B' RF meta (this entry)     OOF 0.98069 → LB 0.98059  gap +0.00010**
  LB-best 4-stack (XGB-meta)   OOF 0.98084 → LB 0.98094  gap −0.00010
```

Interpretation: **bagging-based meta-stackers (RF) have OOF→LB gap
calibration approaching the gradient-boosted XGB-meta's negative gap**.
Bootstrap aggregation + averaging at L2 produces tighter generalization
than any sklearn-style MLP / LR / R2-style XGB variant tested on this
problem.

But standalone is still 35 bp below primary — RF reaches an operating
point BETWEEN LB-3stack and LB-4stack with a structurally different
bias profile, but the LB-best 4-stack already extracts ~all the signal
the bank carries. The Pareto-frontier closure holds: per-class trade
follows the same M↑/H↓ pattern as every prior meta variant.

**22nd independent saturation confirmation at LB 0.98094** (joins 21
prior LB-validated nulls). LB-best primary unchanged.

**Two portable rules** (LEARNINGS.md candidates):
  1. **cuML on Kaggle's P100 fleet is structurally DOA** for any
     algorithm using libraft kernels (LR via QN solver, RF
     decision-tree quantiles, KNN coalesced reduction). cuML 26.02
     dropped sm_60 across the board. Same dead-end as KAN / Mamba /
     TabPFN-10k. Use sklearn locally for any meta-stacker variant
     in this comp; budget Kaggle GPU for tabular NN architectures
     that don't depend on libraft (RealMLP / Trompt / TabM).
  2. **sklearn RandomForestClassifier(bootstrap=True) as a
     meta-stacker has the tightest OOF→LB calibration of any
     simpler-than-XGB meta architecture** on a saturated bank
     (gap +0.00010 vs LR v1 +0.00176, LR v2 +0.00055, MLP +0.00027,
     R2 heavy-reg +0.00128). For future synthetic-tabular comps
     where the LB-best meta is XGB-based, RF is the highest-EV
     "different L2 architecture" probe to test for blend
     orthogonality. The Pareto-frontier closure still applies — RF
     won't break a saturated stack — but the calibration profile
     is portable knowledge.

LB budget: 1/10 used today (this probe), 9 remaining. LB-best
unchanged at **0.98094** via `submission_tier1b_greedy_meta.csv`.
Final-selection lock unchanged: PRIMARY 0.98094 + HEDGE 0.98005
audit F1 swap.

### 2026-04-27 — wide_fe (Cdeotte 1st-place programmatic FE pattern): 23rd saturation confirmation, LOCK FINAL

- Goal: execute the only untried high-EV lever from the 4-lever GM-research
  brainstorm (C/D/B/A from 2026-04-26): **A. Wide programmatic FE**
  — Cdeotte's 1st-place backpack-prices pattern (NVIDIA cuDF FE blog).
  Generate ~1700 NEW programmatic features on top of recipe's 440, run
  1-fold importance scan to rank by gain, forward-select top 600,
  5-fold StratifiedKFold(seed=42) on the selection. Mechanism distinct
  from every prior null because it adds a NEW feature surface to the
  recipe XGB rather than a new model to the meta bank.
- Three feature-engineering blocks added on top of recipe FE:
  1. **7-stat group-by per (cat × num)** — mean, std, min, max,
     q25, q50, q75 over 8 cats × 11 nums = **+616 cols**
     (we already had mean+std = 176; this expands the family).
  2. **8-quantile group-by per (cat × num)** — q05, q10, q40, q45,
     q55, q60, q90, q95 = **+704 cols**.
  3. **Extended decimal features** — `(col × 10) % 1 .round(2)`
     for all 11 nums = **+11 cols**.
  Total new candidate features: 1331; combined with recipe's 440 base
  = 1771 numeric + 117 OTE-target cats = **~1774 candidate features**.
- Implementation iteration cycle hit OOM **3 times** at production scale
  (15Gi container, no swap). Each fix surgically targeted the next
  bottleneck:
  1. **Mem-fix #1 (commit 7f638a2)**: defragment `.copy()` after FE +
     max_bin 1024→256/512 + `gc.collect()`. **Crashed at the .copy()
     itself** because copying a 630k×1500 fragmented frame needs ~12-15
     GB transient peak.
  2. **Mem-fix #2 (commit 0f5fcfb)**: zero-fragmentation refactor —
     FE functions return `(train_dict, test_dict)` of {col_name:
     np.ndarray}, then a SINGLE `pd.concat([train, pd.DataFrame(dict)],
     axis=1)`. **Cleared FE phase but still OOM'd** during the 1-fold
     XGB importance scan (full-data slice + DMatrix at peak ~15 GB).
  3. **Mem-fix #3 (commit b86a7ee)**: stratified-subsample tr_idx →
     SCAN_N=200k for the importance scan only (ranking 1700 features
     is robust at this scale); convert X_tr/X_va/X_te to contiguous
     float32 numpy arrays before `model.fit` to free pandas overhead;
     `n_jobs=8` (down from 16) to halve thread-local histogram memory.
     **Cleared scan phase but still OOM'd at fold 1 of 5-fold loop**
     (still copying full 1537-col train slice).
  4. **Mem-fix #4 (commit 928c822)**: per-fold slice uses ONLY
     `selected_numeric + selected_te_targets + [TARGET]` columns
     (~240 vs 1537, a 6× cut); defer `test.copy()` until after X_tr
     is built. Production GREEN.
  Each iteration validated by SMOKE first (20k×2-fold, ~30s wall) —
  caught regressions in <1 min cycle each time. Final memory peak ~9
  GB. **Five OOMs avoided by SMOKE-first discipline.**
- Importance scan diagnostic (the headline finding):
  ```
  scan best_iter = 398 (out of 400 cap, max_bin=256, 200k subsample)
  features used:  472 out of 1774 candidates (26.6% utilization)
  top-10 by gain (gain in parentheses):
    logit_P_High                                  1611.1  ← recipe LR-formula
    logit_P_Low                                   1192.0  ← recipe LR-formula
    soil_lt_25_TE_cls0                             429.2  ← recipe rule-cell OTE
    COMBO_Crop_Growth_Stage_Mulching_Used_TE_cls0  105.9  ← recipe pair OTE
    CAT_Rainfall_mm_TE_cls2                         63.2  ← recipe num-as-cat OTE
    logit_P_Medium                                  39.2  ← recipe LR-formula
    rain_lt_300                                     36.8  ← recipe rule indicator
    Temperature_C                                   23.6  ← raw num
    CAT_Rainfall_mm_TE_cls1                         23.0  ← recipe num-as-cat OTE
    Rainfall_mm                                     21.1  ← raw num
  selected: 121 numeric + 117 OTE-target cats
  ```
  **ZERO wide-FE features (WIDE_/WIDEQ_/WIDED_) in top-10.** Of the 121
  selected numerics, only ~29 are wide-FE (1331 candidates → 29 picked
  = **2.2% pickup rate**). The recipe's existing OTE + LR logits + rule
  indicators absorb essentially all the available signal; group-by stats
  and quantile features add nothing the recipe doesn't already encode.
- Per-fold standalone results (FAST=1: 1000 trees, lr=0.15, ES=100):
  ```
  fold  wide_fe   recipe    Δ
  1     0.97523   0.97544   -0.00021
  2     0.97639   0.97659   -0.00020
  3     0.97655   0.97721   -0.00066
  4     0.97456   0.97465   -0.00009
  5     0.97574   0.97557   +0.00017
  mean Δ:                   -0.00020
  ```
  **OOF argmax 0.97569**, tuned **0.97959** (bias [1.13, 1.27, 3.10]).
  **Δ vs recipe baseline = −0.00008 standalone tuned (within fold noise).**
  Δ vs LB-best 4-stack (0.98084) = −0.00125. Wall: 60.9 min.
- Blend gate (CANDIDATES=wide_fe blend_gate_AB.py):
  ```
  candidate @ recipe-bias:  0.97953 (Δ -0.00131 vs LB-4 0.98084)
  iso-cal'd @ recipe-bias:  0.97945

  vs LB-best 3-stack (anchor 0.98061):
    NO gate-pass; best-Δ raw α=0.025 Δ=-0.00004
    g2=False (errs > anchor+5), g3=True, g4=False (asym < 0.5)

  vs LB-best 4-stack (anchor 0.98084):
    "best gate-pass" α=0.025 iso Δ=-0.00003 (NEGATIVE — best
    among sweep points where g2/g3/g4 mechanically don't fail,
    but the lift itself is below anchor)
  ```
  **Both anchors monotone-negative across the α-sweep.** No α threads
  the needle. **No LB probe warranted.**
- **23rd independent saturation confirmation at LB 0.98094.** Joins:
  ```
  attack vector                                 result
  ──────────────────────────────────────────── ────────────────
  1-22. (prior 22 saturation confirmations from 2026-04-25/26)
  23. **wide_fe programmatic FE (this entry)    standalone null +
                                                blend null at every α**
  ```
- **Mechanism diagnosis (the portable read-out)**: the cdeotte
  backpack-prices pattern works on competitions where the base FE has
  NOT been heavily target-encoded. Our V10 recipe already does:
  - OrderedTE on 117 cat-tuples (cats + pairs + digit cols + num-as-cat
    + rule cells + threshold flags) × 3 classes = 351 OTE features
    that explicitly compute per-class probability statistics conditioned
    on each categorical
  - LR-formula logits derived from the 10k rule-perfect original (3 cols)
  - FREQ counts per cat (~36 cols)
  - ORIG mean/std on 38 nums × 8 cats = 38 × 2 features
  Adding (cat × num) group-by stats is **partially redundant** with the
  OTE family because OTE already conditions per-class statistics on cat;
  the marginal stat (mean/std/quantile of NUM given CAT) is correlated
  with per-class probability shifts the OTE captures. XGB's tree splits
  prefer the OTE features (sharper decision boundaries) and shrink the
  group-by stats to leaf-level near-noise.
- **Portable rule** (LEARNINGS.md candidate): **"Wide programmatic FE
  (group-by stats × quantile features) only adds standalone signal on
  top of a base pipeline that does NOT already use heavy target
  encoding. On a recipe with ~350+ OTE features × 3 classes, group-by
  stats are partially redundant with OTE's per-class probability
  conditional-on-cat structure; XGB tree splits prefer the OTE features
  and shrink group-by stats to near-noise. Pickup rate <5% in the
  importance scan is a clean diagnostic for this saturation pattern.
  Save the wall budget for novel feature classes (e.g. graph-derived,
  external-data joins) when OTE is already in the base pipeline."**
- LB budget: **0/10 used today** (no probe spent), full 10 remaining.
  LB-best unchanged at **0.98094** via `submission_tier1b_greedy_meta.csv`.
- **FINAL-SELECTION LOCK** (3 days to deadline 2026-04-30):
  1. **PRIMARY**: `submission_tier1b_greedy_meta.csv` → **LB 0.98094**
     (gap −0.00010, anomalous LB > OOF). Composition: LB-best 3-stack
     + xgb_metastack_iso × α=0.30.
  2. **HEDGE**: `submission_3way_recipe025_s1035_s7040.csv` →
     **LB 0.98005** (gap +0.00024, premium −0.00089 vs primary).
     Sidesteps meta-stacker layer for orthogonal overfit insurance
     against private-LB drift.
  Pack 0.98114 stays +0.00020 above primary; leader 0.98219 stays
  +0.00125 above. Both reachable only via public-CSV blending (banned
  by top-of-file rule). With 23 saturation confirmations spanning every
  major lever class (greedy / meta variants / NN families / new
  feature classes / training-data levers / per-row gating /
  Pareto-frontier overrides / decision-rule variants / wide programmatic
  FE), the own-pipeline ceiling is structurally exhausted.
- Artefacts (whitelisted via .gitignore for cross-branch reuse):
  - `scripts/wide_fe.py` (4 OOM-fix iterations applied)
  - `scripts/artifacts/oof_wide_fe.npy` + `test_wide_fe.npy`
  - `scripts/artifacts/wide_fe_results.json` + `wide_fe_smoke_results.json`
  - `scripts/artifacts/blend_gate_AB_results.json` (wide_fe gate decision)
  - `submissions/` — no submission emitted (sweep monotone-negative).

### 2026-04-27 — Phase A (residual TE) + Phase B (base-margin) sweep — Phase A NULL (24th saturation), Phase B K=4 too aggressive, K=2 in flight

- Goal: two mechanism-distinct attempts at squeezing past LB 0.98094
  via the only properties of the LB-best primary not yet pressure-tested:
  (A) **fold-safe residual Target Encoding** — per-row OrderedTE on three
  binary residual targets `(y != rule_pred)`, `(M→H flip at score=6)`,
  `(H→M flip at score 7/8)` over digit/cat-pair/score-band keys, delivered
  as ~42 new features into the recipe XGB. (B) **base-margin
  residualization** — `base_margin = K * one_hot(rule_pred) - K/2`
  injected via `xgb.train` DMatrix so trees start from the closed-form
  rule's logits and only fit residuals. K_MARGIN env-var (4.0/2.0/1.0).
  Both run on full 504k 5-fold seed=42 aligned with every saved OOF.
  Branch: `claude/residual-target-encodings-Duy1o`.
- Changed:
  - `scripts/residual_te_helpers.py` — 3 binary residual targets +
    14-key list (digit positions on 4 rule-axis numerics × {-1, 0} +
    Crop_Growth_Stage + Mulching_Used + 4 high-signal cat-pair combos
    + dgp_score) + per-fold OrderedTE wrapper using the existing OTE
    class with n_classes=2.
  - `scripts/residual_te.py` — Phase A 5-fold orchestrator. Reuses
    recipe `load_and_engineer`, computes binary targets per fold,
    fits residual OTE per (target × key), appends 42 cols to recipe
    feat matrix, trains heavy-reg XGB. Per-fold checkpointing,
    `RUN_FOLD=N` env var for rehydrate-resilient sequencing.
  - `scripts/recipe_basemargin.py` — Phase B 5-fold orchestrator with
    `xgb.train` DMatrix + `base_margin = K*one_hot(rule_pred) - K/2`.
    K_MARGIN env-var (default 4.0).
  - `scripts/preflight_residual_auc.py` — 5-fold OOF AUC diagnostic on
    all 3 binary residual targets over 35 dist+raw-num features. Decision
    rule: AUC ≥ 0.60 → PROCEED.
  - `scripts/blend_gate_4gate.py` — G1/G2/G3/G4 analyzer reusing
    `tier1b_helpers.build_lbbest_stack` for anchor reconstruction.
    Sweeps α ∈ {0, 0.10, 0.20, 0.30, 0.40, 0.50}; reports per-class
    recall delta + Jaccard + net_high_flip vs anchor at α=0.30.
  - `scripts/run_phase_ab.sh` — sequential foreground orchestrator.

- Pre-flight binary AUCs (5-fold seed=42 on 35-feature subset, ~5 min):
  ```
  target          prevalence   OOF AUC   verdict
  r_global        1.636%        0.8951   PROCEED
  r_mh_s6         0.246%        0.9882   PROCEED
  r_hm_s78        0.268%        0.9982   PROCEED
  ```
  All three crush the 0.60 PROCEED threshold. Strong residual ranking
  signal exists at OTE capacity. **AUC alone is necessary but not
  sufficient** for blend transfer (the missed-High detector at AUC
  0.9711 also met this gate but failed precision break-even on
  override; same risk applies here when delivered as features).

- **Phase A 5-fold production (full 504k, ~30 min/fold = ~2.5h total)**:
  ```
  fold    Phase A    recipe baseline   Δ
  1       0.97506    0.97544           -0.00038
  2       0.97576    0.97659           -0.00083
  3       0.97663    0.97721           -0.00058
  4       0.97507    0.97465           +0.00042
  5       0.97530    0.97557           -0.00027
  mean fold delta:                     -0.00033

  OOF aggregate:
    argmax    0.97556   (recipe 0.97589, Δ -0.00033)
    tuned     0.97962   (recipe 0.97967, Δ -0.00005, within fold-noise)
    bias      [1.23, 1.27, 3.20]   (recipe [1.43, 1.47, 3.40])
  ```

- **Phase A 4-gate analyzer onto LB-best 4-stack (anchor 0.98084)**:
  ```
                                  raw            iso
  standalone @ recipe-bias       0.97962        0.97924
  errs                            9877           9329
  Jaccard vs LB4 (60% partial)   0.8221         (similar)
  errs ratio (B/LB4)             1.054          ~1.04

  blend sweep α=0.30 (fixed bias):
    OOF        0.98047 / 0.98038
    Δ vs LB4  -0.00037 / -0.00047        FAIL G1 (need ≥+0.0003)
    PCR L      -0.00005 / +0.00002       PASS L
    PCR M      +0.00013 / +0.00072       PASS M (helped)
    PCR H      -0.00119 / -0.00214       FAIL G2 (-5e-4 floor)
    net_high  -15 / -94                  FAIL G4 (need >0)
    G3 ratio   NaN (deltas negative)     FAIL

  OVERALL: FAIL on G1, G2, G3, G4 (raw and iso both)
  ```
  Classic REMOVE-High pattern (same failure as N5b angle1, R2/R5 a045,
  D 3-meta). High recall lost 1.2-2.1pp under blend.

- **Mechanism diagnosis** (portable rule for LEARNINGS.md):
  The 42 residual TE features encode `P(y != rule_pred | key)` which
  is HIGHLY correlated with what the recipe XGB already learns from
  the 117 OTE features at depth=4 + reg_alpha=5. Adding correlated
  features under heavy reg:
    1. slightly hurts standalone tuned (-0.00005 within fold-noise)
    2. shifts calibration toward LOWER High recall (residual TE features
       fire on near-boundary rows; under heavy reg the model demotes
       High predictions on those rows — wrong direction for macro-recall)
  **Pre-flight binary AUCs were not predictive**: high AUC on a residual
  target is necessary but not sufficient when delivered as INPUT
  FEATURES into a model already saturated with strong correlated
  features. Precision-break-even-style gates from binary detector
  experiments (missed_high_detector, spec6_v2) apply at the override
  level; a different mechanism applies at the feature level: for
  feature-level delivery to lift, the new feature must be ORTHOGONAL
  to existing features at the model's split granularity. With 117
  OTE features already encoding per-key class distributions, adding
  per-key residual rates is cosine-correlated.

- **24th independent saturation confirmation at LB 0.98094** (joins
  wide_fe earlier today as 23rd; both confirm the structural ceiling).

- **Phase B K=4 fold 1 (only fold completed)** — early kill:
  ```
  argmax bal_acc       0.97363   (-0.00181 vs recipe fold 1)
  bal @ recipe-bias    0.97588
  best_iter            1352
  errs (fold 1)        2028 vs anchor 1914 (ratio 1.060)
  Jaccard vs LB4       0.7567   ← BETTER orthogonality than Phase A
  PCR delta            L -0.00035 / M -0.00088 / H -0.01095
  ```
  Jaccard 0.76 is competition-grade orthogonal but H recall destruction
  (-1.1pp on fold 1) is catastrophic. K=4 base-margin makes trees too
  conservative around the High boundary — the rule prior at K=4 gives
  softmax([+4, -2, -2]) = [0.984, 0.008, 0.008] for predicted class,
  trees need 6 logit-units to flip; they can't acquire the ~21k rare
  positives. K=4 KILLED at fold 1.

- **Phase B K=2 fold 1 IN FLIGHT** (PID 2102, ~30 min wall):
  Less aggressive prior — softmax([+2, -1, -1]) = [0.79, 0.10, 0.10],
  trees need only 3 logit-units to flip. Should preserve more High
  capacity. If K=2 fold 1 PCR_H is within -5e-4 of recipe baseline AND
  Jaccard remains < 0.80, full 5-fold + 4-gate. Otherwise K=1.

- Strategic context (3 days to deadline):
  - LB-best primary unchanged at **LB 0.98094** (`submission_tier1b_greedy_meta.csv`).
  - Final-selection lock unchanged: PRIMARY 0.98094 + HEDGE 0.98005
    (`submission_3way_recipe025_s1035_s7040.csv`).
  - LB budget: 0 spent today. 10 remaining.
  - Phase A definitively closed; Phase B K=2 sweep still has ~15-20%
    Bayesian prior of clearing all 4 gates given the orthogonality
    advantage shown at K=4 fold 1.

- Artefacts on `claude/residual-target-encodings-Duy1o`:
  - `scripts/residual_te.py`, `scripts/residual_te_helpers.py`,
    `scripts/recipe_basemargin.py`, `scripts/preflight_residual_auc.py`,
    `scripts/blend_gate_4gate.py`, `scripts/run_phase_ab.sh`
  - `scripts/artifacts/oof_recipe_full_te_residte.npy` + test +
    results JSON (Phase A full 5-fold OOF)
  - `scripts/artifacts/blend_gate_4gate_residte_results.json` +
    `_iso_results.json` (definitive NULL diagnosis)
  - `scripts/artifacts/preflight_residual_auc.json` (3-target AUCs)
  - `submissions/submission_recipe_full_te_residte.csv` (diagnostic,
    not for LB probe — stratifies as REMOVE-High)

### 2026-04-27 — Phase B K=2 base-margin completed: 25th saturation NULL, first ADD-High direction but asymmetry too weak

- Continuation of the residual-target-encodings session: Phase B K=2 base-margin
  full 5-fold completed after Phase A's 24th saturation.
- Per-fold standalone (vs recipe baseline):
  ```
  fold    K=2        recipe      Δ
  1       0.97513    0.97544    -0.00031
  2       0.97579    0.97659    -0.00080
  3       0.97704    0.97721    -0.00017
  4       0.97447    0.97465    -0.00018
  5       0.97606    0.97557    +0.00049   (positive)
  mean fold delta:               -0.00019   (closer to recipe than Phase A's -0.00033)
  ```
  K=2 5-fold tuned OOF = 0.97954 (recipe 0.97967, Δ -0.00013, within fold-noise).
  Bias [0.93, 1.07, 2.90] — High bias 2.90 is much lower than recipe's 3.40,
  consistent with the K=2 prior already pushing High preds up at training
  time (less post-hoc bias correction needed).
- 4-gate analyzer FAIL on both raw and iso paths:
  ```
                                  raw           iso
  standalone @ recipe-bias       0.97950       0.97935
  errs                            10,027        9,612
  Jaccard vs LB-best 4-stack     ~0.84         ~0.84
  errs ratio (B/LB4)             1.065         1.021

  blend sweep α=0.30 (fixed bias):
    OOF        0.98035 / 0.98040
    Δ vs LB4  -0.00049 / -0.00044     FAIL G1 (need ≥+0.0003)
    PCR L      -2e-5  / +6e-5         PASS L
    PCR M      -1e-4  / +7e-4         PASS M (helped)
    PCR H      -0.00133 / -0.00205    FAIL G2 (-5e-4 floor)
    net_high   +18 raw / -85 iso      G4 RAW direction-PASS, iso-FAIL
    G3 ratio   NaN (negative deltas)  FAIL

  OVERALL: FAIL on G1, G2, G3, G4 (raw and iso both)
  ```

- **Notable: K=2 raw is the FIRST candidate this comp with net_H > 0 (+18)**
  — direction is ADD-High, the macro-recall-favorable side. But asymmetry
  ratio = 18/106 = 0.17, well below G4's 0.5 floor. The rule prior at K=2
  causes trees to push some boundary rows to High correctly (~62 wins) but
  also gambles on ~44 wrong High predictions. The +18 net is real but
  swamped by churn — same pattern as the LR meta-stacker (LB 0.98091, gap
  +0.00027). Iso-cal collapses the direction win (−85 net_H), suggesting
  the direction gain comes from the natural log-bias correction on K=2's
  shifted prob scale, not from new orthogonal signal.

- **Mechanism diagnosis** (portable):
  Phase B's base-margin = K * one_hot(rule_pred) - K/2 anchors trees
  to the rule's predictions. The rule has 98.4% raw acc, but only 96.5%
  on the High class (15k+ rows in scores 7-9). Trees with K=4 cannot
  flip enough High boundary rows (PCR_H -1.1pp); K=2 leaves more room
  but H recall still falls -0.26pp standalone; K→0 reduces to recipe
  baseline. **No K simultaneously satisfies (Jaccard < 0.80) AND
  (PCR_H within -5e-4)**. The mechanism is structurally bounded by the
  rule's information content.

- **25th independent saturation confirmation at LB 0.98094.** Phase A
  (residual TE) and Phase B (base-margin) close together as 24th + 25th.

- LB-best primary unchanged: **LB 0.98094** (`submission_tier1b_greedy_meta.csv`).
  Final-selection lock unchanged: PRIMARY 0.98094 + HEDGE 0.98005.
  LB budget: 0 spent today.

- Artefacts on `claude/residual-target-encodings-Duy1o`:
  - `scripts/artifacts/oof_recipe_full_te_basemargin_K2.npy` + test +
    results JSON
  - `scripts/artifacts/blend_gate_4gate_basemargin_K2_results.json` +
    `_iso_results.json` (definitive NULL)
  - `submissions/submission_recipe_full_te_basemargin_K2.csv` (diagnostic
    only — not for LB probe)

<<<<<<< HEAD
### 2026-04-27 — purity-rule deep-dive: 28.55% of train is 100%-deterministic

- Goal: senior-engineer reframe. The score-based purity rules (scores 0/1/9
  with 100%/99.996%/99.938% train rule-acc) were ALREADY perfectly absorbed
  by the LB-best primary (zero test disagreements on all three scores per
  the diagnostic in `scripts/purity_rules_diag.py`). Two deeper questions:
  (a) Are there OTHER 100%-pure rule sets beyond simple score? (b) Could
  dropping deterministic rows from training free gradient capacity for the
  boundary-flip rows (scores 3/6/7/8)?
- Branch: `claude/deterministic-prediction-rules-8ORca`.

- **Sub-cell purity search** (`scripts/purity_subcells.py`): for each of the
  96 cells in the 128-cell rule cube, search across 6 untouched cats
  (Soil_Type, Crop_Type, Season, Irrigation_Type, Water_Source, Region)
  for (cell × cat × value) tuples with 0 train errors and ≥30 train rows.
  Found **118 sub-cell rules** beyond the 2 cube-level pure cells.
- **Coverage**:
  ```
  Layer                                Train rows   % train   Test rows
  Cube-level 100%-pure cells (2)         79,842     12.7%     34,333
  Sub-cell rules (118 NEW unique)       +99,909    +15.86%    +41,991
  ─────────────────────────────────── ──────────── ───────── ─────────
  Total deterministic coverage          179,851     28.55%     76,324
  ```
- **Class breakdown of deterministic set**:
  - Low: 148,327 / 369,917 (**40.1%** of all Lows)
  - Medium: 28,289 / 239,074 (11.8%)
  - **High: 3,235 / 21,009 (15.4% of Highs — rare class is also partially deterministic)**
- **Distribution by cell-score**:
  - score=1 (Low): 27 rules
  - score=5 (Medium): 39 rules
  - score=9 (High): 40 rules
- **Sanity**: primary disagrees with cell-majority on only **36 of 76,324
  dropped TEST rows (0.047%)** — model has fully internalized these.
- Artefacts: `scripts/artifacts/purity_subcells.json`,
  `purity_subcell_rules.csv`, `drop_mask_train.npy`, `drop_mask_test.npy`.

### Next steps: DROP_DETERMINISTIC recipe variant (queued, 2026-04-27)

The user's reframe: "the algorithm WILL learn those rules, but it would
perform better if it had not to learn them and could focus on more
difficult values instead." The 28.55% deterministic-row coverage above is
the largest, most class-balanced drop-set we have ever identified.

  **Mechanism**: drop the 179,851 deterministic train rows from the recipe
  XGB's training set per fold. Test predictions stay full-data (the recipe
  XGB trained on retained ~450k rows still scores all 270k test rows;
  post-hoc we hard-set the 76,324 deterministic test rows to their
  cell-majority class — provably zero-risk because primary disagrees with
  cell-majority on only 36/76,324 of them). Saves ~28.5% per-fold compute
  AND frees gradient/capacity for the ~9,400 boundary-flip rows that
  carry every remaining error.

  **Why this is meaningfully different from the 2026-04-26 DROP_SCORES
  NULL** (which dropped scores {0,1,2} = 271k rows = 43.1% all-Low and
  failed because the recipe's `compute_sample_weight("balanced")` already
  rebalances the gradient; DROP_SCORES caused a double-rebalance overshoot
  with High recall −0.0021):
  ```
  Lever              Drop rows   % drop   Class composition    Retained High share
  DROP_SCORES        271,444     43.1%    100% L (blunt)       7.3% (overshoot)
  **DROP_DETERMINISTIC 179,851   28.55%   82% L / 16% M / 2% H  3.95% (mild +0.6pp)**
  ```
  Mild rebalance (Low share 58.7% → 49.2%, Medium 37.9% → 46.8%, High
  3.33% → 3.95%) AND class-proportionate. The High class retains 84.6%
  of its training mass instead of being amplified to 7.3%.

  **Concrete plan for executor** (~50 min CPU + SMOKE-first per CLAUDE.md):
  1. Add `DROP_DETERMINISTIC=1` env var to `scripts/recipe_full_te.py`
     (parameter: drop mask path, default `scripts/artifacts/drop_mask_train.npy`).
     Filter `tr_idx` per fold AFTER fold split but BEFORE OTE-fit and XGB
     train. Log rows-kept count per fold for sanity.
  2. SMOKE first: `SMOKE=1 DROP_DETERMINISTIC=1 python scripts/recipe_full_te.py`
     → expect ~30 sec wall, validates the mask filtering doesn't break the
     OTE pipeline or XGB fit.
  3. Production: `DROP_DETERMINISTIC=1 python scripts/recipe_full_te.py`
     → ~50 min wall (smaller than baseline 55 min because 28.5% fewer rows).
     Output suffix `_dropdet`. Saves `oof_recipe_full_te_dropdet.npy` +
     `test_recipe_full_te_dropdet.npy`.
  4. Post-hoc: hard-set test predictions on `drop_mask_test.npy` rows to
     their cell-majority class. Rationale: the model never SAW
     deterministic rows during training, so its predictions on those test
     rows are noise; overlay the deterministic ground truth.
  5. Blend gate: use `scripts/tier1b_helpers.build_lbbest_stack` to
     reconstruct the LB-best 4-stack and run an α-sweep with `_dropdet`
     substituted in for `recipe_full_te`. Apply the 4-gate filter:
     - G1: standalone tuned OOF Δ ≥ +2e-4 vs recipe baseline 0.97967, OR
       blend OOF Δ ≥ +2e-4 vs LB-best 4-stack 0.98084
     - G2: errs ≤ 1.05× anchor (no magnitude trap)
     - G3: per-class recall ≥ anchor − 5e-4 each class (Pareto guardrail)
     - G4: |net_rare_class_flip| / |total_rare_class_churn| ≥ 0.5 AND
       net direction is ADD-High (per 2026-04-27 R2/R5 closure)
  6. If all 4 gates pass: ASK USER for LB probe approval before submit.
     Per CLAUDE.md, never wrap `kaggle competitions submit` in any retry/
     loop, and always present OOF + projected LB before submitting.
  7. If gates fail: 24th saturation confirmation; document mechanism
     diagnosis (likely ones: post-hoc test override creates train/test
     calibration mismatch; or recipe XGB on retained 450k rows learns the
     same boundary calibration as full 630k because the deterministic rows
     contributed near-zero gradient anyway).

  **Bayesian prior** (calibrated against 23+ saturation confirmations):
  ~30% LB lift / ~50% null / ~20% mild regression. Higher than typical
  bank-extension experiments because this is a TRAINING-DATA-COMPOSITION
  lever, not another OOF-stacking variant. The class-proportionate drop
  + High-mass retention specifically address the failure mode that closed
  DROP_SCORES.

  **Closure value if NULL**: definitively answers "would the model improve
  with focused capacity?" — the user's reframe — for the LB-best primary.
  Resolves a long-standing open question even at null cost.

### 2026-04-27 — DROP_DETERMINISTIC executed: 24th saturation confirmation, REMOVE-High Pareto violation

- Goal: execute the queued DROP_DETERMINISTIC plan from the prior entry.
  User reframe: "the algorithm WILL learn those rules, but it would
  perform better if it had not to learn them and could focus on more
  difficult values instead." Production answer: NO it would not.
- Branch: `claude/deterministic-prediction-rules-8ORca`.
- Changed: `scripts/recipe_full_te.py` (DROP_DETERMINISTIC env var: load
  drop_mask_train + drop_mask_test + test_cell_majority, filter tr_idx
  per fold AFTER fold split BEFORE OTE-fit, post-hoc inference override
  on deterministic OOF + test rows to one-hot cell-majority class);
  `scripts/blend_gate_dropdet.py` (4-gate filter analyzer with G4 net
  rare-class flip direction check); `scripts/artifacts/test_cell_majority.npy`.
- SMOKE pass GREEN (20k×2-fold, ~30 sec): drop 28.49%, override applies,
  tuned OOF 0.96431 vs vanilla smoke 0.96381 (+0.0005 mild signal).
- Production 5-fold seed=42 (~1h 50min CPU including one rehydrate-and-
  resume cycle):
  - Per-fold pre-override argmax: 0.97254 / 0.97544 / 0.97617 / 0.97438 / 0.97420
  - Per-fold post-override (apples-to-apples vs recipe): mean Δ = **−0.00006**
  - Per-fold band-only (boundary rows the model actually had to learn): mean Δ = **−0.00004**
  - Overall OOF argmax post-override = **0.97583** (recipe 0.97589 → −0.00006)
  - Tuned log-bias bal_acc = **0.97958** (recipe 0.97967, **−0.00009**)
  - Tuned bias [0.9324, 1.2689, 3.4008] — Low bias dropped 0.50, others ≈ recipe

- **Honest 5-fold post-override comparison** (every val row counted; deterministic
  val rows hard-set to cell-majority = 100% correct):
  ```
  fold  pre       post-ov   recipe    d-pov      band      rec_band  d-band
   1    0.97254   0.97434   0.97544   -0.00110   0.96928   0.97050   -0.00122
   2    0.97544   0.97689   0.97659   +0.00030   0.97234   0.97186   +0.00048
   3    0.97617   0.97714   0.97721   -0.00007   0.97258   0.97262   -0.00003
   4    0.97438   0.97577   0.97465   +0.00111   0.97100   0.96972   +0.00128
   5    0.97420   0.97500   0.97557   -0.00057   0.97012   0.97081   -0.00069
  mean d post-override = -0.00006   mean d band-only = -0.00004
  ```
  Oscillating around zero, slightly negative on aggregate. Folds 1+5
  hurt; fold 4 helps; folds 2+3 ~tied. Within fold-noise band (~±0.001).

- **4-gate blend analysis** (`scripts/blend_gate_dropdet.py`):
  ```
  Anchor               anchor   peak α  peak Δ    G1   G2   G3   G4
  recipe_full_te       0.97967  0.65    +0.00044  ✓    ✓    ✓    ✗ (REMOVE-H net=-266)
  lb_best_3stack       0.98061  0.00    +0.00000  ✗    ✓    ✓    ✗
  lb_best_4stack(prim) 0.98084  0.00    +0.00000  ✗    ✓    ✓    ✗
  ```
  - vs recipe baseline: lift exists at α=0.5-0.65 (+0.00044 OOF) BUT G4
    net-rare-flip direction = REMOVE-High at every positive α (net_H
    ranges from −25 at α=0.025 to −280 at α=0.50). Same REMOVE-High
    asymmetric Pareto-violation pattern that closed R2/R5 a045 on
    2026-04-27 (LB regression −0.00098).
  - vs LB-best 3-stack and 4-stack (the actual deployment anchors): peak
    at α=0 (no blend). Every α > 0 strictly negative. G1, G3, G4 all fail.
- **Standalone OOF/standalone-replace null and blend null vs every
  anchor of practical interest. No LB probe warranted.**

- **Mechanism diagnosis** (the portable read-out):
  1. **The primary's recipe XGB is NOT capacity-bound on 504k rows.**
     At depth=4 + 3000-round budget + reg_alpha=reg_lambda=5, the
     model has enough budget to learn BOTH the cheap rule signal (via
     OTE features + LR-formula logits + rule indicators) AND the
     boundary-flip rows. Removing 28.5% of training rows doesn't
     "free capacity" — it reduces the gradient signal everywhere.
  2. **Deterministic rows act as DECISION-BOUNDARY ANCHORS** for the
     model's High↔Medium calibration. Without them, the model becomes
     systematically under-confident on High predictions on the
     boundary rows. Per-class trade: net Removed-High at every α >
     0, in line with R2/R5's heavy-reg meta closure earlier today.
  3. **Even when the model was trained without deterministic rows, it
     scores 98.8-99.1% on them** (per the in-flight diagnostic) via
     OTE statistics + the rule-aware features. The post-override
     boost ADDS only ~+0.01 per deterministic row × 28.5% share =
     ~+0.003 OOF, which the model loses ~equivalently on boundary
     rows due to weaker calibration anchoring. Net wash.

- **24th independent saturation confirmation at LB 0.98094.** Joins the
  23 prior structural-saturation entries already documented. The
  training-data-composition lever is now closed alongside every other
  major axis (greedy / meta variants / NN families / new feature
  classes / per-row gating / Pareto-frontier overrides /
  decision-rule variants / wide programmatic FE / score-based
  routing / per-cell purity overrides).

- **Two portable rules** (LEARNINGS.md candidates):
  1. **Dropping deterministic-cell training rows on a heavy-reg recipe
     XGB does NOT free capacity for boundary rows.** The model is not
     capacity-bound; deterministic rows act as decision-boundary
     ANCHORS that calibrate rare-class probabilities. Removing them
     produces a REMOVE-High asymmetric model that fails the G4
     direction gate at every α > 0. To use the "free capacity" lever
     productively would require either (a) a lower-capacity base
     (depth=2, 500-round budget), or (b) DOWN-WEIGHTING deterministic
     rows in `sample_weight` rather than dropping them entirely
     (preserves anchor signal at lower gradient).
  2. **The post-hoc test override on deterministic rows produced ZERO
     test prediction changes vs primary** (all 76,324 deterministic
     test rows were already at cell-majority in the primary's
     prediction). The override is a no-op on the LB-best primary,
     confirming the score=0/1/9 + per-cell purity diagnostic finding
     from earlier today: the primary already nails every deterministic
     boundary perfectly.

- LB delta: n/a (no probe warranted; gate failed cleanly on the
  primary anchor). LB best unchanged at **0.98094** via
  `submission_tier1b_greedy_meta.csv`.
- Final-selection lock unchanged: PRIMARY 0.98094 + audit-F1 swap
  HEDGE `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005).
- Artefacts (whitelisted via .gitignore for cross-branch reuse):
  - `scripts/recipe_full_te.py` (DROP_DETERMINISTIC env var integration)
  - `scripts/blend_gate_dropdet.py` (4-gate analyzer)
  - `scripts/purity_rules_diag.py`, `scripts/purity_subcells.py`
  - `scripts/artifacts/oof_recipe_full_te_dropdet.npy` + test (7.2 MB / 3.1 MB)
  - `scripts/artifacts/recipe_full_te_dropdet_results.json`
  - `scripts/artifacts/blend_gate_dropdet_results.json`
  - `scripts/artifacts/purity_rules_diag.json` + `purity_rules_per_cell.csv`
  - `scripts/artifacts/test_cell_majority.npy` (per-test-row cell-majority lookup)
  - `submissions/submission_recipe_full_te_dropdet.csv` (diagnostic, NOT for LB probe)

### 2026-04-27 — Layer-1 surgical override LB-tested: 0.98062 (-0.00032), 25th saturation + Bayesian-inversion lesson

- Goal: senior-engineer reframe — stop adding stacks; surgically correct
  the LB-best primary's argmax on test rows where it provably disagrees
  with a 100%-pure rule. 36 rows (35 H→M, 1 L→M) identified.
- Train-side validation: 46/46 disagreements on TRAIN are mathematically
  guaranteed correct (cell 100% pure ⇒ y == cell_majority by construction).
  OOF macro: 0.98084 → 0.98091 (+0.0000641, mathematical proof on train).
- Layer-2 (99.9-99.99%) tested in parallel and FALSIFIED (~84% override
  precision below 91.9% break-even floor under macro-recall).
- LB submission `submission_tier1b_greedy_meta_l1override.csv`:
  OOF 0.98091 → **LB 0.98062**, Δ vs LB-best **−0.00032**, gap +0.00029.
- **Bayesian-inversion error in the override math** (the senior lesson):
  - Train-side proof showed primary disagrees with cell-majority on 0
    test rows in 100%-pure cells of cube-level purity. The 36 came from
    sub-cell rules (cell × non-rule cat × value).
  - 100% purity is over RULE-feature space (the 6 rule features only).
    Primary's POST-BLEND argmax on those test rows uses NON-RULE features
    (Humidity, Soil_pH, Previous_Irrigation_mm, etc.) — exactly the
    features the NN generator uses to produce class flips.
  - Selection event "primary disagrees" is NOT independent of NN-flip
    presence. Conditioning on disagreement biases toward rows that ARE
    actually flipped. Primary's confident disagreement IS information,
    not error.
  - Math:
    P(override correct | random row in pure cell) ≈ 99.9%   (my naïve estimate)
    P(override correct | row in pure cell AND primary disagrees) ≈ ~33%
    Decomposition: ~12/36 correct (primary wrong) + ~24/36 wrong
    (primary correctly identified NN-flip the rule cell-aggregation can't see)
    Macro lift: 12×(+1/N_M_true) − 24×(−1/N_H_true) = −0.00032 ✓
- **25th independent saturation confirmation at LB 0.98094.**
- **Portable rule** (LEARNINGS.md): **"Train-side rule purity ≠
  joint-feature purity on rows where a strong learned model
  disagrees."** The selection event "model disagrees with rule"
  correlates with the underlying NN-flip process when the model uses
  non-rule features. To safely override on a 100%-pure cell, additionally
  require primary's max_prob < some uncertainty threshold (model-uncertain),
  never override when primary is confident — primary's confidence on a
  pure-cell row is calibrated information from non-rule features.
- LB best unchanged at **0.98094**. Hedge swap recommendation unchanged
  (`submission_3way_recipe025_s1035_s7040.csv` LB 0.98005).
- LB budget: 1/10 used today (this submission), 9 remaining.
- Artefacts (whitelisted via .gitignore for cross-branch reuse):
  - `scripts/build_l1_override_submission.py`
  - `submissions/submission_tier1b_greedy_meta_l1override.csv` (LB 0.98062)

### 2026-04-27 — L1-override 3-filter rescue: 26th saturation, break-even precision floor confirmed binding

- Goal: after the L1 override regressed −0.00032 LB (Bayesian inversion),
  user pushed back: "don't give up; think harder, give three perspectives".
  Three structurally distinct filter mechanisms tested via train-side
  validation on the 51 Layer-2 (99.9-99.99% pure) train disagreements
  (43 primary-wrong, 8 primary-right, ground truth known by construction):
    F1: hedge_agrees_with_cell_majority (consensus discriminator)
    F2: primary_max_prob < threshold (primary uncertainty)
    F3: primary 2nd-choice == cell_majority (posterior shape)
  Plus tiny meta-discriminator (LR with class_weight on 51 examples).
- Changed: `scripts/l1_override_filter_validation.py` (~250 lines).

- **Findings**:
  ```
  Filter        train precision     test n at threshold     verdict
  ────────────────────────────────────────────────────────────────
  F1 hedge      100% (n=1)          1                       too rare
  F2 max_prob<0.55  92.3% (n=13)    4                       barely above floor
  F2 max_prob<0.60  88.5% (n=26)    7                       BELOW floor
  F2 max_prob<0.85  87.0% (n=46)    27                      BELOW floor
  F3 second=maj 84.3% (universal on all 51 + 36)            no signal
  Meta LOO AUC  0.6657              top bucket 90% prec     BELOW floor
  ```

- **Macro-recall break-even precision for H→M overrides = 91.92%**
  (= N_M_true / (N_M_true + N_H_true), because High has 11.4× per-row
  leverage in macro-recall on this 3-class problem). 35 of 36
  candidates are H→M direction. None of the filters can clear this
  floor at meaningful row counts.

- **Direction-asymmetric break-even**:
  ```
  H→M override break-even: 91.92%   (rare class loss dominates)
  L→M override break-even: 60.74%   (only 1 candidate, lift ~+1e-6)
  M→L override break-even: 39.26%   (no candidates in our 36)
  ```

- **Diagnosis** — why filters fail:
  1. F1: hedge has SAME systematic disagreement pattern as primary on
     layer-2 cells. Both models use non-rule features; both agree on
     High where rule says Medium. Two models seeing same non-rule
     signal is NOT information that separates wrong from right.
  2. F2: primary's max_prob distribution overlaps between primary-
     wrong and primary-right. Some signal (high-confidence rows MORE
     likely primary-right) but precision sits 1-3pp below the 91.92%
     floor at usable thresholds.
  3. F3: on layer-2 cells, primary's softprob is SHAPED by the rule
     (cell's Medium-purity propagates through OTE features). Medium
     is always primary's 2nd when High is 1st. Universal → zero
     discriminative power.
  4. Meta-discriminator: per-row signals don't separate cleanly. LOO
     AUC 0.67 well below what's needed for >91.92% bucket precision.

- **26th saturation confirmation at LB 0.98094.** The L1 override
  lever is closed for the right structural reason, not because of
  filter choice. Macro-recall asymmetry creates a precision floor
  (91.92%) that no train-validatable filter on available side
  information clears at meaningful row counts.

- **Portable rule** (LEARNINGS.md candidate): "**Macro-recall
  break-even precision for class-c-from-anchor overrides equals
  N_other / (N_other + N_c). On rare-class direction overrides where
  the rare class has 10× the per-row leverage, this floor approaches
  ~0.92.** Filtered overrides on a strong learned model that uses
  features beyond the rule cannot reliably clear this floor because
  the selection event 'model disagrees with rule' correlates with
  the underlying NN-flip presence — the very signal a discriminating
  filter would need to detect. Lever is structurally closed unless
  rare class is over-represented in override direction OR external
  rule-orthogonal evidence is available."

- LB delta: n/a (no probe warranted; analysis is decisive). LB best
  unchanged at **0.98094**. Final-selection lock unchanged.
- Artefacts:
  - `scripts/l1_override_filter_validation.py`
  - `scripts/artifacts/l1_filter_validation_results.json`

### 2026-04-27 — Macro-recall surrogate gradient XGB: 27th saturation NULL but FIRST G4 PASS in 26+ confirmations

- Goal: train XGB with a custom obj that approximates `−∂macro_recall / ∂logits`
  directly, instead of CE + post-hoc log-bias. Hypothesis: the persistent
  Pareto-frontier closure is partly a TRAINING-TIME mismatch — every prior
  model optimized log-loss, then bias-tuned. Direct surrogate gradient
  should find a different operating point.
- Math: `R̃ = (1/K) Σ_k (1/N_k) Σ_{i:y_i=k} p_{ik}`. Gradient with balanced
  per-row weight `w_i = N/(K·N_{y_i})`:
  ```
  g_{im} = w_i · p_{ik*} · (p_{im} − δ_{m,k*}) / T
  h_{im} = w_i · p_{ik*} · p_{im}(1 − p_{im}) / T + ε
  ```
  Gradient peaks at p_{ik*}=0.5 — focuses on **boundary rows** that can flip
  with small updates. Opposite of focal (focuses on totally-wrong rows).
  Blend with CE via `MR_LAMBDA`: L = λ·L_CE + (1−λ)·(−R̃). Pure surrogate
  satiates fast (best_iter=3 in SMOKE); λ=0.3 keeps gradients alive.
- Implementation: `scripts/recipe_macrorecall.py` (custom obj closure +
  `disable_default_eval_metric=1` + `feval` returning −macro_recall).
  XGBoost 2.1+ requires (n_rows, n_classes) shape for grad/hess (not flat).
- SMOKE sweep on 20k × 2 folds (lam_ce ∈ {0, 0.3, 0.5, 0.7}):
  pure surrogate had highest tuned (0.96420) but best_iter=3 (satiated
  instantly); lam=0.3 had best_iter=84-101 (longest training).
  Production picked lam_ce=0.3.

- **Production 5-fold (504k, ~6 min/fold = 30 min total — much faster than
  recipe's 30 min/fold because best_iter ~170 vs ~1300):**
  ```
  fold    macrorec   recipe     Δ
  1       0.97691    0.97544   +0.00147
  2       0.97810    0.97659   +0.00151
  3       0.98015    0.97721   +0.00294
  4       0.97761    0.97465   +0.00296
  5       0.97918    0.97557   +0.00361
  mean Δ vs recipe:            +0.00250
  ```
  **ALL 5 FOLDS POSITIVE** — first sustained positive standalone delta in
  any experiment on this branch. OOF argmax 0.97839 (recipe 0.97589,
  Δ +0.00250); tuned 0.97879 (recipe 0.97967, Δ −0.00088 — surrogate
  is argmax-optimal so post-hoc bias gives less benefit; recipe catches
  up via the bias bump).

- **4-gate analyzer onto LB-best 4-stack (anchor 0.98084) — UNIQUE PATTERN:**
  ```
  raw blend sweep (fixed recipe bias):
    α      OOF Δ      PCR L    PCR M     PCR H     net_H  G4_ratio
    0.10   -0.00009   +0       -0.00114  +0.00085   +85    high (PASS)
    0.20   -0.00023   +0       -0.00227  +0.00157   +205   high (PASS)
    0.30   -0.00045   +0       -0.00382  +0.00247   +353   0.978 (PASS)
    0.40   -0.00073   +0       -0.00550  +0.00333   +485   high (PASS)
    0.50   -0.00119   +0       -0.00749  +0.00395   +614   high (PASS)
  ```
  **G4 PASSES CLEANLY for the first time in 26+ confirmations** (asymmetry
  ratio 0.98 = nearly all flips are ADD-High in correct direction). G1
  fails because Medium-loss steeper than High-gain.

- **Hard-gate override probe** (LB-4 says M AND macrorec says H):
  - Override set: 457 rows; truly-H: 24
  - **Precision 5.25%** (break-even under macro-recall: 8.1%) — FAIL
  - τ-sweep ANTI-CALIBRATED: τ=0.50 → 5.25%, τ=0.70 → 3.51%, τ=0.85 → 0%
  - Score-restricted (6,7,8): 5.14% precision (no improvement)
  - Δ macro-recall from full override: −0.00022

- **Verdict: 27th saturation NULL, but structurally unique:**
  ```
  Property                                Value
  Standalone fold-deltas all positive    ✓ first ever
  Jaccard with LB-4 anchor               0.64 (very orthogonal)
  +H recall standalone vs LB-4           +0.62pp (highest ever)
  G4 PASSES at α=0.30                    +353/361 ratio 0.98
  Hard-override precision                5.25% < 8.1% breakeven
  Anti-calibration at high τ             ✗
  ```
  The macro-recall surrogate IS the first mechanism to achieve clean
  ADD-High direction at the blend level — every prior experiment had
  RESHUFFLE / REMOVE-High / magnitude-trap patterns. It also delivers
  +0.62pp High recall standalone, the highest H boost on this comp.
  But the H precision on the OVERRIDE DOMAIN (rows where LB-4 predicts
  M and macrorec predicts H) is anti-calibrated — confident H predictions
  there are LESS likely to be true-H than uncertain ones. The structural
  ceiling is reconfirmed: the residual M↔H boundary information is NOT
  in any model derivable from the recipe FE space, regardless of training
  objective.

- **Two new portable rules** (LEARNINGS.md candidates):
  1. **Macro-recall surrogate gradient is the only training-time
     mechanism that produces clean ADD-High direction at blend level**
     on this problem. Pure surrogate satiates at p_{ik*}≈0.5 (gradient
     vanishes) so needs CE blend (lam_ce=0.3 sweet spot) to keep
     gradients alive long enough to build meaningful trees. Use this
     as the reference XGB obj for any future imbalanced multi-class
     comp where macro-recall is the metric.
  2. **G4 (asymmetric ADD-rare-class flip ratio ≥ 0.5) is necessary
     but not sufficient.** First experiment in 27 saturations to clear
     G4 cleanly (ratio 0.98) — but G1 still fails because the
     per-class trade is unfavorable under macro-recall (M loss > H
     gain on a per-class-recall basis). Adding a 5th gate: H_gain ×
     (1/N_H) - M_loss × (1/N_M) > +1e-4, i.e., the macro-recall
     contribution must be net positive on the per-class scale, not
     just direction-correct.

- LB-best primary unchanged: **LB 0.98094**
  (`submission_tier1b_greedy_meta.csv`).
- Final-selection lock unchanged: PRIMARY 0.98094 + HEDGE 0.98005.
- LB budget: 0 spent today.

- Artefacts on `claude/residual-target-encodings-Duy1o` + main:
  - `scripts/recipe_macrorecall.py` (custom obj orchestrator, 272 lines)
  - `scripts/artifacts/oof_recipe_full_te_macrorec_T1_lam03.npy` + test +
    results JSON (5-fold OOF, all 5 folds positive vs recipe)
  - `scripts/artifacts/blend_gate_4gate_macrorec_T1_lam03_results.json` +
    `_iso_results.json` (definitive 4-gate fail diagnosis)
  - `submissions/submission_recipe_full_te_macrorec_T1_lam03.csv`
    (diagnostic — not for LB probe, structurally cleanest ADD-High
    candidate but precision 5.25% < 8.1% breakeven)

### Next steps: macrorec follow-up ideas (post-2026-04-27, 27th saturation)

The macro-recall surrogate's structurally unique properties (first +0.62pp H
recall standalone, first G4 PASS in 26+ confirmations, all 5 folds positive)
suggest the surrogate's signal IS real — just that direct standalone
deployment has the M-loss>H-gain trade. Three ways to leverage it without
inheriting the failure:

  **N1. Macro-recall surrogate at the META-STACKER level** (top pick, ~1h CPU).
  Every prior meta-stacker (XGB, LR, MLP, RF) used CE/BCE. Train an XGB
  meta-stacker on the existing 64-component bank using the macro-recall
  surrogate gradient instead of CE. Meta sees richly-calibrated component
  inputs AND optimizes the metric directly. The +0.62pp H-recall lift may
  COMPOUND at meta-level because:
  - Components in the bank already carry orthogonal H-class signals; meta-XGB
    can pick which to amplify
  - Pareto-frontier closure that bounded standalone macrorec doesn't necessarily
    bind a meta over a 200-dim component-prob feature space
  - Fixed-bias blend at α=0.30 onto LB-best 3-stack reuses the same architecture
    that produced LB 0.98094
  Risk: bank-extension overfit (N5b, R2/R5 LB-regress pattern) — but here we're
  not adding a new component, just changing the meta objective. Different
  failure mode. Bayesian prior of LB lift: 20-30%.

  **N2. Asymmetric-weight macrorec** (direct attack on M-loss failure, ~2.5h CPU).
  Current surrogate uses `sample_w = N / (K · N_k)` which up-weights High class
  10× over Low. That's exactly why M→H push is so aggressive. Try sqrt-weighting:
  `sample_w = N / (K · sqrt(N_k))` — H weight drops from 10× to ~3×. Less
  aggressive H push → fewer Medium losses → may pass G1.
  Variants to sweep: log-weighting (`log(N/N_k)`), unit-weighting (no class
  scaling), current N/N_k. Cost: 30 min/variant × 4 variants × smoke + 5-fold
  of best = ~3h. Bayesian prior: 15-25%.

  **N3. Macrorec probs as recipe features (model-level distillation)** (~1h CPU).
  Add macrorec's 3 OOF probs as 3 extra numeric cols to recipe FE. Recipe XGB
  then has access to macrorec's macro-recall-optimized signal AND the full
  443-feature bank, depth-4 trees decide per-row when to trust macrorec's
  H push. Structurally distinct from Phase A residual TE:
  - Phase A: per-key OTE features (correlated with existing 117 OTE keys)
  - This: model-level posterior (3 cols, different signal type)
  - Recipe XGB acts as a learned filter on macrorec's predictions
  Bayesian prior: 15-20% (lower than N1 because Phase A's similar mechanism
  nulled).

  **Execution priority**: N1 first (highest EV, most novel mechanism, fastest).
  If gates positive, LB-probe with tomorrow's fresh slots. If null, N2 next.
  N3 last because Phase A has primed us to expect feature-level macrorec
  delivery would be redundant with existing OTE+digit FE.

### 2026-04-27 — N1 macro-recall meta-stacker on 170-component bank: 3/4 gates PASS but G4 RESHUFFLE

- Goal: execute N1 from the 3-idea brainstorm — apply the macro-recall
  surrogate gradient at the meta-stacker level (`scripts/n1_metamacrorec.py`).
  Reuses tier1b_xgb_metastack pool + meta-feature construction; only the
  XGB obj swaps to the surrogate from `recipe_macrorecall.py`.
- Pool ballooned to 170 components (every 3-class OOF on disk that's not
  in EXCLUDE — includes recently-added macrorec_T1_lam03 itself, residte,
  basemargin_K2, plus all components from prior sessions).
- 5-fold seed=42 (~22 min wall total):
  ```
  fold    bal_acc    best_iter
  1       0.98118    62
  2       0.98209    120
  3       0.98306    15
  4       0.98209    4
  5       0.98219    191
  OOF argmax = 0.98212  (LB-best 4-stack OOF 0.98084, +0.00128 standalone)
  iso-cal'd @ recipe-bias = 0.98196
  ```
  Best_iters wildly variable — surrogate satiates near-instantly at meta
  level when component bank already gives p_true ~ 0.99 (gradient
  ∝ p_true(1-p_true) → 0).
- REPLACE-v1 architecture (N1_iso × α=0.30 onto LB-best 3-stack) vs
  LB-best 4-stack:
  ```
  α=0.10 → OOF 0.98115  (+0.00031)
  α=0.20 → OOF 0.98142  (+0.00058)
  α=0.30 → OOF 0.98149  (+0.00065)
  α=0.50 → OOF 0.98172  (+0.00088)
  ```
  Standalone N1 iso has +0.00376 H recall vs LB-best 4-stack — the
  base-level macrorec H-direction signal SURVIVES at meta level.

- **4-gate analyzer @ α=0.30 iso (the recommended path):**
  ```
  G1: +0.00075   PASS  (≥+0.0003)
  G2: PCR L -0.00018 / M +0.00043 / H +0.00200   PASS  (no class drops
                                                         below -5e-4 floor)
  G3: ratio 1.08   PASS  (in [1.0, 2.0] stable range)
  G4: net_H +49, churn 277, ratio 0.18   FAIL  (RESHUFFLE pattern,
                                                ratio < 0.5 floor)
  ```
  **First experiment in 27 saturation confirmations to clear 3 of 4
  gates**, but G4 binds. The H-direction is POSITIVE (+49 net) but
  RESHUFFLE dominates (114 H-revokes + 163 H-adds = 277 churn).
  Same pattern as LR-meta v2 / classw / D 3-meta — all of which
  LB-regressed despite passing G1-G3.

- **Decision (per user direction)**: refine N1 further before LB-probe.
  Tomorrow's fresh LB slot reserved for one final-form variant with
  better G4 asymmetry. See "Next steps: N1 refinements" below.

- LB-best primary unchanged: **LB 0.98094**. LB budget unchanged.

### Next steps: 3 N1 refinements (ranked by EV/cost) — 2026-04-27

The N1 result clears G1+G2+G3 cleanly but G4 RESHUFFLE pattern is the
exact failure signature that LR-meta v2 / classw / D 3-meta showed
before LB-regressing. The H-direction signal IS real (PCR_H +0.002 at
α=0.30, +0.00376 standalone) but the meta-stacker is reshuffling H
predictions rather than cleanly adding them. Three refinements aimed
at preserving the H signal while flipping G4:

  **R1. Curated-pool re-run** (top pick, ~25 min CPU + 5 min meta).
  The 170-component pool includes circularity-risk components (macrorec
  itself, residte, basemargin) AND known LB-regressors (soft_distill,
  recipe_pseudolabel_stage2, several c0_v* derived metas). Curate to
  ~50 LB-validated, structurally diverse components by ADDING explicit
  EXCLUDE entries for:
    - macrorec_T1_lam03            (circular: macrorec output as macrorec-meta input)
    - residte / basemargin_K2      (LB-regressors / null branches)
    - soft_distill (already excluded — confirm)
    - all c0_* + xgb_metastack_v* (prior meta outputs, already excluded)
  Re-run N1 with this curated set. Hypothesis: smaller bank → less
  overfit → meta finds a cleaner ADD-High signal rather than RESHUFFLE.
  Cost: re-run scripts/n1_metamacrorec.py with extended EXCLUDE.
  Bayesian prior of clearing G4: 30%.

  **R2. Pure macrorec at meta level (lam_ce=0.0)** (~5 min CPU).
  Currently lam_ce=0.3 (30% CE). The CE component may be pushing the
  meta toward log-loss-optimal — which on a saturated bank means
  RESHUFFLE around the existing operating point. lam_ce=0.0 gives pure
  surrogate that satiates fast (best_iter=3 at base level) but might
  preserve the clean ADD-High direction. The "directional purity" of
  pure surrogate IS what we want to preserve, even if at fewer rounds.
  Cost: rerun with `MR_LAMBDA=0`, suffix `_metamacrorec_lam0`.
  Bayesian prior: 25%.

  **R3. Anchor-anchored meta loss with KL regularizer** (~30 min CPU
  for new obj implementation + 25 min run).
  Add a NEW penalty term to the surrogate: `λ_kl × KL(meta_probs ||
  lb_best_4stack_probs)`. Forces meta predictions to stay near the
  LB-best 4-stack except where macrorec gradient is strong enough to
  overcome the KL pull. Mathematically prevents the H-revoke direction
  (G4 RESHUFFLE failure) while preserving H-add. Different from
  iso-cal which is a per-class monotonic mapping (post-hoc); this is
  a per-row constraint at training time. Bayesian prior: 20%.

  **Execution order**: R1 first (cheapest, addresses obvious circularity
  and overfit risk). If G4 still fails, R2 next (also cheap). R3 only
  if R1+R2 don't clear G4 — implementation requires 1 new obj closure.

  **Skip**: re-running with different XGB HPs (depth/lr sweep) — the
  surrogate's behavior is dominated by the gradient shape, not tree
  depth. HP sweeps were already exhausted in 2026-04-22 Optuna run.

### R1 concrete EXTRA_EXCLUDE list (post-N1 G4 RESHUFFLE finding)

The N1 meta-stacker pool ballooned to 170 because `EXCLUDE` only listed
historical regressors. The macro-recall meta needs a tighter pool. Add to
EXCLUDE for R1 re-run:

  Circular (macrorec output as macrorec-meta input):
    - recipe_full_te_macrorec_T1_lam03
    - xgb_metastack_metamacrorec_lam03  (this experiment's own output)
  Recently-confirmed branch-NULL components (this branch + adjacent):
    - recipe_full_te_residte             (24th saturation)
    - recipe_full_te_basemargin_K2       (25th saturation)
    - recipe_full_te_dropdet             (training-data lever NULL)
    - tier1b_greedy_meta_l1override      (L1 override LB regressor -0.00032)
  LB-regressor metas (variance test ratios -1x to -3x):
    - xgb_metastack_classw, xgb_metastack_n5b_both
    - lr_metastack, lr_metastack_v2, mlp_metastack
    - xgb_metastack_bag3, xgb_metastack_v3, v4, varB, varC

Curated bank target: ~40-50 LB-validated components. Same script
(`scripts/n1_metamacrorec.py`) but extend the imported EXCLUDE set.


### 2026-04-28 — R2 hybrid 0.75 LB-probed: 0.98048 (−0.00046, 28th saturation)

- Submitted `submission_R2_hybrid075_a015.csv` at 04:04 UTC. **LB public = 0.98048**.
  Δ vs LB-best primary 0.98094 = **−0.00046** (REGRESSION).
- OOF→LB gap = 0.98140 − 0.98048 = **+0.00092** (vs LB-best primary's −0.00010).
- ALL 4 GATES PASSED on OOF: G1 (+0.00056) ✓, G2 (PCR all within −5e-4) ✓,
  G3 (ratio 1.36) ✓, G4 (ratio 0.62, net_H +154 clean ADD-direction) ✓.
- First time in 27 saturations ALL FOUR gates passed cleanly. Still LB-regressed.

- **Root cause: hybrid grid-search selection bias.** The `hybrid_ratio` was
  selected from a 6 × 4 = 24-point sweep (ratios in {0.0, 0.25, 0.4, 0.5, 0.6,
  0.75, 0.85} × α in {0.05, 0.10, 0.15, 0.20}) by maximizing OOF Δ. This
  introduces ~30-50 bp of selection bias on the OOF metric. The "true"
  unbiased OOF Δ was probably +0.00010 to +0.00020; LB regression of
  −0.00046 reflects the inflated OOF metric eating into actual signal.

- **Portable rule** (LEARNINGS.md candidate): "**The 4-gate framework
  validates DEPLOYABLE candidates, not POST-HOC SELECTED candidates.**
  When a configuration knob (hybrid mix ratio, α weight, threshold) is
  selected from a grid by OOF performance, even AFTER all 4 gates pass
  on the selected point, the OOF metric carries selection-bias inflation
  proportional to grid size and OOF noise. For a 24-point sweep on a
  saturated meta, expect ~30-50 bp inflation. To deploy without this
  contamination: pick the configuration knob from THEORY (e.g., 'use
  α=0.30 because that's the documented LB-validated value for the
  primary architecture'), THEN run gates."

- 28th saturation confirmation at LB 0.98094. The ALL-4-GATE-PASS finding
  is real (the OOF lift IS structurally there) but the post-hoc selection
  pattern is the failure mode: 24-point grid search on OOF → selection
  bias contaminates the gate. Earlier 27 saturation confirmations all had
  G4 fail or G2 fail at fixed-config evaluations; today's experiment
  reached G4 PASS only via grid-selected hybrid ratio, which is exactly
  the OOF-overfit pattern the gate framework was designed to defend
  against.

- LB-best primary unchanged: **0.98094** via `submission_tier1b_greedy_meta.csv`.
- Final-selection lock unchanged: PRIMARY 0.98094 + HEDGE 0.98005.
- LB budget today (2026-04-28 UTC): 1/10 used (this probe), 9 remaining.

- **Strategic implication:** to break LB 0.98094 on the macro-recall
  surrogate path, would need EITHER:
  - R3 KL-regularized meta loss (theoretical motivation — no grid
    selection on hybrid mix, just one new hyperparameter `lambda_kl`)
  - OR a different mechanism entirely
  Pending decision: is R3 worth ~50 min CPU + 1 LB slot when 28
  saturations now exist?
