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

## ⚠️ ALWAYS CHECK KAGGLE LB SUBMISSIONS BEFORE RECOMMENDING ANY CANDIDATE

**Kaggle's `kaggle competitions submissions playground-series-s6e4` is
the ONLY authoritative source of truth for what has been LB-tested
and what score it got.** Git commit messages, CLAUDE.md prose, and
session-log entries can lag behind, omit results, or describe
candidates that were emitted but never submitted.

**Mandatory pre-recommendation check**: before recommending ANY
submission CSV as an "unprobed candidate" or "highest-EV next probe",
run:
```
python scripts/lb_status.py | grep <filename>
```
or directly:
```
kaggle competitions submissions playground-series-s6e4
```
and verify the candidate filename does NOT appear in the list.

If it appears, the LB score is the documented outcome — STOP, do not
re-recommend, and surface the actual score to the user.

Cost asymmetry: re-recommending an already-tested LB-regressor wastes
user attention, can lead to a duplicate submission burning a slot,
and erodes trust in the agent's analysis. The check costs <5 seconds.

This rule was added 2026-04-30 after recommending
`submission_rawashishsin_k4_overridden.csv` (already submitted at
LB 0.98112, −0.00022 regression vs prior LB-best 0.98134) as a
"highest-EV unprobed candidate" — the candidate had been probed
and regressed 8 hours earlier, but I hadn't checked Kaggle's CLI.

## ⚠️ NEVER SUGGEST LOCKING FINAL SUBMISSIONS

**Do not recommend "lock the 2 finals and stop"** in any form — not as a
primary recommendation, not as a fallback, not as an option. The user
has explicitly disabled this advice. If asked "what next?", surface
substantive next experiments only. Final-selection slot management is
the user's call, not advice the agent should volunteer. This rule
overrides any session log that says "lock + stop".

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

## ⚠️ DEFEND AGAINST LEAKAGE — STACKING LIFTS ARE OFTEN OOF-INFLATED

**Every "OOF Δ ≥ +0.0005 vs LB-best" candidate must pass leakage checks
BEFORE LB-probe.** This rule applies to stacking experiments, hybrid
blends, distillation students, bank-extension metas, and any
configuration that's selected by maximizing OOF.

Across this competition, **7 leakage / OOF-overfit incidents have cost
~0.0045 LB total** (each measured separately):
  - 2026-04-23 stage-2 pseudo-label (-0.00009): labeler+target same folds
  - 2026-04-23 stacking-inflation ceiling: 3+ blends at OOF 0.98030 → LB ~0.97995
  - 2026-04-24 soft-distillation (-0.00148): student memorizes teacher OOF noise
  - 2026-04-25 LR meta v1 (-0.00103) + v4 ET+kNN (-0.00102) + P3 perturbed (-0.00139)
  - 2026-04-26 DROP_DETERMINISTIC: removed boundary-anchor rows
  - 2026-04-27 R2 hybrid grid-selected (-0.00046): 24-point grid → OOF inflation
  - 2026-04-28 stacking feature leak: 80% gain from circular meta-of-metas

**Mandatory pre-LB-probe checks** (cheap, total ~10 min CPU):
  1. **Minimal-input meta test**: train candidate meta with ONLY 2
     components (anchor + candidate). If 2-component OOF lands BELOW
     anchor at recipe-bias, the N-component lift was cross-component
     memorization, NOT orthogonal signal. Don't deploy.
  2. **Theory-only hyperparameter check**: any knob (hybrid mix, α,
     threshold) chosen by maximizing OOF carries 3-50 bp of selection
     bias. Use LB-validated defaults (e.g., α=0.30 for the LB-best
     primary architecture) instead.
  3. **Cross-meta error correlation**: if 2-3 same-objective metas have
     pairwise Jaccard ≥ 0.85, sparse averaging won't decorrelate the
     gap. Skip ensemble experiments.
  4. **Feature importance audit**: if top-N gain features in a
     meta-stacker are themselves prior meta outputs, the meta is
     meta-of-metas — drop them via EXTRA_EXCLUDE.

**Full rules in `LEARNINGS.md` § "Leakage & OOF-honesty".** Five
portable patterns documented:
  - Stacking feature leak + minimal-input meta detection
  - Grid-search selection bias on OOF
  - Stacking-inflation ceiling on saturated banks
  - Soft-distillation student-memorizes-teacher-OOF
  - OOF-honesty via GroupKFold (30-second sanity check)

**Red flags that mandate `LEARNINGS.md` re-reading**:
  - "All folds positive but no LB lift" → stacking feature leak
  - "Hyperparameter chosen from grid" → selection bias
  - "Meta-stacker with N ≥ 50 components" → check feature importance
  - "OOF→LB gap suddenly +0.0005+" → leak amplification
  - "Distillation from bagged-OOF teacher" → student memorization

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


## Current state (compact summary, 2026-04-30)

The full pre-2026-04-30 session log, hypothesis board, playbook, and
all addenda have been archived to
`audit/CLAUDE-md-archive-2026-04-30.md` to keep this file under
50k tokens (haiku subagent compatibility). This section is the
canonical short-form summary; the archive is the canonical long-form
record.

### LB state (top of leaderboard for this team)

- **PRIMARY (LB-best)**: `submission_idea4b_selective_override.csv`
  → **LB 0.98150** — triple-consensus override on B (LB 0.98140),
  108 selective flips (105 H→M, 2 L→M, 1 M→L). Construction:
  bagged_v1' disagrees with B + {raw, tier1b} unanimous + 14-bank
  majority all agree.
- **HEDGE candidate**: `submission_sklearn_rf_meta_natural_v1_lb98129.csv`
  → LB 0.98129 — sklearn RandomForest meta-stacker on 7-component
  natural-cal bank. Orthogonal failure mode from primary.
- **Pack**: 0.98148 (+below us by 0.00002).
- **Leader**: Cdeotte 0.98219 (+0.00069 above).

### Calibration ladder (representative entries)

```
recipe_full_te                       OOF 0.97967 → LB 0.97939   gap +0.00028
recipe × pseudo 2-way                OOF 0.98012 → LB 0.97998   gap +0.00014
3-way multi-seed                     OOF 0.98029 → LB 0.98005   gap +0.00024
greedy 3-way log-blend (digit-XGB)   OOF 0.97375 → LB 0.97296   gap +0.00079
LB-best 3-stack (lb3+rmlp+nr_iso)    OOF 0.98061 → LB 0.98008   gap +0.00053
LB-best 4-stack (tier1b_greedy_meta) OOF 0.98084 → LB 0.98094   gap −0.00010
v1 RF natural standalone             OOF 0.98063 → LB 0.98129   gap −0.00066
rawashishsin v3 standalone           OOF 0.98016 → LB 0.98109   gap −0.00099
2-OTHER raw+tier1b k=2 unanimous (B) OOF 0.98088 → LB 0.98140   gap −0.00052
Idea 4b triple-consensus (PRIMARY)   OOF ~0.98088 → LB 0.98150   gap −0.00062
```

Negative-gap entries (LB above OOF) are the 14-bank-majority + override
mechanism family. The 4b primary's −0.00062 gap is documented as
structurally a function of WHICH 145 candidate rows the override fires on,
not a general margin to spend.

### DGP rule (closed-form, ~98.4% accurate on synthetic data)

Reverse-engineered from the 10k-row original dataset and confirmed on
the 630k synthetic train. The host trained a NN on the original 10k,
then the NN labeled the 630k+270k synthetic. ~1.6% of synthetic rows
are "flipped" by the NN from the rule prediction to a neighbouring class.

```
dry     = (Soil_Moisture < 25)
norain  = (Rainfall_mm   < 300)
hot     = (Temperature_C > 30)
windy   = (Wind_Speed_kmh > 10)
nomulch = (Mulching_Used == "No")
Kc      = 2 if Crop_Growth_Stage in {Flowering, Vegetative} else 0
score   = 2*(dry + norain) + (hot + windy + nomulch) + Kc
→ Low if score <= 3 ; Medium if 4 <= score <= 6 ; High if score >= 7
```

Implementation: `scripts/dgp_formula.py`.

### Saturation count: 40+ independent confirmations at LB 0.98150

The structural ceiling at LB 0.98150 is confirmed across every
mechanism family tested: greedy / meta-stacker variants / NN families
(18 nulls including TabPFN-10k, RealMLP n_ens={1,2,4}, FT-T, KAN,
Mamba, Trompt, TabM, ExcelFormer) / wide programmatic FE / per-row
gating / Pareto-frontier overrides / decision-rule variants /
training-data-composition (DROP_DETERMINISTIC) / soft-distillation
across 4 capacity points / multi-seed pseudo-labeling / leak-honest
4-gate / boundary-confined TTA / symbolic regression / k-NN
label-propagation / OOD-anchored 10k features (N5b family
3 LB-validated regressions at -2.3× to -3.2× carryover).

The CLAUDE.md NEVER-GIVE-UP rule still applies — every prior plateau
in this comp was broken by a mechanism not yet tried. But the
saturation evidence is now exhaustive across the documented mechanism
classes, so any new candidate must come from a structurally distinct
direction (external supervision, NN inversion, novel feature view,
or a calibration mechanism not yet on this hypothesis board).

### Most recent results (2026-04-29 to 2026-04-30)

```
2026-04-29  RF natural standalone        0.98063 OOF → 0.98129 LB    NEW PRIMARY
2026-04-29  4 surprise-options           NULL ×4 (37th-38th saturations)
2026-04-30  k=4 unanimous override       0.98134 LB    +0.00005
2026-04-30  2-OTHER raw+tier1b k=2 unan  0.98140 LB    +0.00006 (became B)
2026-04-30  Idea 4b triple-consensus     0.98150 LB    +0.00010 ← LB-BEST
2026-04-30  Idea 5 anchor-switch         0.98148 LB    -0.00002
2026-04-30  98150 minus 176 L->M flips   0.98148 LB    -0.00002
2026-04-30  4b + W5(M->H) + strict90     0.98143 LB    -0.00007
2026-04-30  T6 directional compose       0.98121 LB    -0.00029  (40th saturation)
```

### Pointers to other key files

- `audit/CLAUDE-md-archive-2026-04-30.md` — full historical session log
  (the prior CLAUDE.md content). All ⚠️ rules in this file remain
  authoritative; the archive contains experimental detail and
  methodology.
- `audit/2026-04-30-T6-directional-compose-result.md` — most recent
  saturation entry, with the OOF→LB transfer-asymmetry portable rule.
- `LEARNINGS.md` — portable patterns for future competitions.
- `REPORT.md` — work report.
- `brief.md` — verbatim host material.

### Hypothesis board (compact)

- **Current best (LB)**: `submission_idea4b_selective_override.csv`
  → LB 0.98150 (mechanism described above).
- **Untried mechanism categories** (ranked by EV):
  1. External supervision (LLM judge, agronomic expert) — blocked
     locally without external API key; haiku subagents need
     CLAUDE.md compaction (this file).
  2. NN inversion / DGP archaeology — host's NN architecture
     unknown; speculative.
  3. Composition of LB-validated submissions via new override axes
     (orthogonal to triple-consensus). All three direction-restricted
     compose variants tested in 2026-04-30 produce regression at
     -1× to -3× carryover (W3_MHonly, W5, T6 directional).
- **Skip on principled grounds**: meta-stacker bank extension (10+
  saturations); NN architectures (18 saturations); wide programmatic
  FE (recipe redundancy); public-CSV blending (banned by ⚠️ rule).

