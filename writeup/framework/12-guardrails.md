# 12 — Guardrails

Eleven invariants. Each has a trigger condition and a worked example
from the irrigation comp.

## 1. Ask-first / no-loop on submissions

**Rule**: every `kaggle competitions submit` invocation needs explicit
human confirmation, single-shot, never wrapped in a retry / `until` /
`while` / `for` loop. Monitors that POLL submission status are fine;
monitors that WRITE submissions are forbidden.

**Trigger**: any time an LB probe is recommended.
**Example**: 04-26 retry-loop burned 4 slots on the same CSV due to a
case-mismatched success marker (`successfully` vs Kaggle's capital-S
`Successfully submitted`).

## 2. Smoke-test + 1-fold time-probe + 1h GPU cap

**Rule**: before any multi-hour run: (a) smoke at 1 fold / 1 trial /
50k subsample; (b) 1-fold full-data time-probe to estimate 5-fold
wall time. If projected ≥1h, shrink config. If a running kernel is
still in preprocessing at t+30min with no fold output, kill it.

**Trigger**: any new pipeline, any GPU kernel, any Optuna sweep.
**Example**: pytabkit RealMLP on Kaggle GPU ate 3h34min of CPU
preprocessing before training due to `n_ens=8` × `cv=5` multiplier
the published claim hadn't disclosed. Killed at t+3h34min, zero
output. A 5-min smoke run would have caught it.

## 3. 4-gate leakage filter pre-LB-probe

**Rule**: every candidate with OOF Δ ≥ +0.0005 vs LB-best must pass
all four gates before LB probe.

- G1 — Standalone OOF clears the prior LB-best anchor.
- G2 — Blend with anchor lifts at α* > 0.
- G3 — Net rare-class-flip ratio ≥ 0.5.
- G4 — Direction asymmetry: more correct flips than incorrect.

Plus a minimal-input meta sanity check: train candidate meta with
ONLY 2 components (anchor + new). If 2-comp OOF < anchor, stop.

**Trigger**: any stacking / hybrid / distillation candidate.
**Example**: 7 leakage incidents costing ~0.0045 LB before this rule
existed (see [postmortem/04-what-failed.md](../postmortem/04-what-failed.md)).

## 4. NEVER-GIVE-UP / saturation-is-bounded / never-lock-and-stop

**Rule**: saturation evidence proves we tested *known* levers, not
that no lever exists. Don't recommend "lock and stop" while LB
budget remains. After every null, brainstorm 3 mechanisms NOT yet
on the hypothesis board.

**Trigger**: after every null candidate; after any session-log
sentence containing "structural ceiling" or "lock final".
**Example**: every plateau in this comp (0.97097 → 0.98150) was
declared structural and refuted within a week. The mechanisms that
broke each plateau were on the agent's "skip on principled grounds"
list at the time.

## 5. Keep CLAUDE.md fresh / archive-on-bloat

**Rule**: cap any agent-loaded doc at ≤150 lines / ≤50k tokens.
Archive when bloated. Subagents load slices, not full files.

**Trigger**: CLAUDE.md > 50k tokens, or any agent-loaded file > 150
lines.
**Example**: CLAUDE.md crossed 1MB at one point, triggering API idle
timeouts and burning subagent context. Archived to
`audit/CLAUDE-md-archive-2026-04-30.md`.

## 6. Heuristics before heavy compute

**Rule**: before reaching for Optuna / GPU / 5-fold-bagging, try a
closed-form rule, a threshold, or a hand-coded baseline. Bound the
lift available before spending compute.

**Trigger**: any new mechanism family.
**Example**: H1 (Soil_Moisture alone, 2 thresholds) covered 2/3 of
the random→LGBM distance. The DGP rule (closed form, 6 features)
explained 98.4% of training labels. Both took <1 hour each;
LGBM-tuned took longer and got us 0.97097.

## 7. Research before saturation

**Rule**: at every plateau, spend a research turn on external sources
(Kaggle public notebooks, prior-comp writeups, papers) before
declaring the ceiling structural.

**Trigger**: 3 consecutive nulls OR 5 saturation events at the same
LB OR 2 days without LB lift.
**Example**: every plateau-break in this comp came from external
research (override mechanism = read another team's notebook approach;
RF on natural-cal bank = sklearn pattern from a prior comp).

## 8. Settled-once facts

**Rule**: LB stability, public-split %, eval metric, deadline,
team-size limit, data license — ask once on Day 1, write to
`comp-context.md`, never re-ask.

**Trigger**: any session.
**Example**: the irrigation agent kept asking mid-run if the LB was
stable. Answer settled Day 1 (80/20 split, large enough that 0.00005
is the floor probe resolution); should never have been re-asked.

## 9. File-size cap ≤150 lines

**Rule**: every committed doc ≤150 lines. Long docs cause Anthropic
API idle timeouts on writes and explode subagent context loads.

**Trigger**: any new file or edit pushing a file over 150 lines.
**Example**: this rule was added in CLAUDE.md after multiple
long-file timeout incidents and the 1MB CLAUDE.md episode.

## 10. Pull-style updates

**Rule**: no proactive minute-level chatter during long jobs. On
human pull, give a 1-2 sentence summary of the latest concrete fact,
no recap.

**Trigger**: any long-running job.
**Example**: the human wanted "what's the latest fact?" answered in
1-2 sentences. The agent's defaults of either silence or
minute-level updates both failed this contract.

## 11. Model-routing / token economy

**Rule**: route by task. Haiku-tier (cheap) for routine read-only
checks (lb-status grep, file existence, smoke verifications). Sonnet
for default work. Opus for hard reasoning (plan design, leakage
diagnosis, novel mechanism brainstorm).

**Submission-budget discipline pairs with token-budget discipline**:
use the daily 10/day Kaggle slots every day; don't sit on slots
(that's its own failure). Don't burn tokens on stale-context
re-derivations.

**Trigger**: every subagent invocation, every long-running session.
**Example**: this comp ran on top-tier model for everything; cost
was disproportionate to the lift extracted. Routine LB-status checks
and file-existence verifications are Haiku-tier work.
