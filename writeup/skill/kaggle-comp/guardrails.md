# Guardrails — the 11 invariants

These are **hard rules**. Each has a trigger, the rule itself, and
the failure mode it prevents.

## 1. Ask-first / no-loop on submissions

**Trigger**: any time `kaggle competitions submit` would be invoked.
**Rule**: explicit per-call human confirmation. Single-shot. Never
wrap in retry / `until` / `while` / `for`.
**Prevents**: a case-mismatched success-marker loop burning multiple
slots on the same CSV.

## 2. Smoke + 1-fold time-probe + 1h GPU cap

**Trigger**: any new pipeline, GPU kernel, or Optuna sweep.
**Rule**: smoke at 1 fold / 50k rows first. Then 1-fold full-data
time-probe. If projected wall time ≥ 1h, shrink config.
**Prevents**: a kernel that ate 3h34min of CPU preprocessing before
training started.

## 3. 4-gate leakage filter pre-LB-probe

**Trigger**: any candidate with OOF Δ ≥ +0.0005 vs LB-best.
**Rule**: G1 standalone OOF / G2 blend lift / G3 net-rare-class-flip
ratio / G4 direction asymmetry. Plus minimal-input meta sanity check
(2-component meta beats anchor or stop).
**Prevents**: 7 leakage incidents costing ~0.0045 LB.

## 4. NEVER-GIVE-UP / saturation-is-bounded / never-lock-and-stop

**Trigger**: any session-log sentence with "structural ceiling",
"lock final", "stop spending compute".
**Rule**: saturation evidence proves we tested *known* levers, not
that no lever exists. After every null, brainstorm 3 untried
mechanisms. Locking is a fallback for the final 3-day window only.
**Prevents**: every prior plateau in the reference comp was
declared structural and refuted within a week.

## 5. Keep CLAUDE.md fresh / archive-on-bloat

**Trigger**: CLAUDE.md > 50k tokens OR any agent-loaded file > 150
lines.
**Rule**: archive when bloated. Subagents load slices, not full
files.
**Prevents**: 1MB CLAUDE.md, API idle timeouts, exploded subagent
context loads.

## 6. Heuristics before heavy compute

**Trigger**: any new mechanism family.
**Rule**: closed-form rule / threshold / hand-coded baseline before
Optuna / GPU / 5-fold-bagging. Bound the lift available first.
**Prevents**: spending compute on a mechanism a 30-minute heuristic
would have falsified.

## 7. Research before saturation

**Trigger**: 3 consecutive nulls OR 5 saturation events at the same
LB OR 2 days without LB lift.
**Rule**: web search top-N public notebooks; read 2 prior-comp
writeups in same domain; list 5 untried mechanisms with citations.
**Prevents**: introspection-only convergence onto already-explored
mechanism families. Every plateau-break in the reference comp came
from this loop.

## 8. Settled-once facts

**Trigger**: any session.
**Rule**: LB stability, public-split %, eval metric, deadline,
team-size limit, data license, external-data rules — ask once on
Day 1, write to `comp-context.md`, never re-ask.
**Prevents**: re-asking the same facts mid-run, burning user
attention and tokens.

## 9. File-size cap ≤150 lines

**Trigger**: any new file or edit pushing a file over 150 lines.
**Rule**: split into multiple short files with clear
responsibilities. One file for model, one for features, one for
training loop, one for orchestrator. Never write a single large
file in one shot.
**Prevents**: Anthropic API idle timeouts on long writes.

## 10. Pull-style updates

**Trigger**: any long-running job.
**Rule**: no proactive minute-level chatter. On human pull, give a
1-2 sentence summary of the latest concrete fact, no recap. The
human will ask when they want an update.
**Prevents**: noise spam during long runs; agent silence for an
hour with no signal.

## 11. Model-routing / token economy

**Trigger**: every subagent invocation.
**Rule**:

- **Haiku**: routine read-only checks (lb-status grep, file
  existence, smoke verifications, CSV diff).
- **Sonnet**: default work — Planner, Runner, Bookkeeper.
- **Opus**: hard reasoning — leakage diagnosis, novel mechanism
  brainstorm, plan design, persona rotations on stuck loops.

Pair with submission-budget discipline: use the daily 10/day, don't
sit on slots.
**Prevents**: top-tier model for routine `ls` calls. Disproportionate
cost vs lift.

## How to check yourself

Before any LB submit, answer in audit:

- [ ] G1 G2 G3 G4 all PASS?
- [ ] Minimal-input meta beats anchor?
- [ ] PI explicitly approved this submit?
- [ ] Single-shot, not in any loop?
- [ ] Candidate is not already in `kaggle competitions submissions`
      output?

Before declaring saturation:

- [ ] Did I run the Research-loop? (web search + 2 prior-comp
      writeups + 5 untried mechanisms listed)
- [ ] Did I rotate at least one persona on this problem?
- [ ] Have I checked `LEARNINGS.md` for similar plateau patterns?
