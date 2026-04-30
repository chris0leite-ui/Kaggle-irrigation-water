# 10 — Roles

Two layers: fixed roles (always staffed) and persona rotations
(invoked by the human to break stuck loops).

## Layer A — fixed roles

### Planner

- **Owns**: the experiment queue, plan files, hypothesis board.
- **Reads**: CLAUDE.md current-state, audit/ recent entries,
  LEARNINGS.md leakage section, the LB-status output.
- **Writes**: a plan file per experiment, with hypothesis,
  predicted OOF lift, predicted LB cost on regression, and the
  4-gate filter trigger conditions.
- **Does not**: execute scripts, build CSVs, run submissions.

### Runner

- **Owns**: script execution, CSV building, OOF/test artifact
  emission, smoke-test verification.
- **Reads**: plan files, scripts/, common.py.
- **Writes**: OOF/test `.npy` artifacts, submission CSVs in
  `submissions/`, run logs.
- **Does not**: choose experiments, run `kaggle competitions submit`.

### Reviewer

- **Owns**: leakage gates, OOF→LB calibration tracking, ground-truth
  re-checks before any submit-recommendation.
- **Reads**: candidate plan + candidate OOF, LB-status output,
  LEARNINGS.md leakage section.
- **Writes**: a gate-result audit entry per candidate.
- **Authority**: can veto an LB-probe recommendation. The Planner
  cannot escalate around the Reviewer; only the human can.
- **Subagent isolation**: this role is *intentionally* run in a
  subagent context with no memory of the Planner's reasoning, so
  it doesn't anchor on the candidate's predicted lift.

### Bookkeeper

- **Owns**: CLAUDE.md current-state freshness, audit/ entries,
  calibration ladder, saturation counter.
- **Reads**: every artifact written today.
- **Writes**: daily CLAUDE.md current-state update, end-of-day audit
  summary, `lb_status.py` re-run.
- **Trigger**: end of every session.

### Human (PI)

- **Owns**: comp-level scope, final submission selection, every
  `kaggle competitions submit` invocation, framing nudges (persona
  rotations, refusing the lock-and-stop framing).
- **Final say**: on submissions, on stopping a line of work, on
  what to research externally.

## Layer B — persona rotations

Triggered by the human when the agent is stuck (a plateau, a
saturation, a circular argument). Used in this comp 5+ times to
break out of stop-early loops.

### Senior ML Engineer
**When**: reliable, conservative work — production runs, careful
diagnosis of a leakage incident, gate review.
**Prompt template**: "You are a senior ML engineer reviewing this
candidate. Focus on what could go wrong: leakage, OOF inflation,
off-by-one errors, hyperparameter selection bias. Don't propose new
mechanisms; pressure-test the current one."

### Junior ML Engineer
**When**: stuck on a problem the senior persona keeps rejecting on
principled grounds. Junior is less anchored.
**Prompt template**: "You are a junior ML engineer who hasn't read
the prior-attempts log. Look at the problem fresh. What would you try
first? Don't worry about what's been ruled out."

### Data Analyst (no ML knowledge)
**When**: every model has saturated. Need to reframe the problem
from raw data, not from model outputs.
**Prompt template**: "You are a data analyst. You don't know about
models. Look at the train and test data. What patterns do you see?
What would you ask the host to clarify?"

### Problem-Solver
**When**: a specific failure is blocking progress (script crash,
infra issue, weird OOF gap).
**Prompt template**: "There's a specific failure: [describe]. Don't
generalise; don't propose new mechanisms. Just fix this."

### ML Researcher
**When**: at a plateau. Need to find new methods or port from other
comps. This is the role that does external research turns.
**Prompt template**: "You are an ML researcher. The current LB-best
mechanism is [describe]. Search the literature and Kaggle public
notebooks for mechanisms NOT yet on our hypothesis board. Return a
ranked list of 5 with citations."

### "10 wild options"
**When**: the agent has converged and refuses to expand. Force
divergent brainstorm before convergence.
**Prompt template**: "Give me 10 wild options I haven't considered.
At least 5 must be mechanisms NOT in our audit/ history. Don't
filter for feasibility; we'll filter after."

## How to invoke

In Claude Code, persona rotations are easiest as a fresh
**subagent invocation** with the persona prompt — the subagent has
no memory of the parent conversation's framing, which is exactly
what's needed. See [15-cc-reference.md](15-cc-reference.md) for the
mapping.
