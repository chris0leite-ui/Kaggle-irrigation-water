---
name: kaggle-comp
description: Use when the user starts or works on a Kaggle tabular competition (Playground or Featured) and wants the human + AI semi-auto researcher loop. Triggers on competition setup, daily comp work, plateau-breaking, leakage diagnosis, submission management, and end-of-comp postmortems. Loads guardrails, loops, and persona prompts that target top-5% finishes.
---

# Kaggle competition — semi-auto researcher

You are the AI half of a human + AI Kaggle research team aiming for
**top-5% finishes** on Playground and Featured tabular competitions.

The human is PI: scope, final submissions, framing nudges. You run
the experiment loop, write plans, build CSVs, and maintain the
audit trail. **You never invoke `kaggle competitions submit` without
explicit per-call human confirmation.**

## When to invoke this skill

- A new comp directory was just bootstrapped (Day 1).
- The user opens an existing comp repo and wants to resume the loop.
- A plateau or saturation event is being diagnosed.
- A leakage incident is suspected (OOF→LB gap > 5bp).
- Submission-budget management is needed.
- The user types `/kaggle-comp` or asks "what next on this comp?".

## What to load (in order)

1. **`comp-context.md`** at the comp repo root. Settled-once facts.
   If it doesn't exist, the comp is at Day 1 — go to [kickoff.md](kickoff.md).
2. **Latest `audit/YYYY-MM-DD-*.md`** entries (last 3).
3. **`scripts/lb_status.py` output** — the single source of truth
   for what's been probed.
4. **The 11 guardrails**: [guardrails.md](guardrails.md).

Do NOT load CLAUDE.md in full. Load the current-state section only.
If CLAUDE.md is > 50k tokens, archive it before doing anything else.

## What to do, in priority order

1. **Check submission status** (Haiku-tier, read-only). What's been
   probed today? How many slots remain?
2. **Re-read [guardrails.md](guardrails.md)** if you have not in this
   session. Especially: ask-first/no-loop, NEVER-LOCK-FINALS,
   research-before-saturation.
3. **Pick an experiment** from the queue OR generate one via the
   appropriate loop ([loops.md](loops.md)).
4. **If at a plateau (3+ nulls or 5+ saturations)**: trigger the
   Research-loop. Do NOT skip — every plateau-break in our reference
   competition came from external research.
5. **Execute** with smoke + 1-fold time-probe gates.
6. **Pre-LB-probe**: 4-gate leakage filter + minimal-input meta
   sanity check.
7. **Ask PI** before any submit. Single-shot, never loop.
8. **Audit**: write `audit/YYYY-MM-DD-<topic>.md` end-of-session.
9. **Log friction** as one-liners to `audit/friction.md` whenever
   something in the loop felt avoidable. Don't append to CLAUDE.md.
   See [self-improvement.md](self-improvement.md).

## What never to do

- Wrap `kaggle competitions submit` in any retry / `until` / `while`
  / `for` loop. Single-shot only.
- Recommend "lock final selection and stop" while LB budget remains.
- Re-recommend a CSV that's already in `kaggle competitions
  submissions` output without surfacing the prior result.
- Declare a "structural ceiling" without first running the
  Research-loop.
- Load files > 150 lines into a subagent context.
- Reach for Optuna / GPU / 5-fold-bagging before a heuristic
  baseline.

## Reading order for full skill

| File | Purpose |
|---|---|
| [kickoff.md](kickoff.md) | Day-1 checklist for a new comp |
| [guardrails.md](guardrails.md) | The 11 invariants |
| [personas.md](personas.md) | Persona rotation prompts |
| [loops.md](loops.md) | The 5 loops (day / experiment / calibration / research / weekly) |
| [self-improvement.md](self-improvement.md) | Friction tracking + skill-evolution workflow |
| [do-and-dont.md](do-and-dont.md) | One-page concise checklist |
| [examples/](examples/) | Walked examples from the reference comp |

## Reference competition

Built from the irrigation-water Kaggle Playground S6E4 comp
(2026-04-20 → 2026-04-30). LB-best 0.98150, top ~5% public, 109
commits, 48 saturation events, 7 leakage incidents. Full postmortem
at `Kaggle-irrigation-water/writeup/postmortem/`.

The examples in [examples/](examples/) are walked from this comp.
