# Kaggle Irrigation Water — Write-up

A postmortem of one Kaggle competition and the framework we want to lift
out of it for future ones.

We finished **Playground S6E4 — Predicting Irrigation Need** at LB
**0.98150** (top ~5% public). Across 10 days and 109 commits, we logged
48 saturation events, paid ~0.0045 LB to leakage incidents, and burned a
handful of submission slots to coordination failures. The point of this
write-up is to convert that experience into a reusable human + AI
*semi-auto researcher* that can hit top-5% reliably on Playground and
Featured tabular comps.

## How to read this

Audience: ML basics is enough — we explain jargon as it appears (see
[B-glossary.md](appendix/B-glossary.md)). Tone is technical but plain.

Two surfaces, share content:

1. **Narrative form** — `postmortem/` and `framework/` markdown. Read
   start-to-finish like a report.
2. **Operational form** — `skill/kaggle-comp/`. A drop-in Claude Code
   Skill the user copies to `~/.claude/skills/` for the next comp. It
   condenses the framework into agent-loadable instructions.

## Reading order

| If you want… | Read |
|---|---|
| The 60-second summary | [postmortem/01-overview.md](postmortem/01-overview.md) |
| What actually moved the LB | [postmortem/03-what-worked.md](postmortem/03-what-worked.md) |
| Where coordination broke | [postmortem/05-coordination.md](postmortem/05-coordination.md) |
| Final standing + private-LB shake | [postmortem/06-final-results.md](postmortem/06-final-results.md) |
| Recommendations for next comp | [postmortem/07-next-comp-recommendations.md](postmortem/07-next-comp-recommendations.md) |
| The framework spec | [framework/README.md](framework/README.md) |
| To set up next comp | [framework/14-kickoff-playbook.md](framework/14-kickoff-playbook.md) |
| Just the rules | [framework/12-guardrails.md](framework/12-guardrails.md) |
| Skill to install | [skill/kaggle-comp/SKILL.md](skill/kaggle-comp/SKILL.md) |

## Core lessons (the TL;DR)

1. **The agent's default disposition is to stop early.** It calls
   ceilings structural and wants to lock submissions. The human has to
   refuse this framing — every plateau in this comp was broken by a
   mechanism the agent had previously rejected on principled grounds.
2. **Real lift comes from external research, not introspection.** Once
   the agent had explored its own mechanism families it converged.
   Plateau-breaks needed web search / public-notebook reading /
   prior-comp writeups.
3. **OOF→LB calibration drifts under stacking.** Selecting on OOF
   inflates ~5–50bp of LB regression. Use a 4-gate filter and a
   minimal-input-meta sanity check before every LB probe.
4. **Submission slots are precious; never loop submits.** One
   case-mismatched retry-loop burned 4 slots. Ask first, single-shot.
   But also: *use* the daily 10/day budget, don't sit on slots.
5. **Heuristics before heavy compute.** Closed-form rules,
   thresholds, and hand-coded baselines beat Optuna sweeps and GPU
   kernels on time-to-signal.
6. **Files small, context fresh.** CLAUDE.md crossing 1MB triggered
   API timeouts and burned tokens. Cap files at ~150 lines and archive
   on bloat.

## Repo layout

```
writeup/
├── README.md                  ← you are here
├── postmortem/                What happened
├── framework/                 What to build for next time
├── appendix/                  Reference tables
└── skill/kaggle-comp/         Drop-in Claude Code Skill
```

## Provenance

This document was written 2026-04-30, the deadline day of the
competition. All numbers are taken from `git log main`, `CLAUDE.md`,
`LEARNINGS.md`, `REPORT.md`, and the `audit/` postmortems committed
during the comp. We did not re-run experiments to write this — every
result cited is from the committed record.
