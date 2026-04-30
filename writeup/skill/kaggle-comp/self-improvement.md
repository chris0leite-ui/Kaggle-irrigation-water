# Self-improvement — friction tracking + skill evolution

The skill's job is to get better at its job. Every comp surfaces
new friction (re-asked questions, wasted slots, slow loops, things
the human has to nag the agent about). This file describes how to
**capture friction without spamming CLAUDE.md** and **distill it
into skill edits**.

## The two-file model

| File | Where | Purpose | Update cadence |
|---|---|---|---|
| `audit/friction.md` | Per-comp repo | One-liner per friction event | Append-only during the day |
| `~/.claude/skills/kaggle-comp/improvements.md` | The skill itself | Cross-comp distilled patterns | Edited weekly + at end of comp |

**Both are ≤150 lines.** Both are NOT CLAUDE.md. CLAUDE.md stays a
running log of *experiments*; friction logs are separate.

## What goes in `audit/friction.md`

One-liner per event. Terse. No prose. Format:

```
YYYY-MM-DD  <tag>          <one-line description>
```

Tags (use existing ones; create new sparingly):

- `retry-loop`        — submission / pipeline wrapped in a retry
- `re-recommend`      — recommended a CSV already in LB submissions
- `context-bloat`     — file > 50k tokens / 150 lines
- `settled-once`      — re-asked a Day-1-settled question
- `persona-stuck`     — same persona output for 3+ rotations
- `gate-skip`         — LB-probed without 4-gate / minimal-input check
- `gpu-overshoot`     — kernel ran past projected wall time
- `update-cadence`    — minute-level chatter or hour-long silence
- `model-misroute`    — Opus-tier on a Haiku-tier task
- `tool-missing`      — wanted a slash command / hook / MCP that doesn't exist

Example real entries (synthesized from this comp's incidents):

```
2026-04-26  retry-loop      until-loop on kaggle submit, case-mismatched success marker, +3 wasted slots
2026-04-30  re-recommend    rawashishsin_k4 recommended as unprobed; LB 0.98112 from 8h prior
2026-04-30  context-bloat   CLAUDE.md crossed 1MB, subagent context timeout
2026-04-25  settled-once    asked "what's the public-LB split %?" 4th time
2026-04-24  gpu-overshoot   pytabkit RealMLP 3h34min CPU pre-train, killed
2026-04-28  gate-skip       R2 hybrid LB-probed without minimal-input meta, -0.00046
2026-04-29  tool-missing    wanted /gate slash command, hand-ran the 4 checks
```

## Capture rules

1. **One line max.** If you can't compress it to a tagline, the
   entry is too detailed for the friction log — file an audit
   postmortem entry instead.
2. **Append-only.** Don't edit prior entries. Don't rewrite them
   into prose. The chronology IS the signal.
3. **Tag matters more than the description.** When 3+ entries share
   a tag, you have an automation candidate.
4. **Don't burn LLM tokens summarizing.** Friction logging is a
   2-second append, not a paragraph. Use `echo "..." >> audit/friction.md`.

## Anti-spam: what does NOT go in friction.md

- Successful experiments (those go in `audit/YYYY-MM-DD-*.md`).
- LB results (those go in calibration ladder + CLAUDE.md current-state).
- Hypothesis-board churn (that's CLAUDE.md).
- Reasoning prose (that's plan files in `~/.claude/plans/`).

If something is worth a paragraph, it's not friction — it's a real
postmortem.

## Weekly distillation (part of Weekly-loop)

```
1. Read audit/friction.md (one scan, ≤30s).
2. Tag-frequency count: any tag with ≥3 entries this week?
3. For each high-frequency tag, decide:
   a. Already covered by a guardrail? → tighten the guardrail
      (edit skill/kaggle-comp/guardrails.md).
   b. Not covered? → add a new guardrail or walked example.
   c. Automatable? → see "Automation candidates" below.
4. Commit skill edits. Reset friction.md to empty (archive prior
   week to audit/friction-archive-YYYY-WW.md).
```

The output of distillation is **edits to the skill itself**, not new
CLAUDE.md content. The skill is the durable artifact; CLAUDE.md is
the running log.

## Automation candidates (the most actionable distillation)

When a friction tag recurs, ask: can this be automated away
entirely? Common targets:

| Friction tag | Automation |
|---|---|
| `settled-once` | SessionStart hook that auto-loads `comp-context.md` |
| `re-recommend` | Pre-recommend hook that runs `lb_status.py` and aborts if candidate name appears |
| `retry-loop` | Permission rule blocking `kaggle competitions submit` from any wrapped context |
| `context-bloat` | Pre-write hook that rejects writes pushing a file > 150 lines |
| `gate-skip` | `/gate <candidate>` slash command that runs the 4 checks |
| `gpu-overshoot` | Pre-push hook on Kaggle kernels requiring a SMOKE artifact + 1-fold time-probe log |
| `tool-missing` | Add the requested slash command to the skill |

Each automation is a one-time edit to `.claude/settings.json`,
`.claude/commands/`, or the skill itself. After automation, the
friction tag should not recur — that's the verification.

## Cross-comp distillation (`improvements.md`)

At the end of each comp (or quarterly), review per-comp
`friction.md` archives against the skill. Promote anything that:

- Appeared in 2+ comps, OR
- Cost > 1 LB slot OR > 1h of agent time, OR
- Required human nag to be fixed.

Edits go into the skill files. Concrete log of the changes lives
in `improvements.md`:

```
2026-04-30  guardrails.md  added invariant #11 model-routing
                           (3+ comp friction entries: model-misroute)
2026-05-08  hooks/         added pre-recommend hook for re-recommend
                           friction (auto-runs lb_status.py)
```

## How the skill knows it's getting better

- **Friction-tag entropy decreasing**: same tags comp-after-comp
  means the skill isn't learning. New tags = new exposure (good).
- **Time-to-first-LB decreasing**: Day-1 setup time should shrink.
- **Submission slot wastage → 0**: retry-loop, re-recommend,
  gate-skip should fall off once automated.
- **Plateau-break time decreasing**: faster Research-loop via
  better persona prompts and pre-staged citation lists.

## Don't

- ❌ Don't append friction to CLAUDE.md. CLAUDE.md is the running
  log, not the friction log.
- ❌ Don't write friction entries longer than one line.
- ❌ Don't try to fix every friction in real time. Capture it,
  distill weekly.
- ❌ Don't let friction.md grow past 150 lines without rotating.
- ❌ Don't archive a friction tag without auditing whether it's
  actually been automated. "Archived" without a guardrail edit is
  just sweeping it under the rug.
