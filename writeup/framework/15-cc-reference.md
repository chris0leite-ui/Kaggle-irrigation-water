# 15 — Claude Code reference implementation

How each piece of the framework maps to Claude Code mechanisms. The
framework itself is tool-agnostic; this file is a sidebar.

## Mechanism map

| Framework piece | Claude Code mechanism |
|---|---|
| Fixed roles (Planner/Runner/Reviewer/Bookkeeper) | **Subagents** with role-specific system prompts. Each subagent has its own context. |
| Persona rotations (Senior/Junior/Researcher/etc.) | **Subagent invocations** with persona prompt. Fresh context = no anchoring. |
| Day-loop state load | **SessionStart hook** that runs a script printing: last 3 audit entries, current LB best, today's submission count, today's plan-file. |
| Submission ask-first gate | **AskUserQuestion** before any `kaggle competitions submit`. |
| Submission never-loop | **Permission rule** in `settings.json` blocking `kaggle competitions submit` from any retry/loop wrapper. |
| Smoke test gate | **Bash hook** that requires a `_smoke_*` artifact before any production-config push. |
| 4-gate leakage filter | A **slash command** `/gate <candidate>` that runs the 4 checks and emits an audit entry. |
| Calibration ladder refresh | A **slash command** `/calibrate` that re-runs `lb_status.py` + builds the ladder. |
| Research-loop trigger | A **slash command** `/research` that rotates the ML Researcher persona and runs WebSearch. |
| Heuristic-first | A **slash command** `/heuristic` that scaffolds a closed-form / threshold baseline before any tree run. |
| Plan files | **Plan mode** + `~/.claude/plans/<plan>.md`. |
| Comp-context.md | A **CLAUDE.md** include line that auto-loads it on session start. |
| LB-status check | A **MCP server** wrapping `kaggle competitions submissions` (read-only). |
| Long-job updates (pull-style) | A **Monitor** tool reading the running script's stdout; human pulls when wanted. |
| Model routing (Haiku/Sonnet/Opus) | **Subagent model overrides**. Cheap subagents for read-only checks; expensive for plan design. |
| Skill bundle | `~/.claude/skills/kaggle-comp/SKILL.md` + supporting files (see [../skill/kaggle-comp/](../skill/kaggle-comp/)). |

## Concrete files to wire up

### `.claude/settings.json` (project-local)

- Permission allow: `Bash(kaggle competitions submissions:*)` — read
  the LB.
- Permission deny: `Bash(kaggle competitions submit:*)` unless
  approved per-call (use the AskUserQuestion gate).
- Hook: `SessionStart` runs `scripts/session-load.sh` that prints
  state load.

### `~/.claude/skills/kaggle-comp/SKILL.md`

The drop-in skill (see [../skill/kaggle-comp/SKILL.md](../skill/kaggle-comp/SKILL.md)).
Triggers when the user starts a new Kaggle comp session or types
`/kaggle-kickoff`.

### Slash commands

```
/gate <candidate>     run 4-gate filter
/calibrate            refresh LB-OOF gap
/research             plateau-break research turn
/heuristic            scaffold a heuristic baseline
/persona <name>       rotate persona (senior/junior/analyst/researcher/wild10)
```

These can ship inside the skill or as project-local commands.

## What about the `loop` skill?

The `loop` skill is fine for **monitoring** — polling LB submissions
(read-only), watching a script's stdout, checking artifact emission.

It is **forbidden** for any `kaggle competitions submit` invocation,
per the no-loop-on-submits guardrail. Wire this into permission
rules: `kaggle competitions submit` should not be allowlisted from
any background-loop context.

## What's NOT in Claude Code yet (worth proposing)

- A first-class **calibration tracker**: built-in widget that shows
  the running OOF→LB gap per mechanism family. Currently we
  implement this in `scripts/lb_status.py` + a markdown ladder.
- A first-class **submission budget meter**: the daily 10/day count
  visible at session start. Currently we eyeball
  `kaggle competitions submissions`.
- **Persona rotations as a first-class concept**. Currently we
  hand-roll subagent invocations with persona prompts. A
  `/persona <name>` shorthand would help.

## What we use that Claude Code already does well

- **Plan mode**: keeps the agent honest about not editing during
  research phase. Used for every plan in `~/.claude/plans/`.
- **Subagents with model overrides**: the Reviewer ran on Sonnet
  while the Planner ran on Opus, etc. This is the easiest way to
  do model routing.
- **Skills**: kickoff (`kaggle-kickoff`), review, security-review
  are all already useful skills. Our `kaggle-comp` skill adds the
  guardrails + loops + personas as a unit.
- **MCP servers**: the Kaggle CLI wrapping is naturally an MCP
  server (read-only, well-typed, reusable across comps).
