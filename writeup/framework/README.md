# Framework — human + AI semi-auto researcher

The framework's goal: place reliably in the **top 5%** on Kaggle
Playground and Featured tabular competitions, with the AI assistant
running most of the loop autonomously and the human acting as PI
(scope, final submissions, framing nudges).

## Design principles

1. **The agent runs the loop; the human owns scope and submissions.**
   The agent picks experiments, writes plans, executes, evaluates,
   and updates the audit trail. The human approves every Kaggle
   submission and every change to comp-level scope.

2. **Visible artifacts make autonomy auditable.** Every experiment
   produces an audit entry (`audit/YYYY-MM-DD-*.md`); every plateau
   produces a calibration entry; every saturation increments a
   counter. The human reads audit/ end-of-day to follow what
   happened without watching every step.

3. **Heuristics before heavy compute.** Always probe a closed-form
   rule, a threshold, or a hand-coded baseline before reaching for
   Optuna / GPU / 5-fold-bagging.

4. **External research at every plateau.** The agent's introspection
   converges on already-explored mechanism families. Plateau-breaks
   come from web search / public notebook reading / prior-comp
   writeups.

5. **Calibration over speed.** OOF→LB gap is the single most
   important number. Track it; trust it; don't ship a candidate
   whose mechanism inflates it.

6. **Modular files, fresh context.** All agent-loaded docs ≤150
   lines / ≤50k tokens. Archive-on-bloat. No file the agent reads
   should be more than ~3 minutes of human reading.

## Reading order

| File | Topic |
|---|---|
| [10-roles.md](10-roles.md) | Fixed roles + persona rotations |
| [11-loops.md](11-loops.md) | Day / Experiment / Calibration / Research / Weekly |
| [12-guardrails.md](12-guardrails.md) | Eleven invariants |
| [13-repo-template.md](13-repo-template.md) | Folder skeleton + lift list |
| [14-kickoff-playbook.md](14-kickoff-playbook.md) | Day-1 → final-day checklist |
| [15-cc-reference.md](15-cc-reference.md) | Claude Code mapping (sidebar) |

## Tool-agnostic vs Claude Code-specific

The framework is described in tool-neutral terms (roles, loops,
hooks, gates). [15-cc-reference.md](15-cc-reference.md) shows how
each piece maps to Claude Code mechanisms (slash commands,
SessionStart hooks, subagents, MCP servers, Skills). The narrative
files don't depend on Claude Code; the [skill](../skill/kaggle-comp/)
package does.

## Scope of the framework

**In scope**:

- Kaggle Playground tabular comps (Season-X-Episode-Y format).
- Kaggle Featured tabular comps with similar structure (3-class /
  binary / regression on tabular features, public LB + private LB,
  daily submission budget).

**Out of scope** for the first version:

- CV / NLP / code / simulation comps (different infra needs).
- Multi-team coordination beyond one human PI + one AI assistant.
- Kaggle Notebooks-only comps where you submit code, not CSVs
  (some pieces still apply, but the submission loop differs).

## What we're optimizing

Top 5% reliably means ranking better than rank `0.05 × N` on the
public LB at deadline. On Playground (~2000-3000 teams) that's rank
~150. On Featured (~5000+ teams) that's rank ~250+. The framework
optimises for *consistent placement*, not for shooting at gold.
