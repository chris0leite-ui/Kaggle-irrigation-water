# Example — settled-once Q&A

Some facts about a Kaggle comp don't change across the comp's
lifecycle. The agent should ask them once on Day 1, write the
answers to `comp-context.md`, and never re-ask.

## What goes in comp-context.md

```yaml
# Filled out once on Day 1. Never re-asked. Never re-derived.
slug: playground-series-s6e4
url: https://www.kaggle.com/competitions/playground-series-s6e4
task: 3-class classification (Low / Medium / High)
metric: balanced accuracy (macro-recall)
public_split_pct: 20
lb_stability: stable    # ← settled once
train_rows: 630_000
test_rows: 270_000
feature_count: { numeric: 11, categorical: 8 }
class_priors: { Low: 0.587, Medium: 0.379, High: 0.033 }
deadline: 2026-04-30
team_size_limit: 3
submission_budget: 10/day
final_submissions: 2
data_license: CC BY 4.0
external_data_allowed: yes  # original Irrigation Prediction dataset
                            # is allowed (per host post)
lb_best_at_kickoff: 0.98219
pack_score_at_rank_100: 0.98114
probe_resolution_floor: 0.00005  # 80/20 split × 270k test
```

## What the agent kept re-asking

In the irrigation-water comp, the agent re-asked these mid-run
several times:

1. "Should we probe LB stability before relying on the public-LB
   gap as our calibration anchor?"
2. "What's the public/private split — is it 20/80?"
3. "What's the team size limit?"
4. "Are we allowed to use external data?"

Each of these is settled by reading the comp page on Day 1. The
agent kept re-asking because:

- The information wasn't in CLAUDE.md current-state.
- Subagents started fresh and hadn't seen prior turns.
- The session-start hook didn't auto-load `comp-context.md`.

## The fix

### 1. Auto-load comp-context.md at session start

`.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "cat comp-context.md"
          }
        ]
      }
    ]
  }
}
```

This puts the settled facts into every session's initial context.

### 2. Subagents inherit comp-context.md

When invoking a subagent, prepend a line:

```
Subagent prompt:
"<Persona / role prompt>

Comp context (settled facts, do not re-derive):
<paste comp-context.md>

Task: ..."
```

This costs ~50 tokens per subagent call but eliminates the
re-asking failure mode entirely.

### 3. Make re-asking a guardrail violation

In SKILL.md, list the facts that are settled-once. Any subagent
that asks about them should be redirected to `comp-context.md`.

## Edge cases

- **"LB stability could change."** It can't — it's a property of
  the host's split methodology, not the data. If you see OOF→LB
  drift > 5bp on calibrated mechanisms, you've found leakage or
  distribution shift, not LB instability.
- **"Team size could change."** It can't. The team-merger deadline
  IS settled-once and lives in `comp-context.md`.
- **"External-data rules are ambiguous."** Ask the host on the
  forum Day 1. Capture the answer. Don't re-ask internally Day 5.
- **"LB-best at kickoff goes stale."** Yes — that's why the
  current LB-best lives in CLAUDE.md current-state (daily-updated),
  not in `comp-context.md`. Three categories: settled-once,
  daily-updated, per-experiment. Don't conflate them.

## The portable rule

Maintain three categories:

1. **Settled-once facts** (`comp-context.md`): asked Day 1, never
   re-asked.
2. **Daily-updated facts** (in CLAUDE.md current-state): top of
   leaderboard, our LB-best, submissions used today.
3. **Per-experiment facts** (in audit/): OOF, LB result, gate
   verdicts.

Subagents auto-load category 1 + the relevant slice of category
2-3. They never need to re-derive category 1.

## Why this matters

Re-asking settled-once facts:

- Burns tokens (30-200 per re-ask).
- Burns user attention.
- Erodes trust in the agent's bookkeeping.
- Creates session-to-session drift on facts that should be stable.

The Day-1 cost of writing `comp-context.md` is ~5 minutes. The
running savings across a 30-day comp are large.
