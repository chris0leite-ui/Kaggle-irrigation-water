# 05 — Coordination challenges

The most expensive failures in this comp were not technical. They
were coordination failures between the human and the AI assistant.
Ten distinct patterns appeared often enough to deserve names.

## 1. Context loss between sessions

CLAUDE.md grew past 50k tokens; some loaded contexts crossed 1MB.
Subagents loaded the full file per call, exploding token spend and
triggering Anthropic API idle timeouts on long-file writes. The agent
also re-recommended already-submitted CSVs because its working memory
of "what's been probed" lagged the Kaggle CLI ground truth.

**Fix**: `kaggle competitions submissions` is the single source of
truth for what's been probed. Check it before recommending any
candidate. Archive CLAUDE.md when it grows past ~50k tokens. Keep
files modular and ≤150 lines.

## 2. Stop-early bias / structural-ceiling fallacy

The agent's default disposition was to stop early. At every plateau
it argued the ceiling was structural and recommended locking
submissions. Refuted at every plateau: 0.97097 → 0.97296 → … →
0.98150. The breakthrough mechanisms (override rules, RF on natural
cal bank, triple-consensus) were *all* on the agent's "skip on
principled grounds" list at one point.

**Fix from the human side**: refuse the lock-and-stop framing. Demand
3 untried mechanisms after every null. The CLAUDE.md NEVER-GIVE-UP
and NEVER-LOCK-FINALS rules exist *because of this bias*. Treat
saturation evidence as bounded ("we tested known levers"), not as
ceiling proof.

## 3. OOF→LB calibration drift / leakage

Selecting on OOF inflates the score by 5–50 bp on saturated banks.
Seven incidents cost ~0.0045 LB total (see
[04-what-failed.md](04-what-failed.md) §2).

**Fix**: 4-gate filter + minimal-input-meta sanity check before every
LB probe. Theory-only hyperparameters (no grid search on OOF for
final picks). Cross-meta error correlation gate (skip if pairwise
Jaccard ≥ 0.85 on error rows).

## 4. Submission-budget management

One retry-loop burned 4 slots on a single CSV (case-mismatched
success marker). Once-tested candidates got re-recommended because
ground truth wasn't checked.

**Fix**: ask-first protocol. Single-shot `kaggle submit` invocations
only — never wrap in a retry / `until` / `while` / `for` loop.
Monitors that POLL submission status are fine; monitors that WRITE
submissions are forbidden. AND: *use* the daily 10/day budget. Don't
sit on slots; that's its own failure mode.

## 5. External-knowledge starvation

Real lift came from researching what others were doing: public
notebooks, Kaggle discussions, prior-comp writeups, papers. The agent
left to itself converged onto already-explored mechanism families.
Every plateau-break in this comp came from external reading.

**Fix**: schedule explicit research turns at every plateau. Web
search top-N notebooks for the comp slug. Read 1-2 prior-comp
writeups in the same domain. List mechanisms NOT yet tried locally,
*before* declaring saturation.

## 6. Heavy-compute-before-heuristics

Agent's instinct was 5-fold-bagging / GPU kernels / Optuna sweeps.
Cheaper signal was usually available from a closed-form rule, a
threshold, or a hand-coded baseline.

**Fix**: heuristic-first. Always try the dumbest possible thing first
to bound the lift available. Smoke-test + 1-fold time-probe before
any multi-hour run — Kaggle GPU wall-time *estimates* were not
reliable; only a same-hardware 1-fold probe is trustworthy. The
RealMLP kernel that ate 3h34min of CPU before training started would
have been caught by a 1-fold probe in 5 min.

## 7. Re-asking settled-once questions

The agent kept asking mid-run: is the LB stable? what's the public
split %? what's the eval metric? These are facts to settle ONCE on
Day 1 from the comp page. Re-asking burns user attention and tokens.

**Fix**: a `comp-context.md` fixed-fact file at the repo root,
populated on Day 1, loaded by every subagent on session start. After
Day 1, no re-asking allowed.

## 8. CLAUDE.md context bloat

CLAUDE.md crossed 1MB at one point. Files this size:

- Trigger Anthropic API idle timeouts on long writes.
- Force every subagent to load 100k+ tokens before doing anything.
- Make it impossible for the agent to find the relevant section.

**Fix**: archive-on-bloat. CLAUDE.md was archived once during this
comp (`audit/CLAUDE-md-archive-2026-04-30.md`). Strict file-size cap
≤150 lines / ≤50k tokens for any agent-loaded doc. Split guidance
into small modular files; let subagents load slices selectively.

## 9. Update cadence wrong

When long jobs ran, the agent either went silent for an hour or
pushed minute-level chatter. The human wanted *pull-style* updates:
"what's the latest fact?" answered in 1-2 sentences, on demand. Not
push-style every-N-minutes.

**Fix**: communication guardrail. No proactive minute-level updates.
On-pull: 1-2 sentence summary of the latest concrete fact, no recap.

## 10. Token cost / model routing

The whole project was expensive. Top-of-the-line model on every call,
including routine read-only checks (`ls`, grepping LB status, file-
exists checks). Model tier was not matched to task difficulty.

**Fix**: route by task. Haiku-tier (cheap) for routine read-only
checks, smoke verifications, file existence. Sonnet for default work.
Opus for hard reasoning (plan design, leakage diagnosis, novel
mechanism brainstorm). Cap subagent context loads. Submission-budget
discipline pairs with token-budget discipline.

## What the framework will do about all this

See [framework/12-guardrails.md](../framework/12-guardrails.md) — each
of the ten patterns above maps to one of the eleven invariants the
framework codifies.
