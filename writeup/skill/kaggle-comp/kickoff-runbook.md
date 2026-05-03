# Kickoff runbook — agent step-by-step

User invokes by saying "let's do the kickoff" / "start a new comp"
(see SKILL.md trigger). **Conversational free-form**: each numbered
step is one chat turn (≤4 sentences) followed by a wait for PI
reply, OR one Bash batch from [kickoff-bash.md](kickoff-bash.md).

**Hard rule**: never invoke `kaggle competitions submit` without the
explicit Q6 ask-PI gate. Single-shot only.

## Sequence

### Pre-flight (silent unless missing)

Run `pre-flight` from kickoff-bash.md. If `MISSING_TOKEN`: ask PI
in chat for the token (single-shot, redact in logs). If network
fails: ask PI for fix path; do NOT loop.

### Q1 — chat

> "Which Kaggle competition slug? (e.g., `playground-series-s7e1`,
> visible in the comp URL after `/competitions/`)"

Validate with `kaggle competitions view <slug>`; if errors, re-ask.

### Q2 — chat

> "Where should I create the comp directory? Default
> `~/projects/<slug>/`."

### Bash batch A — scaffold

Run `batch-A-scaffold` from kickoff-bash.md. Copies templates,
substitutes `{{COMP_SLUG}}`, `git init`.

### Bash batch B — populate comp-context.md from API

Run `batch-B-context` from kickoff-bash.md. Pulls from
`kaggle competitions view` + `leaderboard --download`.

### Q3 — chat (batch confirm)

> "comp-context.md auto-populated from the Kaggle API. Here it is —
> anything to fix or add? [paste populated YAML]"

PI replies any format. Apply edits, write final `comp-context.md`.

### Q4 — chat (4 strategic questions in one batch)

> "Strategic questions — answer in any format, one paragraph fine:
>
> (a) **LB stability** — stable / per-row-seeded / unknown?
> (b) **External data** — use / skip / depends_on_rules?
> (c) **Time budget** — hours/day × days?
> (d) **Compute** — CPU only / Kaggle GPU / local GPU?"

Extract free-form answers into structured fields; write to
`comp-context.md`. Show PI a 2-line interpretation summary for
one-line confirm.

### Bash batch C — download data + EDA

Run `batch-C-data` from kickoff-bash.md. Downloads data, extracts
target_col + id_col from sample_submission.csv header, runs EDA.

If multi-output: ask PI which target is primary; default to first
non-id column.

### Q5 — chat (EDA summary)

> "EDA done. Top features: `<top-3 from eda-summary.md>`.
> Anything to flag for hand engineering before the baseline?
> (Default: skip FE for the baseline; we add FE only after the DGP
> is understood — guardrail #6.)"

Capture any FE notes in `audit/<date>-day-1-kickoff.md`.

### Bash batch D — baseline LGBM

Run `batch-D-baseline` from kickoff-bash.md. Reads OOF + fold-std
from `scripts/artifacts/baseline_lgbm_results.json`.

If baseline fails: 1-sentence summary, ask PI for fix path. Do NOT
loop.

### Q6 — chat (ASK PI BEFORE SUBMIT — critical)

> "Baseline OOF=`$OOF`, fold-std=`$STD`. Submitting calibrates the
> OOF→public-LB gap (Day-1's whole point). 1 of `$DAILY` slots
> today. Submit `submission_baseline_lgbm.csv`?
> [yes / no / show me first]"

**Wait for explicit PI yes.** If no: skip submit, log in audit. If
"show me first": print first 5 + last 5 rows of the CSV; re-ask.

### Bash D2 (only on PI yes) — single-shot submit

Run `batch-D2-submit` from kickoff-bash.md. **Never wrap in a loop.**
If kaggle command fails, surface the error verbatim, ask whether to
retry. PI decides.

### Bash batch E — Day-1 audit + Day-2 queue

Run `batch-E-audit` from kickoff-bash.md. Computes calibration
verdict, writes `audit/<date>-day-1-kickoff.md`, updates CLAUDE.md
current-state, commits.

### Final chat turn — handoff

> "Day 1 done. OOF→LB gap = `$GAP` (`$VERDICT`). Day 2 queue
> in `audit/...-day-1-kickoff.md`; CLAUDE.md current-state
> updated. Next session, say 'continue' or pick from the queue.
> Friction logged: 0 events."

End the session. Do NOT auto-start Day 2.

## Style rules

- Every chat turn ≤4 sentences. Wait for PI reply.
- Bash batches are silent unless they fail. On failure: 1-sentence
  summary + ask PI.
- Never loop `kaggle competitions submit`.
- Never auto-confirm Q6. PI must explicitly say yes.
- Model routing: pre-flight + batches A/C/D2/E = Haiku
  (deterministic). Q&A + Q4 extraction + Day-2 queue = Sonnet.
- Log every avoidable friction (broken CLI, ambiguous Q4 reply,
  failed sed substitution) to `audit/friction.md` per
  `self-improvement.md`.
