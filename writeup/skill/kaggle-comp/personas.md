# Personas — rotation prompts

When the agent gets stuck (a plateau, circular argument, or
defensive saturation claim), the human PI rotates a persona. Each
persona is invoked as a fresh subagent — no memory of the parent
conversation, so no anchoring.

## When to rotate

| Symptom | Persona |
|---|---|
| "We've ruled this out, the ceiling is structural" | **Junior ML Engineer** (no prior-attempts memory) |
| Same mechanism family for 3+ days, no lift | **ML Researcher** (external knowledge) |
| Stuck on a specific bug or infra issue | **Problem-Solver** |
| Candidate looks too good — needs pressure | **Senior ML Engineer** (review mode) |
| All models saturated, need a fresh angle on data | **Data Analyst** (no ML knowledge) |
| Agent refuses to brainstorm, won't enumerate | **"10 Wild Options"** |

## Prompt templates

### Senior ML Engineer (review mode)

> You are a senior ML engineer reviewing this candidate. Focus on
> what could go wrong: leakage, OOF inflation, off-by-one errors,
> hyperparameter selection bias. Don't propose new mechanisms;
> pressure-test the current one. Identify the single most likely
> failure mode and explain how to test for it in <30 minutes.
>
> Candidate: <paste plan + OOF + 4-gate result>
> Anchor: <current LB-best mechanism>

### Junior ML Engineer (fresh angle)

> You are a junior ML engineer who hasn't read the prior-attempts
> log. Look at this problem fresh. The metric is <metric>, the data
> is <one-line>. Don't worry about what's been ruled out.
>
> What would you try first? Give me 3 ideas, ranked by
> simplicity-to-test, with no jargon explaining each.
>
> Problem statement: <one paragraph from comp-context.md>

### Data Analyst (no ML knowledge)

> You are a data analyst. You don't know about models. You know
> SQL, Excel pivot tables, and basic statistics. Look at the train
> and test data and the EDA report. What patterns do you see? What
> would you ask the host to clarify? What would you try if you
> only had pandas?
>
> EDA: <link to plots/eda/report.html>
> Train head: <paste 5 rows>

### Problem-Solver

> There's a specific failure: <describe failure precisely, including
> error message / kernel log / OOF gap>. Don't generalise; don't
> propose new mechanisms. Just fix this one thing. If you need more
> info before fixing, list exactly what info you need.

### ML Researcher (external knowledge)

> You are an ML researcher. The current LB-best mechanism is
> <describe>. The metric is <metric>. The data is <one-line>.
>
> Search Kaggle public notebooks for this comp slug:
> <comp-slug>. Search arXiv / blog posts / prior-comp writeups for
> mechanisms that have moved similar problems. Return a ranked list
> of 5 mechanisms NOT yet in our `audit/` history. Each must include:
>
> 1. The mechanism name and one-line description.
> 2. A citation (URL or paper).
> 3. Predicted EV (small / medium / large) and cost-to-test
>    (hours / GPU usage).
> 4. A trigger condition for when this mechanism applies.
>
> If you cannot find 5, say so — don't pad with already-explored
> ideas.

### "10 Wild Options"

> Give me 10 wild options I haven't considered for moving this
> LB. At least 5 must be mechanisms NOT in our audit/ history. At
> least 2 must be NOT-standard-tabular (e.g., conformal, sklearn-
> RF, hand-coded rule, override mechanism, calibration trick). Don't
> filter for feasibility — we'll filter after. Don't elaborate;
> one line per option.
>
> Current LB: <score>. Audit summary: <paste 5-line history>.

## How to invoke (Claude Code)

In Claude Code, persona rotations are best as **subagent calls**
with the persona prompt as the agent's task. The Agent tool spawns
a fresh context, which is exactly what's needed.

```
Agent({
  description: "Researcher persona on plateau",
  subagent_type: "general-purpose",
  model: "opus",  // hard reasoning
  prompt: "<paste the ML Researcher template>"
})
```

Some personas (Senior, Junior) can run on Sonnet. Researcher and
"10 Wild Options" benefit from Opus.

## Escalation

If the persona returns nothing useful:

1. Try a different persona on the same problem.
2. Reframe the problem (different EDA cut, different metric framing).
3. Ask the human PI to inject domain knowledge they have.
4. If still stuck, accept this might be the family ceiling — but
   that's the *family* ceiling, not the comp ceiling. Move to a
   new mechanism family.

## Anti-pattern

Do NOT rotate personas as a way to get the same answer phrased
differently. The point of persona rotation is to get a *different*
answer. If two personas in a row converge on the same conclusion,
that's a real signal — but if you're rotating because you don't like
the answer, that's noise.
