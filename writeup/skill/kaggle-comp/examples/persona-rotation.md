# Example — persona rotation

A walked example of how the human PI rotated personas to break a
stuck loop in the irrigation-water comp.

## The setup

Day 14. LB had been at 0.98094 for 3 days. The agent (Senior ML
Engineer persona by default) had run 5 stacking variants, all
failing the 4-gate filter. Last session's audit:

> "All variants of the 4-stack meta with OOF Δ ≥ +0.0005 fail G4
> direction-asymmetry. The macro-recall surrogate ran null.
> Suggest pivoting to NN architectures we haven't tested
> (Trompt, TabM, ExcelFormer)."

Three NN families had already failed (TabPFN, RealMLP, FT-T). The
agent was about to spend a GPU kernel slot on the next NN.

## Rotation 1 — Junior ML Engineer

Human invoked Junior persona via subagent:

```
Agent({
  description: "Fresh angle on plateau",
  subagent_type: "general-purpose",
  prompt: """
You are a junior ML engineer who hasn't read the prior-attempts
log. The metric is balanced accuracy. The data is 19-feature
tabular with 3 classes (Low/Medium/High, priors 58/38/3%). Current
LB-best is a 4-stack meta-learner.

Don't worry about what's been ruled out. What 3 things would you
try first? Skip standard tabular tricks (LGBM tuning, XGBoost,
MLPs); we already did those. Think about what a *non-ML* person
would try.
"""
})
```

Junior returned 3 ideas:

1. "Just look at the 145 rows where the LB-best primary is
   uncertain. What do those rows look like? Maybe we hand-correct
   them."
2. "Build a confidence-ranked list of test rows. The top-N most
   confident predictions are probably right; the bottom-N are
   probably the leverage point."
3. "If we have multiple OOF predictors that disagree, the
   disagreement itself is signal. Train a model on the *disagreement
   pattern*, not on the predictions."

## What Senior had been doing

Senior persona was iterating on stacking variants — trying to find
the right meta-learner over the 14-bank. Idea 1 (hand-correct
ambiguous rows) was outside Senior's frame because it's not a
"learning" mechanism. Senior had categorised it earlier as "manual
work, not a model — skip on principled grounds".

## Following Junior's idea 1

The human asked the Planner to scope Junior's idea 1:

```
Plan: For each test row where primary is most uncertain
(top-Margin distance), check what the 14-bank majority votes for.
If the bank-majority disagrees with primary AND the disagreement
is unanimous across ≥7 banks, consider an override.
```

This became the **override mechanism family** that produced:

- Day 15: k=4 unanimous override → LB 0.98134 (+0.00040)
- Day 16: 2-OTHER raw+tier1b k=2 unanimous (B) → LB 0.98140 (+0.00006)
- Day 17: Idea 4b triple-consensus → **LB 0.98150** (+0.00010)

The +0.00056 total lift over 3 days came from a Junior-persona
idea that Senior had categorised as "not worth doing" 3 days earlier.

## What this teaches

1. **Persona rotation is not just a different prompt.** It's a
   subagent invocation with no parent context, which removes the
   "we already ruled this out" anchor.

2. **The agent's "skip on principled grounds" list is biased.**
   It's biased toward mechanisms the agent has language for. Hand-
   coded overrides, threshold tweaks, and "look at the data" type
   ideas often get categorised as not-real-ML and skipped — even
   when they're the highest-EV moves.

3. **Junior's strength is naive enumeration.** Senior knows what
   has been tried and pattern-matches. Junior doesn't, and so
   surfaces ideas that Senior skipped.

4. **Don't dismiss the persona output.** The first thing the agent
   wanted to do was rebut Junior's ideas ("we already considered
   row-level overrides — they don't generalize"). The human had to
   say: "scope idea 1 anyway, even as a smoke test, just to bound
   the lift available".

## When to rotate which persona

| Stuck on... | Rotate to... |
|---|---|
| Stacking saturated, all variants fail G4 | **Junior** (fresh angle) or **Data Analyst** (raw-data view) |
| All NN families failed | **ML Researcher** (literature search) |
| Specific bug or infra issue | **Problem-Solver** |
| Candidate looks too good | **Senior** (review mode) |
| Agent refuses to enumerate | **"10 Wild Options"** |

## The portable rule

After every 3 consecutive nulls in the same mechanism family:

1. Rotate at least one persona.
2. Require the persona to return 3+ ideas, with the first being
   non-ML (heuristic / threshold / hand-coded / data-look).
3. Run a smoke on the cheapest of the 3 ideas, even if you think
   it won't work.

The cost is one smoke run (≤5 min). The expected value is
detection of mechanisms the default persona is filtering out.
