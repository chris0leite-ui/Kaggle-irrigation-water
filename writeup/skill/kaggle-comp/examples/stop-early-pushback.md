# Example — stop-early pushback

The irrigation-water agent declared a structural ceiling at every
plateau. Every one was wrong. This is a walked example of the
human's pushback pattern.

## The pattern

Plateau 1 — LB 0.97097 (Day 2):
> Agent: "We've reached the LGBM tuning ceiling. Recommend locking
> baseline + spec_678 stack and stopping."
> Human: "What about target encoding? Multi-seed bagging?"
> Result: recipe_full_te → LB 0.97939, +0.00842 lift.

Plateau 2 — LB 0.97939 (Day 7):
> Agent: "Recipe has saturated. 5 OOF Δ candidates all regressed
> on LB. Suggest locking final selection."
> Human: "What about stacking the bank we built in Days 4-6?"
> Result: 4-stack → LB 0.98094, +0.00155 lift.

Plateau 3 — LB 0.98094 (Day 10):
> Agent: "All meta-stacker candidates with OOF Δ ≥ +0.0005 fail the
> 4-gate filter or regress on LB. The own-pipeline ceiling is
> ~0.0008 above 4-stack."
> Human: "What about override mechanisms — selective per-row flips
> with consensus gating?"
> Result: Idea 4b triple-consensus → LB 0.98150, +0.00056 lift.

Each plateau-break came from a mechanism the agent had previously
rejected on principled grounds. **48 saturation events at 0.98150**
were finally insufficient evidence to rule out further mechanisms,
because we ran out of comp time, not out of ideas.

## The pushback template

When the agent says "structural ceiling / lock and stop":

```
Human: "Hold on. We have <N> days left and <K> slots/day.
Saturation evidence proves we tested known levers, not that no
lever exists. Let's do the Research-loop now:
  1. Web search top public notebooks for this comp.
  2. Pull 2 prior-comp writeups in same domain.
  3. List 5 mechanisms NOT in our audit/.

Don't tell me 'we should lock'. Tell me what 5 untried mechanisms
look like. Use the ML Researcher persona."
```

Note what the human is NOT doing:

- Not arguing about whether the ceiling is real.
- Not asking the agent to "try harder".
- Not threatening or persuading.

The human is **redirecting the agent into the Research-loop** with
a concrete deliverable (5 untried mechanisms). The agent's
saturation claim is treated as a category-error: it confused
"tested levers exhausted" with "lever space exhausted".

## Why this works

LLMs are trained to be helpful and to avoid falsifying claims they
make. When the agent says "ceiling structural", it has committed
to that conclusion and will defend it under direct pushback. But
the conclusion is *contingent on the search space* — and you can
expand the search space without confronting the conclusion.

The Research-loop is the formal expansion mechanism. Persona
rotations (Researcher, Junior, "10 Wild Options") all do the same
thing: provide a structured way to enumerate mechanisms outside
the agent's current frame.

## The portable rule

```
if agent_says_any_of([
    "structural ceiling",
    "own-pipeline wall",
    "lock and stop",
    "no further lift available",
    "exhausted the mechanism space",
]):
    if days_remaining > 3:
        run_research_loop()
        rotate_persona("ML Researcher")
        require_5_untried_mechanisms_with_citations()
    else:
        # final-3-day window — lock+stop is allowed
        accept_lock_and_stop_recommendation()
```

## What if the Research-loop returns nothing useful?

Possible outcomes after a Research-loop:

1. **5 untried mechanisms with citations** → queue top 3, continue.
2. **2-3 untried mechanisms** → queue them, but flag "search-space
   running low".
3. **0-1 untried mechanisms (even after persona rotation)** →
   genuine evidence the mechanism family is exhausted. *Even then*,
   try a heuristic-first cut at the public-notebook approaches the
   pack is using.

Outcome 3 is rare. In the irrigation-water comp, every plateau-
break Research-loop returned 3+ untried mechanisms. The agent's
introspection reliably underestimates the lever space.
