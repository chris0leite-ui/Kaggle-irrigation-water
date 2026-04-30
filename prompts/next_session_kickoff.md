# Next session kickoff — paste this to Claude

(User: copy the block below into a fresh Claude Code session. It tells
Claude exactly what to do, on which branch, and how to proceed
carefully.)

---

```
Pick up where the previous session left off. The key facts:

REPO: Kaggle-irrigation-water
BRANCH: claude/research-ml-solutions-ulRVx (already pushed, contains
  T1-T6 mechanism research + compact CLAUDE.md)

CURRENT LB STATE
- LB-best PRIMARY: submission_idea4b_selective_override.csv → LB 0.98150
- Pack: 0.98148 (we are above by +0.00002)
- Leader (Cdeotte): 0.98219 (+0.00069 above)
- Most recent saturation: T6 directional compose at LB 0.98121
  (40th confirmation; documented in audit/2026-04-30-T6-directional-
  compose-result.md)

YOUR TASK FOR THIS SESSION: execute T1 (LLM-judge mechanism).

1. Read `CLAUDE.md` (now compact, ~6k tokens) for operational rules.
   In particular respect the ⚠️ rules: never lock/hedge, never give
   up, ALWAYS ASK before LB probe, never wrap submit in retry loops,
   keep files short and modular, smoke-test before long runs.

2. Read `prompts/subagent_llm_judge.md` — it contains the exact
   subagent prompt template and decision rule for T1.

3. Build the T1 orchestrator (~150 lines, modular per CLAUDE.md
   rule). Suggested files:
   - scripts/T1_select_borderline.py  — pick ~100-500 borderline
     test rows where 4b and other LB subs disagree, save IDs.
   - scripts/T1_format_batch.py        — format rows into the
     subagent prompt template.
   - scripts/T1_call_subagents.py      — spawn haiku subagents in
     batches of 50-100 rows, parse responses, save labels.
   - scripts/T1_compose_override.py    — apply the 4-axis override
     decision rule (LLM agrees + bank-maj agrees + H->M direction
     + LLM CONF >= 0.7).
   - scripts/T1_validate_train_oof.py  — TRAIN OOF precision check
     before LB probe.

4. Spawn haiku subagents using the Agent tool with
   subagent_type="claude-code-guide" (smallest harness overhead) and
   model="haiku". The previous session's haiku spawns failed with
   "Prompt is too long" because the harness injected a 272k-token
   CLAUDE.md as parent context. CLAUDE.md is now 6k tokens, and a
   fresh session has no parent-conversation bloat, so haiku should
   work now.

5. Carefully measure first call cost. Start with a 1-row batch.
   Verify response parses correctly. Then scale.

6. Apply the standard 4-gate filter from CLAUDE.md before LB probe:
   - G1: Δ standalone OOF ≥ +0.0001
   - G2: per-class recall ≥ 4b - 5e-4 each class
   - G3: H->M precision ≥ 92% (break-even under macro-recall)
   - G4: net_H direction-positive on test

7. Ask user before LB probe (CLAUDE.md ⚠️ rule).

REFERENCE FILES (read in order):
- CLAUDE.md                                                (compact, current state)
- prompts/subagent_llm_judge.md                            (T1 prompt template)
- audit/2026-04-30-T6-directional-compose-result.md       (40th saturation)
- audit/CLAUDE-md-archive-2026-04-30.md                   (full historical log,
                                                            optional read)
- LEARNINGS.md                                             (portable patterns)

KNOWN MECHANISM CLOSURES (don't repeat):
- All 18 NN family attempts (magnitude trap structural)
- Meta-stacker bank extension (10+ saturations)
- Wide programmatic FE (recipe redundancy)
- Override mechanisms beyond 4b's 3-axis filter (10+ attempts)
- Symbolic regression / DAE / mixup / TTA / per-row gating

T1 LLM-judge is structurally novel: external supervision signal
encoded in LLM training (agronomic priors), orthogonal to all 14
in-bank components.

LB BUDGET: 1/10 used today (T6 probe), 9 remaining.

Confirm you've read this and the next files before starting work.
```

---

## End of paste block
