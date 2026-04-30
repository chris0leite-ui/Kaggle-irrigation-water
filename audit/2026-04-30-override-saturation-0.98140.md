# 2026-04-30 — 2-OTHER k=2 unanimous: NEW LB BEST 0.98140 + TC1 saturation

(Commits `0951e9f` 2026-04-30 04:12 UTC and `d90dd63` 2026-04-30 05:13 UTC.
Override family compounds once on 0.98134 anchor, then saturates.)

## Submission B (current LB-best)
`submissions/submission_2other_raw_tier1b_k2.csv`
- LB public: **0.98140**
- Δ vs prior LB-best 0.98134: **+0.00006** (lift)
- Δ vs original v1 RF natural 0.98129: **+0.00011** cumulative
- OOF 0.98088 → LB 0.98140 → gap **−0.00052**
- Pack 0.98148 now **+0.00008 above** (1bp from pack-busting).
- Leader 0.98219 +0.00079.

## Mechanism
Per-row hard argmax override on the LB-best 4-stack
(`tier1b_greedy_meta`). For each test row, if **2 of 2** STRUCTURALLY-
DISTINCT OTHER LB-validated submissions unanimously agree on a class
different from the 4-stack's argmax, flip to that class.

OTHERS pool (deliberately the 2 most-distinct):
- `submission_rawashishsin_2600_standalone.csv` — different model family
  (single XGB + sklearn TargetEncoder)
- `submission_tier1b_greedy_meta.csv` — different L2 architecture
  (XGB-meta on 63-component bank, recipe-bias regime)

145 test overrides (test side):
```
H→M:  88   (LB-best demoted from H to M by raw+tier1b unanimous)
M→L:  32
M→H:  14
L→M:  11
```

Only **25 test rows** differ from prior 0.98134 (k=4 unanimous), but
those 25 rows alone delivered the +0.00006 LB lift.

## Counter-intuitive carryover finding
Stricter consensus on the 2 most-distinct OTHERS gives BETTER LB
transfer per OOF gain than unanimous-of-4 with mixed correlated subs:
```
mechanism                                OOF Δ vs 0.98129 baseline   LB Δ      carryover
4-OTHER k=4 unanimous (mixed correlated) +0.00015                    +0.00005  0.33×
2-OTHER k=2 most-distinct only           +0.00021                    +0.00011  0.52×
```

Mechanism: rawashishsin and tier1b 4-stack share **no** stacking layer
or feature pipeline. Their unanimous disagreement with anchor on a
target row is structural evidence; correlated-subs unanimity is
inflated by shared error modes.

## TC1 follow-up (saturation confirmation)
Submitted `submissions/tier1_1_TC1_*` at 2026-04-30 05:13 UTC.
- Anchor: B (LB 0.98140)
- OTHERS for second-level override: {v1_rf 0.98129, k4 0.98134}
  (the only LB-validated subs higher than base 4-stack)
- 16 test overrides applied
- LB result: **0.98136** (Δ −0.00004 vs B)

Translation: 12-15 of the 16 overrides were wrong (precision ~5%,
below break-even 9.3%). v1_rf and k4 over-promoted M→H on rows where
B's "no override" was correct.

**TC2-TC6 on larger pools produced 0 overrides** — every other
LB-validated submission's argmax already matches B's. Override
mechanism is **saturated on this OTHERS pool**.

## Two portable rules (LEARNINGS.md candidates)
1. **Override mechanism transfer rate increases with consensus
   strictness.** Among LB-validated OTHERS, the 2 most structurally-
   distinct submissions (different model family + different L2
   architecture) deliver 0.52× carryover; mixed-correlated 4-OTHER
   unanimity caps at 0.33×. **For future override experiments, prefer
   smaller-but-more-distinct OTHERS pools over larger-but-correlated
   pools.**
2. **Once a CSV-level-override candidate beats all OTHER LB-validated
   submissions in its pool, further overrides with the SAME pool
   saturate.** Breaking further requires either (a) a structurally
   NEW LB-validated submission as a new OTHER, or (b) variance-
   reduction at the BASE level (sibling RF natural with different fold
   seed) before re-running B's mechanism.

## LB budget tally
2026-04-30: 2/10 used (B probe + TC1), 8 remaining at deadline-day open.
