# 2026-04-30 — Session closure: override mechanism family is structurally saturated

After CLAUDE.md's 18 NN nulls + 40+ saturation confirmations + the user-pushed
"5 new ideas" exploration this session, the LB-best primary
(`submission_idea4b_selective_override.csv`, LB 0.98150) is at the
operating-point optimum on its OTHERS pool. Three new tests this session
confirm:

## Tests run

### #1 — Test-side pseudo-truth diagnostic
- Levels L1-L4 on agreement subsets are degenerate (4 LB-validated subs
  unanimously agree on 99.7% of test → trivially 100% accuracy on that subset).
- **Useful signal on the 108 disagreement rows where 4b ≠ B**:
  - 14-bank majority agrees with 4b on 108/108 (by construction in 4b's filter)
  - **v1 RF natural agrees with 4b on 98/108 = 90.7%** (within fold-noise of
    91.94% break-even, consistent with measured LB +0.00010)
  - rawashishsin agrees with 4b on 47/108 (rawashishsin is in B's k=2
    construction so it agrees with B on most diffs by construction)
  - v1 + 14-bank BOTH agree with 4b on 98/108 (90.7%) and BOTH with B on
    0/108 — strong independent signal that 4b is right

### #2 — QUAD-consensus 4b' (drop 10 v1-disagreeing flips)
- Built `submission_idea4b_quad_consensus.csv` (98 flips: 4b's 108 minus 10
  where v1 disagrees)
- **Invalidated by confound check**: per `_confound_aware_analysis.py`,
  v1 RF natural ↔ bagged_v1 boundary correlation = **92.6% (same family)**.
  v1 is not an independent 4th axis — it's a single-seed proxy for bagged_v1.
- The 10 v1-disagreeing rows are exactly where bagging did its job (variance
  reduction); reverting them just brings us closer to v1 (LB 0.98129).
- **Projected LB at 95% precision: 0.98155 (+0.00005)**, but at 90% precision
  the candidate REGRESSES vs 4b.
- Not LB-probe-worthy: the gain is fully attributable to the bagging effect
  4b already incorporates.

### #3 — Hunt ADDITIONAL flips outside 4b's filter
- Tested 6 filter loosening directions:
  - 33 rows: B != bagged, bank == bagged, bagged != raw=tier1b=H (recipe-
    unanimous against). Confound analysis projected sub-break-even precision
    (85-92% < 91.94% break-even) — **NOT probe-worthy**.
  - 0 rows: recipe-unanimous AGREE with bagged but bank disagrees → 4b
    already covers all such cases.
  - 2 rows: noise-level p_margin (0.0001-0.0005) → not actionable.

### #4 — Strict bank-unanimity sub-filter
- Restrict 4b to flips with 14-bank agreement = 1.0 (perfect unanimity)
- Keeps 41 of 108 flips. **Projected LB ≈ 0.98148** (drops 2bp vs 4b).
- The 67 dropped flips contribute net-positive even at lower per-row precision
  because there are many of them. 4b's permissive design is at the
  net-macro-recall optimum.

## Conclusions

**The override mechanism family is structurally saturated on this OTHERS
pool.** Every candidate filter perturbation projects LB regression:
- Stricter (drop low-confidence flips): -2bp
- Looser (add raw-disagreeing flips): below break-even
- Cross-axis (replace consensus): correlated, no new signal
- Quadruple-consensus (v1 as 4th axis): same-family confound

Math validation: 4b's 105 H→M flips have **~93.3% measured precision** on
test (back-computed from LB +0.00010 at 105 flips), above 91.94% break-even.
Any modification moves precision OR count in the wrong direction.

## Recommended final-selection lock

**PRIMARY**: `submission_idea4b_selective_override.csv` → LB **0.98150**
(above pack 0.98148 by +0.00002)

**HEDGE**: `submission_rawashishsin_2600_standalone.csv` → LB **0.98109**
- Per confound analysis: ~80% boundary correlation with 4b (vs v1 RF
  natural's 92%); 479 boundary-row diffs (vs v1's 228); recipe-family axis
  only, NOT correlated with 4b's natural-cal mechanism
- Premium -0.00041 LB on public, but materially better private-LB variance
  protection
- ALTERNATIVE: v1 RF natural (LB 0.98129, premium -0.00021) — closer to
  PRIMARY but only 92% boundary-correlated, less variance protection

## Reserve remaining LB slots

4 LB slots remaining. Highest-EV uses:
1. **Variance check**: re-submit PRIMARY at end-of-day to confirm Kaggle
   hasn't drifted (1 slot)
2. **Reserve 3 slots unspent**: with override family closed and 40+
   saturation confirmations, marginal-EV of speculative probes is below
   variance-noise

Do NOT submit:
- `submission_idea4b_quad_consensus.csv` (built this session, projects
  regression per confound check — diagnostic only)
- Any further override mechanism variants (family closed)

## What a senior MLNG engineer concludes

The competition is at a structural plateau. The 0.00069 gap to leader
(Cdeotte 0.98219) is reachable only via mechanisms NOT in this OTHERS
pool — i.e., a model-class or feature-source we don't have. With deadline
today, the call is:
1. Lock NOW (don't burn slots speculating)
2. Pick HEDGE for variance protection, not for public-LB optimality
3. Trust the saturation evidence
