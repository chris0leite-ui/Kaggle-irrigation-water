"""
Deep-dive sweep on top of 4b (LB 0.98150) — round 2 (after first 10-idea sweep).
Run on 2026-04-30 deadline day. Branch: claude/ml-competition-improvements-OuZg8.

Finally answers: are there ANY remaining mechanism-novel levers above the
4b ceiling? Diagnostic-only, no LB slots spent.

=== HARD-VOTE OF TOP-6 LB-VALIDATED SUBMISSIONS ===

Subs analyzed (LB-ranked):
  4b (0.98150), i5 (0.98148), dlm (0.98148), B (0.98140), tc1 (0.98136), k4 (0.98134)

Pairwise agreement: all >99.88%. Subs are heavily nested through 4b's mechanism.

Equal-weight + LB-weighted hard-vote both produce SAME 9 row diffs from 4b:
  - 7 M->H flips (overlap with W5's 9-row M->H set)
  - 2 M->L flips (which are 4b's L->M flips reverted; drop_lm did this and lost
    -0.00002 LB on those 2 reversions, confirming 4b's L->M flips are CORRECT)

For the 7 W5 rows: 14-bank says M (=4b), opposes the H direction. 4 LB-others
saying H is structurally weaker than 4b's natural-cal mechanism saying M
(natural-cal is what catches the NN-flip signal that recipe-family models miss).
CLAUDE.md projected ~39% precision; counter-flipping HURTS.

=== COUNTER-FLIP CANDIDATES ON 4b ===

32 rows where 4b=H, bagged_v1=M, 14-bank=M (natural-cal says M):
  - All 32: raw=H, tier1b=H (recipe-family says H — that's why 4b's filter didn't fire)
  - All 32: rule=M (rule supports natural-cal)
  - All 32: score=5 or 6 (rule's M domain)
  - bagged_v1 raw P(M) = 0.91-0.94 (very confident M from natural-cal)
  - 14 of 32 also have RFnat=M (strictest natural-cal subset)

CONFOUND DISCOVERED: bagged_v1, 14-bank, RFnat are all >99.4% correlated.
They're effectively ONE axis (natural-cal family), not 3.

So evidence on 32 rows is 1-vs-1:
  - Natural-cal axis (bagged + bank + RFnat + rule) says M
  - Recipe-family axis (raw + tier1b) says H

Counter-flip H->M needs >91.9% precision (macro-recall break-even).
Estimated precision on this disagreement set: 80-90%. Below break-even.
Expected LB: -0.00009 to +0.00004. Risk-asymmetric, NOT worth probing.

Candidates emitted (DIAGNOSTIC, NOT LB-PROBE-WORTHY):
  - submission_4b_minus_HM_counterflip.csv         (14 flips)
  - submission_4b_minus_HM_counterflip_supreme.csv (14 flips, RFnat=M+rule=M)
  - submission_4b_minus_14_natcal_HM_counter.csv   (14 flips, full natural-cal)
  - submission_4b_minus_natcal_strict.csv          (8 flips, score=6, P(M)>0.93)

=== 40 ADD-H CANDIDATES (bagged_v1 + RFnat agree on H) ===

40 rows where 4b=M but bagged_argmax=H AND RFnat=H.
DIAGNOSED AS BIAS-ARGMAX ARTIFACTS:
  - bagged_v1 raw P(H) is only 0.06-0.15 (low!)
  - bagged_v1 P(M) is ~0.89 (raw distribution favors M)
  - V1_BIAS adds +3.20 to H column, flipping argmax to H despite low raw H confidence
  - 14-bank says M with full agreement on these rows
  - rule says M on 36 of 40 (only 4 at score 7-8 where rule says H)

These are NOT genuine H consensus. They're calibration-bias artifacts.
Adding H predictions would HURT (~-0.00007 LB at expected precision).

=== CONCLUSION ===

4b is structurally saturated at every consensus-mechanism family I can construct.
The cross-LB-sub diagnostic CONFIRMS 4b's correctness (where it disagrees with
others, drop_lm's L->M reversal LB-regressed, validating 4b).

The natural-cal vs recipe-family disagreement on 32 H rows is at break-even
precision; expected EV near 0 with high variance.

LB-PROBE RECOMMENDATION: do NOT spend slots on these candidates.
Lock final-selection at:
  - PRIMARY: submission_idea4b_selective_override.csv (LB 0.98150)
  - HEDGE:   submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv (LB 0.98129)
"""
