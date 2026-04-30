"""
Confound-aware analysis on top of 4b (LB 0.98150).
Run on 2026-04-30 deadline day. Branch: claude/ml-competition-improvements-OuZg8.

The deep-dive confound finding (bagged_v1 ↔ 14-bank ↔ RFnat agree 99.4%+) was
INCOMPLETE — that's GLOBAL agreement dominated by easy rows. On boundary rows
(score 3,6,7,8 where any axis disagrees), the picture is very different:

=== BOUNDARY-CONDITIONAL INDEPENDENCE (3,294 boundary-active rows) ===

  Pair                        % agreement   Independence?
  bagged_v1 ↔ RFnat              92.6%      Same family
  bagged_v1 ↔ 14-bank            54.1%      Largely INDEPENDENT
  bagged_v1 ↔ rule               20.2%      VERY independent
  bagged_v1 ↔ raw                81.3%      Mildly correlated
  bagged_v1 ↔ tier1b             80.0%      Mildly correlated
  14-bank ↔ rule                 59.0%      Some independence
  raw ↔ tier1b                   68.6%      Recipe-family correlation
  rule ↔ raw                     15.5%      VERY independent

Truly independent triplet on boundary: {bagged_v1 / 14-bank, rule, raw / tier1b}

=== APPLYING THE CONFOUND INSIGHT ===

The 32 H counter-flip candidates (4b=H, bagged=M, bank=M, rule=M, raw=H, tier1b=H)
have THREE truly-independent axes saying M, ONE recipe-family axis saying H.

Bayesian posterior at score=6 (rule prior 96% accurate):
  Optimistic (3 indep axes ~ 2.5 effective): P(true=M | obs) ≈ 0.997
  Realistic (3 indep axes ~ 2.0 effective): P(true=M | obs) ≈ 0.995
  Pessimistic (collapsed to 1 effective): P(true=M | obs) ≈ 0.85

Empirical anchor (4b's 105 LB-validated H→M flips):
  - Have raw==M only 46/105 (44%); tier1b==M only 59/105 (56%)
  - 4b's flips ALLOW recipe-family disagreement
  - LB-confirmed precision ≈ 95-96%

The 32 candidates have STRONGER recipe-family disagreement (raw=H AND tier1b=H,
unanimous) than 4b's 105 (split). Empirical precision likely 85-92%, BELOW 91.9%
break-even. Counter-flipping projects -0.0001 to +0.00004 LB. NOT probe-worthy.

=== THE CONFOUND INSIGHT VALIDATES 4b's DESIGN ===

Built confound-corrected 3-axis filter:
  (B != bagged) AND (bank == bagged) AND (rule == bagged) AND
  ((raw == bagged) OR (tier1b == bagged))
  → 95 flips on B, ALL 95 contained in 4b's 108
  → 13 of 4b's flips DROPPED (12 score 7-8 H→M with rule disagree, 1 L→M)

The dropped 13 are exactly 4b's MOST VALUABLE NN-flip-recovery rows
(per CLAUDE.md: score 7-8 boundary). 4b's permissive design (allowing rule
disagreement at score 7-8) is a feature, not a bug.

=== ACTIONABLE LEVERAGE: HEDGE SELECTION ===

Boundary-conditional correlation with 4b:
  RFnat:           92% boundary correlation (shares natural-cal axis)
  rawashishsin:    ~80% (recipe-family, more orthogonal)
  recipe_full_te:  ~85% (recipe-family without override layer)

Boundary diff vs 4b:
  RFnat:          228 rows (limited variance protection)
  rawashishsin:   479 rows (better variance protection, only -20bp LB)
  recipe_full_te: 638 rows (best variance protection, -211bp LB)

REVISED HEDGE RECOMMENDATION:
  Best: submission_rawashishsin_2600_standalone.csv (LB 0.98109, premium -0.00041)
    - Recipe-family axis only, NOT correlated with 4b's natural-cal mechanism
    - 479 boundary-row diffs provide private-LB variance protection
    - Cost vs 0.98129 hedge: -20bp LB on public, but better orthogonality

  Alternative: submission_sklearn_rf_meta_natural_standalone.csv (LB 0.98129)
    - 92% boundary-correlated with 4b (shares natural-cal axis)
    - Limited variance protection on private LB

CANDIDATE EMITTED (diagnostic, not probe-worthy):
  submission_confound_corrected_3axis.csv (95 flips, strict 3-axis filter)
    - Strictly weaker than 4b (drops 13 most-valuable flips); reference only.
"""
