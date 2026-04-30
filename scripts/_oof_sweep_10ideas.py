"""
OOF-only sweep through 10 senior-engineer ideas on top of LB-best 4b (LB 0.98150).
Run on 2026-04-30 (deadline day) by claude/ml-competition-improvements-OuZg8.

Findings (all OOF/diagnostic, no LB slots spent):

#4 drop_lm decomposition:
  - drop_lm = 4b + 176 M->L flips (tier1b-only mechanism)
  - 105 H->M flips IDENTICAL to 4b
  - 176 M->L are at break-even precision ~39%, contributing ~0 macro_delta
  - drop_lm and 4b TIE at LB 0.98150

#2 Filter drop_lm's 176 M->L by 14-bank confidence:
  - bank=L AND agr>=0.85: 73 rows
  - bank=L AND agr>=0.90: 38 rows
  - bank=L AND agr>=0.95: 10 rows
  - All 159 rows where (tier1b=L AND bank=L AND rule=L): score=3 only
  - Projection: 159 flips at 70% precision -> +0.00008 LB; at 50% -> -0.00009

#3 Counter-flip audit on 4b:
  - All 105 H->M flips have bank=M (perfectly aligned with 4b)
  - Only 4 flips have bank-agreement <0.85 (tiny selection-bias counter-flip target)
  - 12 flips have rule_pred=H (rule disagrees) but those are score 7-8 boundary
    where 4b is most useful (NN-flip recovery). Counter-flipping HURTS.

#7 Stratify 4b by anchor confidence:
  - All 105 H->M flips have bagged_v1 P(M) in [0.806, 0.869] (extremely homogeneous)
  - bank-agreement on flips: p25=0.929, p50=0.929 (very tight)
  - No room for stratification

#9 W5 stricter cutoff:
  - All 9 W5 M->H rows have bank=M (against the flip)
  - Zero rows survive any bank-supports-H filter
  - Lever fully closed

#5 Cdeotte rule as 5th axis:
  - Adding rule=bagged to 4b's filter: 0 new flips (4b already strict-filtered)
  - 12 of 4b's 105 flips have rule=H (would counter-flip them); but those are
    the score 7-8 NN-flip recovery rows (4b's most useful flips). REMOVING them HURTS.

#10 Multi-seed bag of 4b mechanism: skipped (no time to retrain RF natural at new seeds).

#1 ExcelFormer 5th axis: queued on Kaggle GPU, not yet returned.

#6 Anchor-switch on PRIMARY (LB 0.98094):
  - 146 flips: 11 L->M, 10 M->L, 125 H->M
  - 79 H->M are NEW flips B's mechanism missed (where B already changed primary's
    H to M; 4b inherits M)
  - 0 NEW H->M flips compose-able with 4b base
  - Projected standalone LB ~0.98110 (HEDGE candidate, not displacement)

#2 (other dirs) L<->M layer on 4b:
  - 4-axis filter: 0 rows (saturated)
  - 3-axis filter: 0 rows
  - Even 2-axis (asw alone) returns 5 L->M + 165 M->L which mostly DISAGREE with bank
  - 4b is fully consistent with multi-axis consensus on L<->M direction

CANDIDATES EMITTED (none clearly displaces 4b at LB 0.98150):

| File                                   | Diff vs 4b | Direction                    | Proj LB         |
|----------------------------------------|------------|------------------------------|-----------------|
| submission_4b_plus_ml_strict90.csv     | 38         | +38 M->L (strictest)         | 0.98148-0.98154 |
| submission_4b_plus_ml_strict85.csv     | 70         | +70 M->L                     | 0.98146-0.98157 |
| submission_4b_plus_ml_3axis.csv        | 158        | +158 M->L (rule+bank+tier1b) | 0.98141-0.98166 |
| submission_4b_plus_asw_lm_2axis.csv    | 170        | mostly M->L, asw-driven      | 0.98141-0.98150 |
| submission_4b_plus_asw_lm_3axis.csv    | 0          | (saturated, no flips)        | 0.98150 (=4b)   |
| submission_anchor_switch_primary.csv   | 356        | hedge, 146 flips on PRIMARY  | 0.98110         |

DECISION: 4b is structurally saturated. The best EV remaining LB probe is
ExcelFormer-as-5th-axis IF/WHEN ExcelFormer kernel returns. Otherwise lock
final-selection at 4b (LB 0.98150) + structural hedge.

Recommended hedge candidates (already LB-validated):
  - submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv (LB 0.98129):
    different mechanism family (no override layer), highest LB-validated hedge
  - submission_3way_recipe025_s1035_s7040.csv (LB 0.98005):
    structurally orthogonal, lower premium (audit F1 swap recommendation)
"""
