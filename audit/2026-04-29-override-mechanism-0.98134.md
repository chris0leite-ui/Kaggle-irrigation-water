# 2026-04-29 — k=4 unanimous override: NEW LB BEST 0.98134

(Commit `bc47eb7`, 2026-04-29 20:24 UTC. First own-pipeline mechanism in
33+ saturations to break LB 0.98129 since v1 RF natural set it 11 hours
prior.)

## Submission
`submissions/submission_lbbest_overridden_by_unanimous_others.csv`
- LB public: **0.98134**
- Δ vs prior LB-best 0.98129: **+0.00005**
- Pack 0.98148 now +0.00014 above (closer than ever pre-override).

## Mechanism
Per-row hard argmax override on the LB-best 4-stack (`tier1b_greedy_meta`).
For each test row, if **4 of 4** OTHER LB-validated submissions
unanimously agree on a class different from the 4-stack's argmax, flip
the 4-stack's prediction to that consensus class.

OTHERS pool (4 LB-validated submissions):
- `submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv` (LB 0.98129)
- `submission_rawashishsin_2600_standalone.csv` (LB 0.98109)
- `submission_3way_recipe025_s1035_s7040.csv` (LB 0.98005)
- `submission_recipe_greedy_recipe_pseudolabel.csv` (LB 0.97998)

120 test rows changed. Direction breakdown:
```
H→M:  65   (LB-best demoted from H to M by 4-of-4 unanimous)
M→L:  31
M→H:  14
L→M:  10
Net_H: -51 (REMOVE-High in absolute count)
```

OOF precision rates that drove LB lift:
```
H→M direction:  95.6%  (175/183 correct on OOF)  ← dominant signal
M→L:            58.1%
L→M:            35.7%
M→H:            15.2%
Overall:        75.6%  (208/275 correct)
```

## Key insight
When 4-of-4 OTHER LB-validated submissions unanimously say one class
and LB-best says a different class, **LB-best is wrong ~76% of the time
overall, and ~96% on H→M direction specifically**. The unanimous-
consensus signal is more reliable than LB-best on these boundary rows.

This is the **first per-row hard override mechanism** (vs blend or
bank-extension) tested in the saturation log, and the first candidate
to break LB 0.98129 since it was set.

## Two new portable rules (LEARNINGS.md candidates)
1. **REMOVE-High direction is not always LB-regressive.** Net_H = −51
   yet LB lifted by +0.00005. Prior closure rule "REMOVE-High direction
   always LB-regresses" was specific to soft-blend mechanisms where
   per-class probability is redistributed. Hard argmax flips on
   consensus rows isn't subject to per-class probability redistribution.
2. **Consensus signal across LB-validated submissions can cross
   break-even precision under macro-recall** (96% on H→M vs 91.92%
   break-even floor). Unlike learned detectors (missed-H, spec6_v2)
   that capped at 6.5% top-N precision, the consensus mechanism uses
   independent-LB-validation as the precision oracle.

## Reproduction
- `scripts/build_consensus_override_submissions.py` (build helper —
  emits k=4 unanimous + k=3 majority + hardvote5 candidates)
- All 4 OTHER inputs are gitignored-whitelisted submission CSVs.
