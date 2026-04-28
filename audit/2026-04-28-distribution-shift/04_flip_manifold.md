# 04 — Flip manifold: where do the 10,304 flips live?

## Per-score flip rates

| score | n_rows  | n_flips | flip_rate% | row_share% | rule says |
|-------|---------|---------|-----------|-----------|-----------|
| 0     | 33,767  | 0       | 0.0000    | 5.36      | Low       |
| 1     | 115,457 | 5       | 0.0043    | 18.33     | Low       |
| 2     | 122,220 | 365     | 0.2986    | 19.40     | Low       |
| **3** | 102,157 | **4,899** | **4.7956** | 16.22  | Low       |
| 4     | 117,837 | 1,520   | 1.2899    | 18.70     | Medium    |
| 5     | 79,203  | 274     | 0.3459    | 12.57     | Medium    |
| **6** | 38,416  | **1,549** | **4.0322** | 6.10   | Medium    |
| **7** | 15,026  | **1,360** | **9.0510** | 2.39   | High      |
| **8** | 2,680   | **330**   | **12.3134** | 0.43  | High      |
| 9     | 3,237   | 2       | 0.0618    | 0.51      | High      |

**Flip mass concentrates at scores {3, 6, 7, 8}** — these four scores
hold 79% of all flips while occupying only 25% of train rows.

Rule confidence by score:

- score 0: 100.0000% — perfectly clean (33k rows, 0 flips)
- score 1: 99.9957% — essentially perfect (5 flips in 115k rows)
- score 9: 99.9382% — essentially perfect (2 flips in 3.2k rows)
- score 8: 87.6866% — weakest of the rule-correct cells

Per CLAUDE.md's leakage rule, scores {0, 1, 9} are the cells where
hard-override-to-rule is safe AND the LB-best primary already absorbs
them perfectly (per the 2026-04-27 purity-rules diagnostic).

## Per-score flip direction

Within flipped rows, what class is the actual label?

| score | rule says | Low | Med  | High | n     |
|-------|-----------|-----|------|------|-------|
| 1     | Low       | 0   | 5    | 0    | 5     |
| 2     | Low       | 0   | 365  | 0    | 365   |
| **3** | Low       | 0   | **4899** | 0   | **4899** |
| 4     | Medium    | 1507| 0    | 13   | 1520  |
| 5     | Medium    | 78  | 0    | 196  | 274   |
| **6** | Medium    | 0   | 0    | **1549** | **1549** |
| **7** | High      | 0   | **1360** | 0   | **1360** |
| **8** | High      | 0   | **330** | 0    | **330** |
| 9     | High      | 0   | 2    | 0    | 2     |

Flip direction is **deterministic per score**:

- score 3: 100% → Medium (rule says Low)
- score 6: 100% → High (rule says Medium)
- scores 7, 8: 100% → Medium (rule says High)
- score 4: 99.1% → Low (rule says Medium)

This is a strong structural finding: at each score the flip is
**always one step** (Low↔Medium or Medium↔High, never Low↔High). The
NN's perturbation crosses ONE class boundary at a time.

## Per-cell flip rates (top 10)

The 6-bit cell id `(dry, norain, hot, windy, nomulch, kc_active)`
gives 128 cells; only 64 are observed in synth. Cells sorted by
flip rate:

| cell | score | n_rows | n_flips | flip_rate% |
|------|-------|--------|---------|-----------|
| 51   | 7     | 308    | 217     | 70.45     |
| 31   | 7     | 714    | 319     | 44.68     |
| 57   | 7     | 443    | 195     | 44.02     |
| 20   | 3     | 146    | 53      | 36.30     |
| 53   | 7     | 654    | 236     | 36.09     |
| 18   | 3     | 154    | 42      | 27.27     |
| 48   | 4     | 119    | 32      | 26.89     |
| 26   | 4     | 204    | 54      | 26.47     |
| 58   | 6     | 290    | 75      | 25.86     |
| 49   | 6     | 349    | 89      | 25.50     |

**Cell 51 has a 70% flip rate** — by far the densest flip cluster.
It's a small cell (308 rows = 0.05% of train) but every other row is
a flip. score=7 cells dominate the top of the list (4 of 10).

Bottom-5 large cells (≥1% volume = ≥6,300 rows):

| cell | score | n_rows | n_flips | flip_rate% |
|------|-------|--------|---------|-----------|
| 35   | 5     | 14,047 | 16      | 0.114     |
| 2    | 1     | 38,993 | 3       | 0.008     |
| 8    | 1     | 30,389 | 2       | 0.007     |
| 4    | 1     | 46,075 | 0       | 0.000     |
| 0    | 0     | 33,767 | 0       | 0.000     |

**5 cells totaling 163k rows (26% of train) are essentially
deterministic** (≤16 flips total in 163k rows = 0.01%). This matches
the CLAUDE.md 2026-04-27 purity-subcells diagnostic finding (179k
deterministic rows = 28.55%).

## Non-rule features at flip vs clean rows (global)

| col                       | mean_clean | mean_flip | Δ    | Cohen's d |
|---------------------------|------------|-----------|------|-----------|
| Previous_Irrigation_mm    | 62.31      | 63.02     | +0.71| +0.021    |
| Soil_pH                   | 6.482      | 6.498     | +0.02| +0.017    |
| Organic_Carbon            | 0.923      | 0.927     | +0.00| +0.011    |
| Field_Area_hectare        | 7.52       | 7.56      | +0.04| +0.009    |
| Electrical_Conductivity   | 1.745      | 1.750     | +0.01| +0.006    |
| Humidity                  | 61.56      | 61.51     | −0.05| −0.003    |
| Sunlight_Hours            | 7.513      | 7.515     | +0.00| +0.001    |

**Globally, non-rule features show no flip vs clean signal.** This
is an important null. The 2026-04-21 EDA showed per-score effects
(Humidity d=+0.076 at score=3, Prev_Irrig d=+0.107 at score=3), but
those wash out across scores because their direction reverses at
different boundary cells.

The implication: **the flip signal is intra-cell, not marginal.**
Any model that doesn't condition on the rule cell first cannot use
non-rule features to predict flips. This explains why per-cell LR
(2026-04-21) plateaued at 0.96280 OOF and why the score=6 deep-dive
found feature-indistinguishability between teacher-residual missed-H
rows and the cell mean.

## Source

`scripts/dist_shift/flip_manifold.py` →
`scripts/artifacts/dist_shift/flip_manifold_results.json` +
`per_cell_flip_rates.csv`.
