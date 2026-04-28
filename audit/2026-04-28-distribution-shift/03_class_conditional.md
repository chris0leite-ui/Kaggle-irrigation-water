# 03 — Class-conditional shifts

The marginal shift could come from (a) class-prior rebalancing,
(b) within-class feature shift, or (c) label noise (rule-vs-label
flips). This file separates those.

## (a) Class priors

| class  | orig %    | synth %   | Δpp (synth−orig) |
|--------|-----------|-----------|------------------|
| Low    | 58.64     | 58.72     | +0.08            |
| Medium | 38.00     | 37.95     | −0.05            |
| High   | 3.36      | 3.33      | −0.03            |

**Class priors are statistically identical.** Δpp ≤ 0.08 on every
class. The shift is NOT label-rebalancing.

## (b) Per-class numeric shifts — top 15 by |Cohen's d|

| class  | col                       |   KS   | Cohen's d | mean orig | mean synth |
|--------|---------------------------|--------|-----------|-----------|------------|
| Medium | **Rainfall_mm**           | 0.216  | **+0.417**| 1158.60   | 1444.48    |
| High   | **Rainfall_mm**           | 0.238  | **+0.401**| 677.83    | 989.16     |
| High   | Temperature_C             | 0.129  | +0.265    | 32.88     | 34.57      |
| Low    | **Rainfall_mm**           | 0.116  | **+0.246**| 1346.28   | 1500.53    |
| High   | Soil_Moisture             | 0.080  | −0.219    | 19.71     | 17.67      |
| High   | Wind_Speed_kmh            | 0.089  | +0.219    | 13.68     | 14.64      |
| Low    | Soil_Moisture             | 0.048  | +0.113    | 41.72     | 43.31      |
| Low    | Humidity                  | 0.050  | +0.099    | 59.97     | 61.95      |
| High   | Humidity                  | 0.080  | +0.092    | 59.31     | 61.12      |
| High   | Organic_Carbon            | 0.059  | −0.091    | 0.958     | 0.924      |
| Medium | Previous_Irrigation_mm    | 0.056  | +0.088    | 60.25     | 63.18      |
| Medium | Soil_Moisture             | 0.054  | −0.085    | 31.16     | 29.74      |
| Medium | Wind_Speed_kmh            | 0.041  | +0.069    | 11.42     | 11.79      |
| High   | Electrical_Conductivity   | 0.059  | −0.065    | 1.75      | 1.69       |
| Low    | Previous_Irrigation_mm    | 0.057  | +0.062    | 59.53     | 61.72      |

**Rainfall_mm shifts UP within EVERY class.** The shift is not driven
by class-mix at all. The synth has higher rainfall than orig at the
Low, Medium, and High class levels independently.

The High class shows the most dramatic feature shifts — Rainfall up
+311 mm (d=+0.40), Temperature up +1.7 °C (d=+0.27), Soil_Moisture
down (d=−0.22), Wind up (d=+0.22). The High class in synth is
"hotter, windier, drier in soil, but with more rainfall" than the
High class in orig.

## (c) Rule-vs-label match rates

| dataset       | rule predicts y? | n flips |
|---------------|------------------|---------|
| orig          | 100.0000%        | 0       |
| synth-train   | 98.3644%         | 10,304  |

The rule is perfect on the orig (which is how it was reverse-
engineered in the first place). The synth has **10,304 rule
violations (1.64%)** — the standard "flip rows" mass that prior comp
sessions tracked.

## (d) `dgp_score` distribution shift

| score | orig %  | synth % | Δpp     | rule says |
|-------|---------|---------|---------|-----------|
| 0     | 4.10    | 5.36    | +1.26   | Low       |
| 1     | 12.93   | **18.33** | **+5.40** | Low       |
| 2     | 18.46   | 19.40   | +0.94   | Low       |
| 3     | 23.15   | **16.22** | **−6.94** | Low       |
| 4     | 18.90   | 18.70   | −0.20   | Medium    |
| 5     | 12.44   | 12.57   | +0.13   | Medium    |
| 6     | 6.66    | 6.10    | −0.56   | Medium    |
| 7     | 2.55    | 2.39    | −0.17   | High      |
| 8     | 0.61    | 0.43    | −0.19   | High      |
| 9     | 0.20    | **0.51** | **+0.31** | High      |

**Mass redistribution from score=3 (−6.94 pp) toward score=1
(+5.40 pp)**. Both are in the rule-says-Low band, so the rule's
overall accuracy is unchanged (98.36% on the synth, 100% on the orig).
But the boundary band (score 3, where the flip rate is highest at
4.80%) has been DEPLETED in the synth.

Score=9 has a +0.31 pp synth bump. In absolute counts, that's 1953
additional score=9 rows in the synth vs what the orig prior would
predict — the NN over-samples the most-extreme rule cell (everything
critical, active stage).

## Reading

The synth's NN is doing two things at once:

1. **Drifting Rainfall up** in feature space (within every class).
2. **Shifting score-3 mass to score-1** (cleaning up the dominant
   boundary band).

The combined effect is that the synth is, on average, **more cleanly
classifiable than orig** under the rule (98.4% rule-acc with 79% of
flips concentrated in scores {3, 6, 7, 8} — see `04_flip_manifold.md`).
The NN added rule-conformant rows in the safe zones (score 1) and
removed rows from the boundary (score 3). It then introduced a small
1.64% flip-noise budget concentrated at the residual boundary scores.

## Source

`scripts/dist_shift/class_conditional.py` →
`scripts/artifacts/dist_shift/class_conditional_results.json` +
`per_class_shifts.csv`.
