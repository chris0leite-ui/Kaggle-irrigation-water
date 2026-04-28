# 01 — Marginal feature shift

KS, Wasserstein, Cohen's d on each numeric; chi-square + per-level Δ
on each categorical. Sample sizes: orig n=10,000 vs synth-train
n=630,000. With these sizes, even tiny shifts are highly significant
in p-value; **read magnitudes (Cohen's d, KS statistic), not p-values.**

## Numerics — sorted by |Cohen's d|

| col                       |    KS  |    Wass | Cohen's d | mean orig | mean train |
|---------------------------|-------:|--------:|----------:|----------:|-----------:|
| **Rainfall_mm**           | 0.158  | 209.71  | **+0.315**| 1252.50   | 1462.21    |
| Humidity                  | 0.043  |   1.51  |   +0.074  | 60.08     | 61.56      |
| Previous_Irrigation_mm    | 0.041  |   2.50  |   +0.071  | 59.86     | 62.32      |
| Organic_Carbon            | 0.039  |   0.023 |   −0.059  | 0.945     | 0.923      |
| Electrical_Conductivity   | 0.037  |   0.049 |   −0.049  | 1.792     | 1.745      |
| Wind_Speed_kmh            | 0.020  |   0.21  |   +0.037  | 10.16     | 10.38      |
| Soil_Moisture             | 0.012  |   0.34  |   +0.020  | 36.97     | 37.30      |
| Field_Area_hectare        | 0.016  |   0.09  |   −0.019  | 7.60      | 7.52       |
| Soil_pH                   | 0.034  |   0.06  |   −0.006  | 6.49      | 6.48       |
| Sunlight_Hours            | 0.011  |   0.02  |   −0.003  | 7.52      | 7.51       |
| Temperature_C             | 0.008  |   0.07  |   +0.001  | 26.99     | 27.00      |

**Rainfall_mm is by far the largest shift** — d=+0.315 is 4× the next
largest (Humidity 0.074), and the KS statistic 0.158 is 4× the next
largest. Wasserstein 209.7 mm shift in the empirical CDF.

The other "moderate" shifts (Humidity, Prev_Irrig, OC, EC) are all in
the 0.04 ≤ |d| ≤ 0.08 band — small but real. Soil_Moisture,
Temperature_C, Wind_Speed_kmh, Soil_pH, Sunlight_Hours, Field_Area
are essentially untouched (|d| ≤ 0.04).

## Categoricals — sorted by chi-square p-value

| col                |   p-value | n levels | max |Δpp| (orig − train) |
|--------------------|-----------|----------|---------------------------|
| Soil_Type          | 3.93e-03  | 4        | 1.43                      |
| Irrigation_Type    | 2.89e-02  | 4        | 1.03                      |
| Crop_Growth_Stage  | 8.19e-02  | 4        | 0.92                      |
| Region             | 1.24e-01  | 5        | 0.88                      |
| Season             | 1.66e-01  | 3        | 0.81                      |
| Water_Source       | 2.52e-01  | 4        | 0.74                      |
| Crop_Type          | 2.80e-01  | 6        | 0.66                      |
| Mulching_Used      | 8.50e-01  | 2        | 0.10                      |

Categoricals are essentially tied — max |Δpp| = 1.43 on Soil_Type. At
n=10,000 vs 630,000 this is detectable but the magnitude is small.
Mulching_Used is so tied (0.10 pp) that it could plausibly be drawn
from the same distribution.

## Reading

The synth's NN generator preserved the marginal distribution of the 8
categoricals to within ~1 pp. It also preserved 6 of the 11 numerics
to within d=0.04. But it **substantially bumped Rainfall_mm up**
(+210 mm mean shift) and made smaller adjustments to Humidity,
Previous_Irrigation_mm, Organic_Carbon and Electrical_Conductivity.

This is diagnostic of a **conditional generator**: the NN was likely
trained to produce features that match the original's per-class
**label** distribution while letting the joint feature distribution
drift. Rainfall_mm — the rule's most-influential continuous threshold
axis (it shows up twice in the score formula via `2 * norain`) — is
where the drift is largest. We confirm this is per-class in
`03_class_conditional.md`.

## Source

`scripts/dist_shift/marginal.py` →
`scripts/artifacts/dist_shift/marginal_results.json`
