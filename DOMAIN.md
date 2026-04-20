# Irrigation need — domain primer

Reference material for `playground-series-s6e4`. Kept next to `brief.md`
(host material) and `REPORT.md` (work log). The goal here is to make
explicit the physics the label is trying to summarize, so feature
engineering is guided rather than blind.

## What the label means

`Irrigation_Need ∈ {Low, Medium, High}` is a 3-class summary of how
much supplemental water a field should receive from the farmer on top
of rainfall to keep the crop healthy. It's a discretized, real,
physical quantity — the gap between what the crop and soil need and
what nature has already supplied.

The governing equation in agronomy is the **soil-water balance** over
a period Δt:

```
Irrigation_needed  ≈  ETc  −  Effective_rainfall  −  ΔSoil_moisture  +  Runoff + Drainage
```

Every feature in the dataset is a slice of one of those four terms.

## The four physical drivers

### 1) ETc — crop water demand

How much water the crop actually consumes. Factored as:

```
ETc = Kc(crop, growth_stage) × ET0(weather)
```

- **ET0 (reference evapotranspiration)** — the demand a well-watered
  grass reference crop would feel *given the weather alone*. The
  Penman–Monteith equation (FAO-56 standard) says ET0 rises with
  **temperature, solar radiation, wind speed**, and with **vapor
  pressure deficit** (so it *falls* with humidity).
  Features: `Temperature_C`, `Sunlight_Hours`, `Wind_Speed_kmh`,
  `Humidity`.
- **Kc (crop coefficient)** — a multiplier that varies by crop and by
  where in its lifecycle it is. Typical shape: low at sowing (~0.3),
  peak at mid-season / flowering (0.9–1.2 for cereals, up to 1.25 for
  sugarcane), falls at harvest. This is why `Crop_Growth_Stage`
  dominated chi² (97k in our EDA) — it is *literally* a multiplier on
  water demand.
  Features: `Crop_Type`, `Crop_Growth_Stage`.

### 2) Effective rainfall — water supplied from the sky

Not all rain is useful: heavy bursts run off, light rain evaporates
before reaching roots. Typical rule of thumb: effective rainfall ≈
0.7–0.9 × rainfall for moderate storms.

Feature: `Rainfall_mm`. Note its range (0–2500 mm) is far too large
for daily or weekly rainfall — this is almost certainly **seasonal or
annual** accumulation. Important clue for feature engineering: do not
treat it like a short-window flux.

### 3) Current soil water status

How close the field already is to the crop's stress threshold.

- **`Soil_Moisture`** — volumetric water content (%). This is why it
  dominated F-stat (41k in EDA): it is a near-direct proxy for the
  label.
- **`Soil_Type`** sets the **plant-available water-holding capacity**:
  - Sandy: ~6–10% — drains fast, low storage.
  - Loamy: ~15–18% — the goldilocks for most crops.
  - Clay: ~18–25% — high storage but drains slowly, can waterlog.
  - Silt: ~17–22%.
- **`Mulching_Used`** cuts surface evaporation by roughly 20–50%, so
  mulched fields need less irrigation for the same weather. Matches
  its chi² rank (28k, second among categoricals).

### 4) History / management context

- **`Previous_Irrigation_mm`** — recent supply; a field that was just
  irrigated has higher soil water, so the *next* need is lower. The
  sign matters: after controlling for soil moisture, this should be
  anti-correlated with the label.
- **`Irrigation_Type`** — efficiency varies wildly:
  - Drip: 85–95% efficient.
  - Sprinkler: 65–80%.
  - Surface / Canal: 40–60%.
  - Rainfed: no irrigation applied — these fields depend entirely on
    rain, so the label distribution is different in kind.
- **`Water_Source`** — proxy for availability / reliability; matters
  more for farmer decisions than for what the crop actually needs.
- **`Field_Area_hectare`** — mostly a scale variable; per-hectare need
  shouldn't depend on size.
- **`Soil_pH`, `Electrical_Conductivity`, `Organic_Carbon`** — soil
  *chemistry*, not hydrology. They matter for yield but only weakly
  for water need. Consistent with their low F-stat (< 150 each).

## Indian-agriculture context (confirmed by the data)

The `Season` levels `Kharif / Rabi / Zaid` are the three Indian
cropping seasons, and `Region ∈ {North, South, East, West, Central}`
matches Indian administrative geography:

| Season | Months  | Rainfall        | Typical crops                 |
|---     |---      |---              |---                            |
| Kharif | Jun–Oct | Monsoon, heavy  | Rice, Cotton, Sugarcane       |
| Rabi   | Nov–Apr | Dry, cool       | Wheat, Potato                 |
| Zaid   | Apr–Jun | Very dry, hot   | Short-duration veg, Maize     |

So `Season = Zaid`, `Region = West`, high `Temperature_C`, low
`Rainfall_mm` is a signature of high irrigation need; `Season =
Kharif` for `Crop_Type = Rice` is usually rainfed.

## Feature-engineering implications

The domain equation suggests features that should push past the
0.98114 tied pack:

1. **ET0 proxy** = `Temperature_C × (1 − Humidity/100) × Wind_Speed_kmh`
   — a crude Penman surrogate; high when hot, dry, and windy.
2. **Kc-weighted demand** = interaction `Crop_Type × Crop_Growth_Stage`
   (one-hot the product, or target-encoded means).
3. **Net water balance** = `Rainfall_mm + Previous_Irrigation_mm − ET0_proxy`.
4. **Soil-moisture deficit** = `soil_capacity_for_type − Soil_Moisture`
   — how much room is left in the soil bank.
5. **Regime flags** — `Irrigation_Type == "Rainfed"` (known-different
   regime) and `Mulching_Used` as an ET multiplier.
6. **Season × Region** interaction — handles the climatic-regime split
   without making the model discover it.

These target the residual Medium↔High confusion, which is exactly
where the LGBM baseline is losing points.

## Caveats for this (synthetic) dataset

- The data was generated by a deep-learning model from a real
  Irrigation Prediction dataset, so *distributions* are plausible but
  individual rows may not be physically consistent (e.g. a row could
  have low soil moisture, heavy rainfall, and "Low" need
  simultaneously — a model artifact, not agronomy).
- `Rainfall_mm` 0–2500 mm looks like annual total, not rainfall since
  last irrigation.
- `Previous_Irrigation_mm` capped at ~120 mm — probably a per-event
  figure, not cumulative.

Verify each assumed sign / unit before committing to a transformation
(e.g. does higher `Rainfall_mm` actually predict lower
`Irrigation_Need` after controlling for other features?).
