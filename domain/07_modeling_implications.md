# 7 · Modeling implications

A distilled cheat-sheet for turning the preceding domain notes into
modeling choices. Not a set of commands — a set of priors to test
against the data.

## Per-feature priors

Direction = sign of expected effect on irrigation need
(↑ = feature up → need up). Strength is a rough guess on a
5-point scale.

| Feature | Direction | Strength | Notes |
|---|---|---|---|
| `Temperature_C` | ↑ | ★★★★ | Near-linear ET driver |
| `Humidity` | ↓ | ★★★★ | Interacts strongly with T (use VPD) |
| `Sunlight_Hours` | ↑ | ★★★ | Proxy for solar radiation |
| `Wind_Speed_kmh` | ↑ | ★★ | Important when humidity low |
| `Rainfall_mm` | ↓ | ★★★★★ | Most direct water-supply feature |
| `Previous_Irrigation_mm` | ↓ (short-term) / ↑ (regime) | ★★★★ | May be near-tautological with target — inspect carefully |
| `Soil_Moisture` | ↓ | ★★★★ | *Conditional on* `Soil_Type`; absolute value means different things per texture |
| `Soil_Type` | mixed | ★★★ | Sand → more frequent (↑ need), clay → buffered (↓ need). Strong interactions. |
| `Organic_Carbon` | ↓ | ★★ | More organic matter → more storage |
| `Soil_pH` | U-shape | ★ | Extreme pH → stunted crop → *lower* effective demand |
| `Electrical_Conductivity` | ↑ | ★★ | Leaching overhead; interacts with `Water_Source` |
| `Crop_Type` | varies | ★★★★★ | Top-tier split; rice/sugarcane → High, millet/pulses → Low |
| `Crop_Growth_Stage` | arc | ★★★★ | Peaks mid-season (Kc ≈ 1.0–1.2); low at sowing and ripening |
| `Season` | varies | ★★★★ | Kharif ↓ (rain), Rabi ~ , Zaid ↑ (hot+dry) |
| `Region` | varies | ★★★ | West ↑, East ↓; largely captured by weather but still useful |
| `Irrigation_Type` | Flood ↑, Drip ↓ | ★★ | Delivery efficiency; strength depends on whether target is applied or demand |
| `Water_Source` | weak | ★ | Mostly a regional/reliability proxy |
| `Mulching_Used` | Yes ↓ | ★★ | Cuts soil evaporation — strongest early in season |
| `Field_Area_hectare` | ~0 | ★ | Scale variable; weak direct effect |
| `id` | — | 0 | Drop |

## Engineered features worth trying

- **VPD** (vapor pressure deficit): `(1 − Humidity/100) × eₛ(Temperature_C)`
  where `eₛ(T) = 0.6108 × exp(17.27T / (T + 237.3))` (kPa, FAO-56).
  Often beats T and RH separately for ET-driven responses.
- **Reference ET proxy**: a simpler form like
  `0.0023 × (T_mean + 17.8) × sqrt(T_range) × Sunlight` (Hargreaves-like)
  if daily T-max/min aren't available — may linearize the target.
- **Relative soil depletion**: lookup-encode FC / AWC per `Soil_Type`
  and compute `depletion = (FC − Soil_Moisture) / AWC`. Even a
  hand-coded 3-way lookup helps linear models.
- **Net water balance**: `Rainfall_mm + Previous_Irrigation_mm − k·ET_proxy`
  per row. A compact single feature that summarizes the balance.
- **Crop-stage × Kc**: ordinal encode `Crop_Growth_Stage` ∈
  `{Sowing, Vegetative, Flowering, Fruiting, Harvest}` and cross with
  `Crop_Type`.
- **Salinity leaching factor**: binary flag for `EC > 2 dS/m` combined
  with `Water_Source ∈ {groundwater}`.

## Interactions likely to dominate

Trees find these automatically; worth adding explicit cross-features
for linear/NN baselines:

- `Crop_Type × Crop_Growth_Stage` — Kc curve realization.
- `Soil_Type × Soil_Moisture` — absolute moisture means different
  things per texture.
- `Temperature × Humidity` (or direct VPD).
- `Season × Region` — climatological prior.
- `Irrigation_Type × Mulching_Used` — stacked efficiency.

## Class-specific intuition (balanced accuracy matters)

`High` is the minority class (~3.3%). A "physically extreme" High row
probably looks like:

```
  Region=West, Season=Zaid, Crop_Type=Sugarcane,
  Growth_Stage=Mid/Flowering, Soil_Type=Sand,
  Soil_Moisture=low, Rainfall_mm≈0, Temperature_C high,
  Humidity low, Sunlight_Hours high, Mulching_Used=No
```

If the model cleanly recovers such rows, recall on `High` should be
strong. The hard `High` rows are likely those where only some of these
signals align — and that's where threshold tuning pays.

Conversely, `Low` rows should be dominated by:

```
  Season=Kharif, Region=East, high Rainfall_mm,
  Crop_Type=Rice, Soil_Type=Clay, high Soil_Moisture,
  cool + humid weather
```

`Medium` is then the large confused middle. Most misclassifications
likely sit on the Low/Medium boundary under a macro-recall metric this
hurts much less than losing any single `High`.

## Sanity checks before trusting a model

- Does `Crop_Type = Sugarcane/Rice` skew positive in feature importance?
- Does `Rainfall_mm` have negative SHAP effect?
- Does `Humidity` have negative SHAP effect?
- Does `Mulching_Used = Yes` reduce the predicted class?

If any of these violate the domain prior, it's either a data quirk
(e.g. synthetic generator idiosyncrasy), a label encoding bug, or a
target-leaking feature. Investigate before submitting.
