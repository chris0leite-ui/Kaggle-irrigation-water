# 2 · Evapotranspiration — how fast water leaves the field

## ET in one paragraph

**Evapotranspiration (ET)** is the sum of two water-loss paths:
**evaporation** (water turning to vapor from the wet soil surface) and
**transpiration** (water pulled by roots, piped up through the plant,
and released as vapor through pores in the leaves called stomata).
Both are driven by the same thing: the atmosphere's capacity to accept
water vapor. Hot, dry, windy, sunny air accepts a lot; cool, humid,
calm, cloudy air accepts little.

ET is the *demand* side of the water balance in `01_water_balance.md`.
Irrigation is effectively paying that demand.

## Reference ET (ET₀) and the Penman–Monteith equation

Agronomists standardized "atmospheric thirst" as **reference ET**
(ET₀) — the ET of a hypothetical, well-watered grass reference crop.
The FAO-56 Penman–Monteith equation computes ET₀ from four weather
drivers:

| Driver | Effect on ET₀ | Feature in this data |
|---|---|---|
| Air temperature (°C) | ↑T → ↑ET₀ (more energy to vaporize water) | `Temperature_C` |
| Relative humidity (%) | ↑RH → ↓ET₀ (air already holds water) | `Humidity` |
| Solar radiation | ↑sun → ↑ET₀ (primary energy source) | `Sunlight_Hours` (proxy) |
| Wind speed (m/s) | ↑wind → ↑ET₀ (strips saturated boundary layer off leaves) | `Wind_Speed_kmh` |

The equation is nonlinear and the four drivers *interact* — e.g. wind
matters a lot when humidity is low, much less when humidity is high.
Tree models will pick this up from raw features; for linear/NN models,
an engineered feature `VPD ≈ (1 − Humidity/100) × saturation_vapor_pressure(T)`
(vapor pressure deficit) captures most of the T×RH interaction.

## Actual crop ET

The crop a farmer is actually growing uses water at a different rate
from the reference grass. Correct with a **crop coefficient Kc**:

```
  ET_crop = Kc × ET_0
```

Kc depends on the crop *and* the growth stage (see `04_crops.md`).
Typical Kc ranges from ~0.3 (bare young seedlings) through ~1.0–1.2
(full canopy mid-season) and back down to ~0.4 at senescence. So the
same weather can drive very different actual water demand depending on
what's in the field and how mature it is.

## Climate-bucket intuition

| Regime | Daily ET₀ | Typical conditions |
|---|---|---|
| Low | 4–5 mm/day | Cool (<15 °C mean), humid, cloudy, calm |
| Medium | 6–7 mm/day | 15–25 °C, moderate RH, moderate wind/sun |
| High | 8–9 mm/day | Hot (>25 °C), dry, sunny, windy |

This is roughly how the `Irrigation_Need` target should stratify against
weather features *if* crop and soil were held constant — which they
aren't, but the signal is still strong.

## Modeling takeaways

- Temperature, humidity, sun, and wind are near-monotonic drivers of
  need. Expect tree splits like `Temperature > 28 AND Humidity < 40`.
- `Sunlight_Hours` is a proxy for solar radiation; it collinearizes with
  temperature on clear days but decouples on cloudy-hot days.
- A feature like `VPD` (vapor pressure deficit) or simply
  `Temperature_C × (100 − Humidity)` is often a stronger single predictor
  than either T or RH alone.

## Sources

- FAO-56, Ch. 2 — FAO Penman–Monteith equation
  <https://www.fao.org/4/x0490e/x0490e06.htm>
- Michigan State Extension — "What is evapotranspiration and why is it
  important in irrigation?"
  <https://www.canr.msu.edu/news/what-is-evapotranspiration-why-is-it-important-in-irrigation>
- ASCE Standardized Reference ET Equation
  <https://www.mesonet.org/images/site/ASCE_Evapotranspiration_Formula.pdf>
