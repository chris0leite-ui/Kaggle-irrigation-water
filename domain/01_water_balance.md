# 1 · The water balance

Everything downstream is an elaboration of this one equation.

## The mental model

A crop field is a bucket. Water goes **in** (rain, irrigation) and comes
**out** (evaporation from soil, transpiration through the plant, deep
drainage, runoff). The crop is healthy as long as the soil stays wet
enough for roots to keep pulling water.

```
  ΔSoil_moisture  =  Rain + Irrigation − ET_crop − Drainage − Runoff
```

"Irrigation need" is the amount of water a farmer must *add* so that the
soil moisture on the right of the equation stays above the crop's
comfort zone. When rain is plentiful and ET is low, irrigation need is
near zero. When the atmosphere is thirsty and rain is scarce, the
farmer must pour water in at nearly the rate ET is pulling it out.

## Why the target is *Low / Medium / High* (ordinal)

The underlying quantity is continuous (mm of water, or liters/hectare).
The competition discretizes it into three bins. The bins have a natural
order — *High* means *Low* plus more — but the metric (balanced
accuracy) is order-agnostic, so a High → Low error and a High → Medium
error are equally bad. Despite this, ordinal-aware losses (e.g.
CORAL, cumulative-link) often still help models because the class
manifold *is* ordinal; the penalty structure is just blind to it.

## The three levers (and how the features map to them)

| Lever | What it controls | Key features in this dataset |
|---|---|---|
| **Atmospheric demand (ET)** | How fast water leaves | `Temperature_C`, `Humidity`, `Sunlight_Hours`, `Wind_Speed_kmh` |
| **Water supply** | Free water coming in | `Rainfall_mm`, `Previous_Irrigation_mm`, `Water_Source` |
| **Storage & losses** | How much the soil buffers, how efficient delivery is | `Soil_Type`, `Soil_Moisture`, `Irrigation_Type`, `Mulching_Used`, `Field_Area_hectare` |

Crop-specific features (`Crop_Type`, `Crop_Growth_Stage`, `Season`)
scale the ET term via the crop coefficient Kc (covered in
`04_crops.md`).

## Reference irrigation intensities (rule of thumb)

Reference ET classifies climates as:

- **4–5 mm/day** — low atmospheric demand (cool, humid)
- **6–7 mm/day** — medium
- **8–9 mm/day** — high (hot, dry, windy, sunny)

A crop with Kc = 1 in a high-ET climate needs ~9 mm/day of water; over a
100-day season that is ~900 mm. Compare to annual rainfall: if rainfall
covers most of the need, irrigation is *Low*; if it covers almost none,
*High*. This is roughly what the target is encoding.

## Sources

- FAO-56, Ch. 5 — Introduction to crop evapotranspiration
  <https://www.fao.org/4/X0490E/x0490e0a.htm>
- FAO Training Manual 24, Ch. 3 — Irrigation scheduling
  <https://www.fao.org/4/t7202e/t7202e06.htm>
