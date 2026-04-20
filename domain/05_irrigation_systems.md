# 5 · Delivery system, mulching, water source, field size

The water-balance equation in `01_water_balance.md` assumes every mm of
irrigation reaches the root zone. In reality, efficiency varies by a
factor of 2× depending on how water is delivered. Efficiency shows up
as the *gross* irrigation need: to deliver 100 mm to the roots, a flood
system needs to apply ~165 mm at the head while a drip system needs
~110 mm.

## Delivery methods — `Irrigation_Type`

| System | Typical efficiency | Water needed per effective mm | Notes |
|---|---|---|---|
| **Drip** (trickle) | 90–95% | 1.05–1.1× | Emitters at the root zone; minimal evaporation/runoff. Expensive to install. |
| **Sprinkler** | 75–85% | 1.2–1.3× | Spray in air; wind and evaporation losses. |
| **Furrow / surface** | 55–70% | 1.5–1.8× | Water flows in channels between rows; deep percolation + runoff. |
| **Flood / basin** | 40–60% | 1.7–2.5× | Whole field inundated (rice paddies); large losses but cheap. |

**Implication for the target**: A field listed with `Irrigation_Type =
Flood` should, all else equal, require a larger *applied* water volume
than a field with `Irrigation_Type = Drip` — so flood rows may bias
toward `High` need for the same crop/weather, purely because of
delivery losses. Conversely, drip + mulch on the same crop should bias
toward `Low` or `Medium`.

Note the ambiguity: if the target encodes *applied* water (what the
farmer pumps) the above holds. If it encodes *crop demand* (what the
plant would want in ideal conditions) then `Irrigation_Type` should
matter less. Worth testing empirically.

## `Mulching_Used`

Mulch is a layer (plastic film, straw, crop residue) covering the soil
between plants. Its main effect is **cutting soil-surface evaporation**,
which otherwise can be 30–50% of total ET in row crops with incomplete
canopy cover.

Quantified savings in the literature:

- Pepper: 14–29% less irrigation water
- Onion: up to 70% less
- Film-mulched drip vs. unmulched drip: +~30% water-use efficiency
- Cumulative soil evaporation: 21–33% reduction under mulch

So `Mulching_Used = Yes` should systematically shift rows *down* a need
bin (High → Medium, Medium → Low), most strongly early in the season
when canopy is sparse and soil evaporation is a large share of ET.

## `Water_Source`

Water source rarely changes the physical water requirement but encodes
*reliability* and *salinity risk*:

- **Rainwater / harvested** — cheap but unreliable; farmers may
  under-irrigate when the tank is low.
- **Surface water (canal, river, pond)** — moderate reliability,
  usually low salinity.
- **Groundwater (well, tubewell)** — reliable but can be saline or
  sodic in arid regions → interacts with `Electrical_Conductivity`.
- **Reservoir / tank** — seasonal.

Expect `Water_Source` to act as a weak direct driver but to carry
regional / climatic signal by proxy (rainwater is more common in humid
areas).

## `Field_Area_hectare`

Field area is mostly a scale variable — bigger fields need more *total*
water but not more water *per hectare*. The target is almost certainly
a per-hectare or intensity-based classification, so area should have
weak direct effect. It can still matter as a proxy for:

- Farm type (smallholder vs. commercial) — smallholders use different
  systems.
- Uniformity — very large fields have spatial variability that may bias
  recommended irrigation upward.

Don't expect area to be a top feature.

## `Previous_Irrigation_mm`

How much water was applied in the previous interval. This is effectively
a *lagged target* — it captures recent soil water state that isn't
reflected in a single `Soil_Moisture` reading. Expect:

- High recent irrigation + wet soil → current need is *Low* (just
  topped up).
- Low recent irrigation + dry soil + hot weather → *High* (catching up).

Can be one of the strongest single features, and may cause target
leakage concerns if it was generated from the same process as the
target. Worth sanity-checking its distribution across target classes.

## Sources

- DripWorks — Drip vs. traditional watering comparative study
  <https://www.dripworks.com/blogdrip-irrigation-vs-traditional-watering-methods-a-comparative-study/>
- TWDB — Comparison of Drip vs. Flood vs. Center Pivot
  <https://www.twdb.texas.gov/publications/reports/contracted_reports/doc/1813582260-El-Paso-Water-Utilities.pdf>
- Frontiers in Agronomy — Mulching in dryland agriculture
  <https://www.frontiersin.org/journals/agronomy/articles/10.3389/fagro.2024.1361697/full>
- Irrigation Science (Springer) — Mulching effects on soil evaporation,
  ET, and Kc
  <https://link.springer.com/article/10.1007/s00271-024-00924-8>
