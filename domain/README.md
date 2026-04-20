# Irrigation domain knowledge

Background notes on the physical system being modeled: what "irrigation
need" actually *means*, and why each feature in the competition data is
(or isn't) a plausible driver. Written for a reader with no agronomy
background.

No data inspection went into these notes — they are pure domain context,
distilled from FAO and extension-service references (cited at the bottom
of each file). Feature names referenced in passing come from `brief.md`.

## Read in order

1. **[01_water_balance.md](01_water_balance.md)** — the one equation that
   underlies the whole problem. Irrigation need = the leftover gap
   between water a crop loses and water it receives for free.
2. **[02_evapotranspiration.md](02_evapotranspiration.md)** — how
   temperature, humidity, sunlight, and wind determine how fast water
   leaves the field.
3. **[03_soil.md](03_soil.md)** — how much water the soil can *hold*
   between irrigations (texture), how usable that water is (pH, EC),
   and what the moisture reading actually measures.
4. **[04_crops.md](04_crops.md)** — per-crop thirst, growth-stage effects
   (the Kc curve), root depth, and drought tolerance.
5. **[05_irrigation_systems.md](05_irrigation_systems.md)** — why drip
   vs. sprinkler vs. flood matters, what mulching does, and what water
   source tells you.
6. **[06_india_context.md](06_india_context.md)** — Kharif/Rabi/Zaid
   seasons and regional climate, which the `Season` and `Region`
   features are almost certainly encoding.
7. **[07_modeling_implications.md](07_modeling_implications.md)** —
   feature-by-feature prior on direction and strength of effect, plus
   interactions worth engineering.

## One-sentence summary

A field needs *more* irrigation when the atmosphere is thirsty
(hot / dry / sunny / windy), the crop is water-hungry and in a sensitive
growth stage, the soil holds little water and is already dry, rainfall
has been scarce, and the delivery system leaks water (flood, no mulch).
Everything else is a second-order correction on that picture.
