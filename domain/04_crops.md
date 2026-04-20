# 4 · Crop type and growth stage

The same weather and soil can yield wildly different irrigation need
because different plants — and different life stages of the same plant —
pull water at different rates.

## Per-crop thirst (rough ordering)

Ranked approximately by seasonal water requirement, from thirstiest to
most drought-tolerant:

| Tier | Crops | Seasonal water need (mm) | Notes |
|---|---|---|---|
| Very high | **Rice (paddy)**, **Sugarcane** | 1200–2500 | Rice is flooded; sugarcane is a year-round heavy drinker |
| High | **Banana**, **Maize** (grain corn), vegetables (tomato, cabbage) | 600–1200 | |
| Medium | **Wheat**, **Cotton**, **Groundnut** | 400–700 | |
| Low | **Sorghum**, **Millet**, **Barley**, pulses | 300–500 | Drought-tolerant; deep roots |

Expect `Crop_Type` to be one of the strongest splits in the model.
Rice/sugarcane rows should bias toward `High`; millet/pulses toward `Low`.

## Growth stage — the Kc curve

A plant's water demand follows an arc over its life cycle. FAO-56
divides it into four stages, each with a characteristic crop
coefficient Kc (multiplier on reference ET₀):

| Stage | What's happening | Typical Kc |
|---|---|---|
| **Initial** (emergence, "sowing") | Bare soil, tiny seedlings, mostly evaporation from soil | 0.3–0.4 |
| **Crop development** | Canopy expanding, roots deepening | 0.4 → 1.0 |
| **Mid-season** (reproductive: flowering, grain/fruit fill) | Full canopy, peak transpiration, *most stress-sensitive* | 1.0–1.2 |
| **Late season** (ripening, senescence, harvest) | Leaves drying down, demand falling | 1.2 → 0.4 |

The competition's `Crop_Growth_Stage` feature (observed value examples
include `Sowing`) is almost certainly a discrete version of this curve.

### Two-way interaction to expect

`Crop_Type × Crop_Growth_Stage` is a strong predictor because:

- A sugarcane field at *mid-season* in hot weather is the worst case —
  big canopy × high Kc × peak ET → reliably `High`.
- The *same* sugarcane field at *sowing* has Kc ≈ 0.3 and may look
  identical to a wheat field in terms of demand.

Trees capture this via depth; for linear/NN models, explicit
interaction features help.

## Root depth — why some crops tolerate drought

Deeper roots can reach water stored further down, so deep-rooted crops
can go longer between irrigations and tolerate higher MAD:

| Crop | Effective root depth |
|---|---|
| Onion, lettuce, shallow vegetables | 0.3–0.5 m |
| Wheat, rice, maize | 0.8–1.2 m |
| Cotton, sugarcane, sorghum | 1.2–2.0 m |
| Alfalfa, tree crops | 1.5–3+ m |

Root depth × soil AWC sets the *total* reservoir the crop can draw on.
A cotton field on clay has access to ~2 m × 25% = ~500 mm of stored
water; an onion on sand has ~0.3 m × 10% = ~30 mm. That's a factor of
15× difference in how long either can coast without irrigation.

## Drought sensitivity (critical stages)

Stage sensitivity modulates how tightly MAD is enforced:

- **Most sensitive**: flowering, pollination, grain fill, fruit set.
  Water stress here directly costs yield. MAD typically 20–30%.
- **Least sensitive**: late ripening, maturation. Farmers often
  *withhold* water here to improve quality (cotton, grapes).

So `Crop_Growth_Stage = flowering/filling` should skew strongly toward
`High` need, while `ripening/maturation` skews *away* even if the
weather is hot.

## Sources

- FAO-56, Ch. 6 — Single crop coefficient (Kc)
  <https://www.fao.org/4/x0490e/x0490e0b.htm>
- FAO Training Manual 24, Ch. 2 — Crop water needs
  <https://www.fao.org/4/s2022e/s2022e02.htm>
- NC State Extension — Soil, water, and crop characteristics for
  irrigation scheduling
  <https://content.ces.ncsu.edu/soil-water-and-crop-characteristics-important-to-irrigation-scheduling>
- Zwart & Bastiaanssen (2004) — Crop water productivity for wheat, rice,
  cotton, maize
  <https://ui.adsabs.harvard.edu/abs/2004AgWM...69..115Z/abstract>
