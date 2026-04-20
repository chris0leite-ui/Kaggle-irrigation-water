# 6 · Indian agricultural context

The `Season` feature taking values like `Zaid`, along with crops like
Sugarcane and area in hectares, strongly suggests the underlying dataset
is framed around **Indian** agriculture. Reading `Season` and `Region`
through that lens makes them far more informative.

## The three crop seasons

Indian agriculture runs on three overlapping cropping seasons driven by
the monsoon:

| Season | Months | Rainfall regime | Typical crops | Irrigation dependence |
|---|---|---|---|---|
| **Kharif** | Jun – Oct | Southwest monsoon — wet | Rice, maize, cotton, sugarcane, groundnut, soybean, millets | *Lower* — rain does much of the work, but failure of monsoon swings many fields to *High* |
| **Rabi** | Oct – Apr | Post-monsoon, cool, dry | Wheat, barley, mustard, gram (chickpea), peas | *High* — almost entirely irrigation-fed |
| **Zaid** | Mar – Jun | Hot summer, minimal rain | Watermelon, muskmelon, cucumber, gourds, fodder | *Very high* — hot + dry + no rain |

So as a prior, without looking at any other feature:

- `Season = Kharif` → bias toward **Low** or **Medium** (rain helps).
- `Season = Rabi` → bias toward **Medium** (no rain, but cool ET).
- `Season = Zaid` → bias toward **High** (no rain, hot ET, thirsty veg
  crops).

Sugarcane is unusual because it spans *all three* seasons (~12–18 month
crop), which means the `Season × Crop_Type = Sugarcane` interaction
carries different meaning than for seasonal crops.

## The `Region` feature (likely North/South/East/West)

Climate and rainfall vary hugely across India:

| Region | Climate | Annual rainfall | Dominant pattern |
|---|---|---|---|
| **North** (Punjab, Haryana, UP) | Continental — hot summers, cool winters | 500–1000 mm | Canal + tubewell irrigation; wheat-rice belt |
| **South** (TN, Karnataka, AP, Kerala) | Tropical, two monsoons | 800–3000 mm (coastal much higher) | Tank + well irrigation; rice, coconut, coffee |
| **East** (WB, Odisha, Bihar, NE) | Humid subtropical | 1200–2500 mm | Rice-dominant; less irrigation-intensive |
| **West** (Rajasthan, Gujarat, MP, Maharashtra) | Semi-arid to arid | 200–1000 mm | Groundwater + drip; cotton, millets, pulses |

Priors this implies:

- `Region = East` + Kharif → bias **Low** (lots of rain, rice thrives).
- `Region = West` + Rabi/Zaid → bias **High** (arid + hot + thirsty).
- `Region = North` + Rabi → **Medium** / **High** (wheat needs canal
  water; no rain in winter).
- `Region = South` + Kharif → **Low** (double monsoon rain).

If the dataset's region encoding is coarser (e.g. just N/S/E/W) these
are first-order approximations. The raw weather features
(`Temperature_C`, `Rainfall_mm`) should already capture most of the
climate signal, so `Region` is mostly a categorical shortcut that trees
can use for clean splits.

## Interaction cheat-sheet

The model should find these high-signal combinations:

- `Crop_Type = Rice` × `Season = Kharif` × `Region = East` → **Low**
  (rain-fed paddy)
- `Crop_Type = Wheat` × `Season = Rabi` × `Region = North` → **Medium**
  (canonical irrigation-fed wheat)
- `Crop_Type = Sugarcane` × any season in `Region = West` → **High**
  (big thirst, dry climate)
- `Season = Zaid` with any vegetable crop → **High** (hot + irrigated)

## Sources

- TractorGuru — Kharif, Rabi & Zaid crops & seasons
  <https://tractorguru.in/news/agriculture-news/kharif-rabi-and-zaid-crops-and-their-seasons>
- BYJU's UPSC — Major cropping seasons in India
  <https://byjus.com/free-ias-prep/major-cropping-seasons-in-india/>
