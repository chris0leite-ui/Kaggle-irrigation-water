# 3 · Soil — the buffer between waterings

Soil is not just dirt; it's a water reservoir that decouples rainfall
from root uptake. A field with lots of storage can coast through a dry
week; a field with little storage dries out in a day. This makes soil
the feature most responsible for *frequency* of irrigation, while
weather + crop set the *rate*.

## Texture and the three key thresholds

Soil is classified by the mix of particle sizes:

- **Sand** — large particles, big pores, water drains through fast.
- **Clay** — tiny particles, tiny pores, water clings and drains slowly.
- **Silt** — in between.
- **Loam** — roughly balanced mix of sand/silt/clay. Usually the most
  agriculturally productive.

Three water-content thresholds define the storage:

| Threshold | What it means | Sand | Loam | Clay |
|---|---|---|---|---|
| Saturation | All pores full; roots suffocate if prolonged | ~40% | ~50% | ~55% |
| **Field capacity (FC)** | After gravity drainage stops; optimal | 15–25% | 35–45% | 45–55% |
| **Permanent wilting point (PWP)** | Water held too tightly for roots to extract; plant wilts | 5–10% | 10–15% | 15–20% |
| **Available water (AWC = FC − PWP)** | Useful reservoir | ~10% | ~20% | ~25% |

Numbers are volumetric water content (mm water per mm soil, ×100%).
AWC is the storage a crop can actually draw on between waterings.

### Why this matters for "irrigation need"

- **Sand** needs *small, frequent* irrigations — often classified as
  *High* need because the window between waterings is short.
- **Clay** needs *large, infrequent* irrigations — can read as *Low* or
  *Medium* on any given day because storage rides it through.
- **Loam** is the middle case.

So `Soil_Type` is doing two different jobs: it sets storage capacity
*and* it sets the typical irrigation cadence. Expect strong interaction
between `Soil_Type` and `Soil_Moisture`.

## Management Allowable Depletion (MAD)

Farmers don't wait for PWP. They irrigate once the soil has dried to a
fraction of AWC — the **MAD**, typically 30–60%. Drought-sensitive
crops (vegetables) use lower MAD (~20%); drought-tolerant crops (cotton,
sorghum) tolerate higher MAD (~60–70%). MAD also trends lower on sandy
soils. Below MAD, the plant starts closing stomata and transpiration
drops — this is the onset of water stress.

## Reading `Soil_Moisture`

A moisture value of 30% means very different things on different soils:

- On **sand** (FC ≈ 20%): 30% is oversaturated — drainage is happening.
- On **loam** (FC ≈ 40%): 30% is well within the comfort zone.
- On **clay** (FC ≈ 50%): 30% is already approaching stress.

So a more informative engineered feature is *relative* moisture:

```
  depletion = (FC_soil − Soil_Moisture) / AWC_soil
```

Without knowing FC/AWC per soil type, a model still discovers the same
pattern through `Soil_Moisture × Soil_Type` interactions — trees do
this for free, linear models need explicit encoding.

## Soil chemistry features

### `Soil_pH`

pH affects *nutrient availability*, not water-holding capacity
directly. Most crops prefer pH 6.0–7.5. Outside that range, iron,
phosphorus, and micronutrients become unavailable, growth slows,
transpiration drops, and the crop's effective water demand falls. So
extreme pH can paradoxically reduce irrigation need — not because the
crop is happy, but because it's struggling.

### `Electrical_Conductivity` (EC) — salinity

EC measures dissolved salts. High EC = saline soil. Saline soil lowers
the osmotic potential of soil water, so roots must "pull harder" — to a
plant, saline wet soil behaves like drier soil than it is. Mitigation
is **leaching**: apply *extra* irrigation water to flush salts below
the root zone. Therefore higher EC often correlates with *higher*
irrigation requirement, not through thirst but through leaching
overhead. Typical breakpoints:

- EC < 2 dS/m: non-saline, no leaching penalty
- 2–4: slightly saline, sensitive crops affected
- 4–8: moderately saline, most crops affected
- \> 8: strongly saline, only salt-tolerant crops

### `Organic_Carbon`

Organic matter (proxied by OC) *increases* AWC — it acts like a sponge,
so a high-OC loam can store more water than a low-OC loam of the same
texture. Higher OC also means better structure, better infiltration,
and lower runoff. Expect higher OC to correlate with *lower*
irrigation need all else equal, via bigger storage buffer.

## Sources

- Cornell NRCCA — Soil hydrology (FC, PWP, AWC)
  <https://nrcca.cals.cornell.edu/soil/CA2/CA0212.1-3.php>
- Oklahoma State Extension — Soil water content & irrigation thresholds
  <https://extension.okstate.edu/fact-sheets/understanding-soil-water-content-and-thresholds-for-irrigation-management>
- NRCS — Available Water Capacity (Soil Quality Indicator)
  <https://www.nrcs.usda.gov/sites/default/files/2022-10/nrcs142p2_051590.pdf>
- NRCS — Soil Electrical Conductivity
  <https://www.nrcs.usda.gov/sites/default/files/2022-10/Soil%20Electrical%20Conductivity.pdf>
