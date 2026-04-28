# Distribution-shift EDA — original (10k) vs synthetic train (630k)

## TL;DR — six findings, three actionable

1. **Rule features (Soil_M, Rain, Temp, Wind, Mulching, Stage) are
   threshold-preserving but distribution-distorting.** Binary
   `<25 / <300 / >30 / >10` indicators carry over precisely (rule
   accuracy 100% on orig, 98.36% on train), but the *continuous*
   values within each rule-cell drift systematically:
   - **Rainfall_mm**: train mean +210 mm above orig (Cohen's d=−0.34,
     KS=0.16); shift concentrated at boundary scores 4–7 (s=6 d=−0.62).
   - **Soil_Moisture | High**: orig is +0.27 SD wetter than synth Highs.
   - **Humidity** and **Previous_Irrigation_mm** shifted up
     (d≈−0.075 each).

2. **Categorical marginals are nearly identical** across all 8 cat
   columns (JS divergence < 2e-4 every column). The NN preserves
   priors faithfully — but only at the marginal level.

3. **Joint distribution is materially different.** AV-AUC train↔orig
   = **0.6935** (test↔orig = 0.6895 — confirmed test inherits the same
   shift). Restricted to within-rule-cell, AV-AUC peaks at **0.74 for
   score=6 ∩ class=Medium** — the NN distorts the joint manifold
   *most* exactly at the M↔H boundary where the flip mass lives.

4. **All 64 rule-cube cells are shared and 100%-pure in orig.** In
   train, **62 of 64 cells are no longer pure** — the NN introduced
   flips in nearly every cell. The flip rate is identical (1.63% vs
   1.67%) on orig-seen vs orig-unseen 8-cat-tuples → the **flip
   mechanism is rule-cube-conditional only**; categorical 8-tuple is
   irrelevant.

5. **Flips are strictly one-step ordinal** and direction follows score
   monotonically: rule=Low rows always flip → Medium; rule=High rows
   always flip → Medium; rule=Medium rows split based on which
   boundary they're near (s=4 → Low 99%, s=6 → High 100%). No row
   ever flips two classes.

6. **Flip signal lives in continuous values of rule features, not in
   non-rule features.** Within-cell flip-detector AUC: rule features
   only = 0.85, **non-rule features only = 0.53** (essentially chance).
   At s=7 (rule=High, 9% flip rate), flipped rows have Cohen's d on
   their 4 rule axes vs non-flipped peers of: SM +0.74, Temp −0.81,
   Rain −1.48, Wind −0.95. The flipped rows sit at the **edge of the
   rule cell on EVERY axis simultaneously** — they pass every
   threshold but barely.

## Implications for prediction (ranked by expected lift)

### A. The NN is interpolating an original-data manifold that exists at the cell *boundary*. (highest-information finding)
The flipped synthetic rows look like rows the NN would generate if it
"lived" between two cells in the original. Direct cohen's d test on
flipped synth Highs at s=7 vs orig Highs at s=7: Rainfall d=−0.91,
Soil_M d=+0.21. Flipped Highs at s=7 have **much less rain** than the
255 orig Highs at s=7. They're on the rain-light edge of their cell.

**Implication**: distance-to-threshold features are necessary but not
sufficient. What's missing is **multi-axis joint distance** — a row
near 3 thresholds simultaneously is far more likely to flip than a row
near 1 threshold by the same Euclidean amount. The recipe XGB has
single-axis dist features (sm_dist, rf_dist, etc.) but tree splits
combine them axis-by-axis, not as a joint Euclidean shell.

**Action**: try a **boundary-shell feature**:
`shell = min(sm_dist, rf_dist, tc_dist, ws_dist) / IQR_norm`,
plus the count `n_axes_within_threshold_band` (how many of the 4 rule
axes are within ±10% of their threshold). This compresses
multi-axis-near-boundary into 2 scalars trees can split on. Cost ~10
min CPU, distinct from prior P3 instability features (which counted
discrete flips under perturbation, redundant w/ existing dist
features per the 24th saturation closure).

### B. The 80% orig-unseen 8-cat-tuple cells carry signal *only at the rule-cube level*.
80.1% of train rows live in cat-tuples never seen in orig, yet flip
rate is unchanged. This means **the NN's flip mechanism is
deterministic in the rule cube + continuous rule features alone**.
8-cat-tuple OTE features (already in recipe) are useful for OTHER
reasons (per-cell class probability calibration), but they're not
where additional flip signal lives.

**Implication**: skipping further OTE-key expansion (171-pair, etc.)
on principle was correct. The remaining flip signal is rule-feature
continuous geometry, not categorical density.

### C. Test ⊆ train manifold (AV-AUC ≈ 0.50); test ≠ orig (AV-AUC ≈ 0.69).
Anything we learn from train↔orig comparison applies UNIFORMLY to
test↔orig. There is no "use orig to fix test" lever — the orig
manifold is not where the test rows live. This **decisively closes
NN-on-orig and TE-from-original lever families** (already nulled
empirically — now we know structurally why).

### D. The non-rule shift in Rainfall_mm (+210 mm at train mean) is selection-bias on rule cells.
Rainfall_mm is shifted across the board, but conditional on
`norain=1` (Rainfall < 300), the within-bin distribution is
right-skewed in train vs orig. This is an artifact of the NN
preserving the threshold while sampling a different conditional.
**It does NOT mean the test-set has different rainfall than train**
— per the AV check, train↔test is identical. The shift only matters
when comparing to orig, not to test.

**Implication**: heavy-weight original augmentation
(2026-04-21 ruled out at w=20: −0.00026 OOF) is structurally
mis-specified — feeding orig rows at any weight tells the model the
WRONG within-cell distribution for test. Confirm: keep orig at w≤1×
per row only, or skip entirely. Heavy weights are net-negative
because the NN's distortion of the joint distribution within each
rule-cell is the very signal models need to learn.

### E. The flip predicate is a **NN-learned smooth manifold cut**, NOT random noise.
At s=7, the 1360 flipped Highs (going to Medium) have:
- Rain −1.48 SD lower (still <300, but at the upper-rain edge of "norain")
- Wind −0.95 SD lower (still >10, but barely)
- Temp −0.81 SD lower (still >30, but barely)
- Soil_M +0.74 SD HIGHER (still <25, but barely)
*All four shifts simultaneously toward the cell boundary*. This is
exactly what a smooth NN decision surface looks like in the original
6-d rule space: the boundary is a curved manifold cutting through the
"near-the-corner" subset of each cell. **Axis-aligned trees cannot
fit a curved 4-d manifold cleanly with only ~16k iterations.**

**Implication**: the only NN family that has cleared the magnitude
trap (RealMLP n_ens=1, gap +0.00027 in 3-stack) IS the right
architecture for this problem — it's just bounded by feature
representation. **The actionable next step is not another NN
architecture, it's giving the existing RealMLP a richer
boundary-aware feature set** (the shell features in (A) above) and
seeing if its standalone OOF improves.

### F. **Concrete experiment shortlist** (all unsubmitted, ≤30 min CPU each)
1. **Shell-feature recipe variant**: add `min_axis_dist`,
   `n_near_threshold` (count of 4 axes within ±5%/±10% of threshold),
   `mahalanobis_to_centroid` (centroid of orig same-rule-cube cell).
   Recipe XGB retrain, blend gate vs LB-best 4-stack. 25th saturation
   probe with mechanism-novel feature class (multi-axis joint
   geometry, not previously tested).
2. **RealMLP with shell features**: re-run kernel_realmlp with the
   3 new features added to its 19-raw set. Tests whether the +0.0003
   RealMLP edge (already in primary at α=0.20) extends with better
   boundary representation.
3. **Cell-Mahalanobis distance as a single feature**: per
   rule-cube cell, compute the centroid of orig rows in that cell on
   the 4 rule numerics, then feature = squared Mahalanobis distance
   from each train row to its cell's orig centroid. Captures the
   "how far from the orig manifold am I?" signal directly. Recipe
   XGB feature; ~15 min CPU.

All three are mechanism-distinct from the 24+ saturation
confirmations to date. Bayesian prior of LB lift (per pattern of
recent novel-feature experiments): ~10–20% each.

## Method (reproducible)
- `scripts/dist_shift/load_align.py` — column-aligned data + rule cols
- `scripts/dist_shift/univariate_shift.py` — KS / Cohen's d / JS table
- `scripts/dist_shift/conditional_shift.py` — class- and score-conditional shifts
- `scripts/dist_shift/joint_av.py` — AV classifier + cell-restricted AUC
- `scripts/dist_shift/flip_analysis.py` — within-cell flip detector
- `scripts/dist_shift/test_vs_orig.py` — sanity AV check on test
- Artefacts: `scripts/artifacts/_dist_shift_*.json` (5 files)
