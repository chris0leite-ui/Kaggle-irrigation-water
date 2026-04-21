# Next steps

Ranked plan with expected OOF deltas. **Current best:
LGBM+DGP tuned log-bias, 5-fold CV OOF = 0.97271**. LB reference
points: tied pack 0.98114, leader 0.98219. Ten days to deadline,
10 LB submissions/day, 2 spent (baseline LGBM 0.96972, pure rule
0.95835).

**Reframe (2026-04-21): the DGP is a deterministic NN function, not
rule + noise.** See CLAUDE.md §"DGP is a learnable NN function" and
`plots/eda/dgp_residuals.html` for the evidence. The synthetic labels
come from the host's label-generating NN (per `brief.md:74`); "flips"
are *not* random, they're the NN's predictions deviating from the
rule because the NN uses features the rule ignores. Ceiling is 100 %,
not rule + irreducible noise. This promotes NN / MLP to the top of
the list.

## 1 · Burn one submission to calibrate CV ↔ LB ✓ done 2026-04-20

Submitted `submission_baseline_lgbm_tuned.csv`. **LB public = 0.96972**.
OOF↔LB gap = −0.00125, within 1σ. CV is well-calibrated.

## 2 · Original Irrigation Prediction dataset ablation ✓ done 2026-04-20

Concat 10k → tuned OOF 0.97124 (Δ = +0.00027, within 1σ). DGPs
overlap almost completely; small delta reflects data-volume cap.

## 3 · ~~Plug domain features into LGBM~~ — ruled out 2026-04-20

8 water-balance cols moved tuned OOF to 0.97045 (Δ = −0.00052).
Trees already discover these interactions.

## 3b · DGP features into LGBM ✓ done 2026-04-20 (new best)

15 DGP-derived cols → tuned OOF **0.97271** (Δ = +0.00174, ~2σ).
`scripts/boundary_lgbm.py` ties at 0.97284.

## 4 · ~~Meta-stack / hard-gate the flip detector~~ — ruled out 2026-04-20

`scripts/gated_v3.py` evaluated hard-gate (0.95893), soft blend
(0.97249), meta-LGBM stacking (0.97245). None beat LGBM+DGP. The
flip-direction specialist's 99.4 % bal_acc is degenerate (on the
flipped subset, `true_label = ¬rule` by construction); routing via
`P_flip > τ` pulls in false-positive clean rows where the specialist
systematically predicts the opposite of the true label.

## 5 · ~~Neural-net tabular model~~ — **plateaued 2026-04-21**

Three MLP variants on identical seed=42 folds plateaued 0.007 below
LGBM+DGP (0.97271):

| variant                | tuned OOF |         Δ vs LGBM+DGP |
|---                     |     ---:  |                  ---: |
| v1 plain CE            |  0.96437  |              −0.00834 |
| **v3 Balanced Softmax**|**0.96596**|          **−0.00675** |
| v4 LDAM-DRW (fold 1)   |    ~0.962 |     killed after fold 1 |

Balanced Softmax (Menon 2021) *did* work as designed — it's the
training-time equivalent of post-hoc log-bias; residual bias shift
collapsed from `{+1.33, +1.57, +3.40}` (plain CE) to `{+0.3, 0, 0}`.
But the raw ceiling is capacity-limited: 3-layer 256-128-64 with 50k
params can't approximate what LGBM+DGP's axis-aligned boosting finds
on rule-threshold features. LDAM-DRW's effective-number weights
degenerated (β=0.9999 at n_c ≫ 10⁴ → uniform); only the margin was
active and it wasn't enough.

**LGBM+DGP × MLP+BalSoft prob-level blend is also null**:
geometric-mean sweep over w∈[0, 0.5] gave best w=0.15 → 0.97276 (Δ =
+0.00005, below fold-std noise 0.00068). MLP errors are correlated
with LGBM errors — both miss the same boundary-band flips.

Revisit only with a substantially larger/deeper architecture
(FT-Transformer, NumEmb + wide MLP, tabular-ResNet) or as a cRT
(Kang 2020) fine-tune on the v3 backbone (see §9).

Artefacts (on branch `claude/improve-balanced-accuracy-v1UtX`):
`scripts/mlp_{dgp,balsoft,ldam}.py`, `scripts/blend_lgbm_mlp.py`,
`scripts/artifacts/oof_mlp_{dgp,balsoft}.npy`.

## 6 · Rule × non-rule pairwise FE for LGBM — **NEW TOP OPEN BET**

Specifically engineer the interactions the host's NN most likely
learned:

- `Humidity × Soil_Moisture` (wet-air × dry-soil)
- `Previous_Irrigation_mm × Rainfall_mm` (recent water × concurrent water)
- `Electrical_Conductivity × Soil_Moisture`
- `Field_Area_hectare × dgp_score`
- `Humidity × Crop_Growth_Stage` (categorical interaction)

These are small effects individually (d ~ 0.1 at score=3) but together
they're the signal the NN is using to place labels inside the
boundary-band. The MLP experiment confirmed this signal is real (v3
BalSoft +0.0016 vs plain CE) but not capturable by a small MLP.
Trees can already discover such interactions, but explicit columns
may stabilise at a low leaf count or inside a seed-bag. **This is
the cheapest remaining untried lever that matches the diagnostic**
from the 2026-04-21 DGP-residuals EDA.

Plan: copy `scripts/benchmark_dgp.py` → `benchmark_dgp_fe.py`, add
the five interaction columns, re-run the 5-fold pipeline, compare
tuned OOF to 0.97271.

Expected delta: +0.0005 – 0.002.

## 7 · Seed-bag LGBM+DGP

3–5 seeds of `scripts/benchmark_dgp.py`, average OOF + test probs,
retune log-bias. Fold-std ≈ 0.00068 → √5 reduction ≈ **+0.0005–0.001**.
Cheap insurance. Do after §6 lands (or combine: seed-bag the
pairwise-FE variant).

## 8 · XGBoost with DGP features + LGBM+DGP blend

Geometric-mean blend, retune bias. Expected **+0.001–0.002** from
model diversity (unlike MNLogit / MLP / balanced-ensemble, XGB on the
same features is strong enough to plausibly contribute orthogonal
signal). Validate error orthogonality (Jaccard over OOF error sets)
before committing to the full sweep — the MLP blend null made this
check cheap insurance.

## 9 · cRT (Kang 2020) decoupled retrain on v3 MLP

One tail-aware lever not yet tried from the 2026-04-21 user guidance:
freeze v3's feature extractor, retrain **only** the final linear
layer with class-balanced sampling (or τ-normalise the weights). ~1
min per fold vs 3 min for a full retrain. Low expected value after
the v3 blend null, but it's the cheapest MLP variant not yet
executed, and it reshapes the error geometry differently from
BalSoft (decoupled representation vs joint logit adjustment) so it
may clear the error-orthogonality pre-check that v3 failed.

## 10 · Ordinal-aware loss / Medium↔High sample-weighting

Confusion mass lives at Medium↔High. LGBM `multiclassova` with
weights that upweight adjacent-class misclassifications. The 2026-04-21
EDA confirms flips are **always to the adjacent class** (score=3
→ Medium, never → High), so an ordinal model is a structural match
to the noise pattern. Expected +0.001 or null.

## 11 · LGBM HP refresh — ruled out 2026-04-20

Optuna TPE landed at 0.97047 prior-reweight, same plateau as
baseline.

## Suggested immediate action

**Execute §6 (pairwise FE).** After the MLP plateau (§5) and the
balanced-ensemble null, pairwise FE is the only untried lever that
directly targets the non-rule-feature signal flagged by the
2026-04-21 EDA (Previous_Irrigation_mm, Humidity, Electrical_
Conductivity with d ≈ 0.08–0.11 at score=3). It's also the cheapest
remaining experiment — a ~30-line patch to `benchmark_dgp.py`, one
5-fold run (~3 min on CPU), no new dependencies.

If §6 clears +0.001, stack §7 (seed-bag) and §8 (XGB+DGP blend) on
top for a combined ~+0.003 lift off the current 0.97271 baseline. If
§6 is null, revisit the pack's recipe via DGP archaeology (parked) —
the 0.007-to-pack gap would then require something outside our
current attack surface.
