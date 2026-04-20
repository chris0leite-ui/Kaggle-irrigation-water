# Next steps

Ranked plan with expected OOF deltas. **Current best:
LGBM+DGP tuned log-bias, 5-fold CV OOF = 0.97271** (boundary-LGBM ties
at 0.97284 within 1σ). LB reference points: tied pack 0.98114, leader
0.98219. Ten days to deadline, 10 LB submissions/day, 2 spent (baseline
LGBM 0.96972, pure rule 0.95835).

Updated 2026-04-20 after the DGP-features / boundary-LGBM / flip-detector
sweep. Steps 1–3 are closed (✓ or ruled out); step 4 is the new top
open bet.

## 1 · Burn one submission to calibrate CV ↔ LB ✓ done 2026-04-20

Submitted `submission_baseline_lgbm_tuned.csv` (LGBM + tuned log-bias).
**LB public = 0.96972** at rank 726 / 2357 (top 31 %). OOF 0.97097 vs
LB 0.96972 → −0.00125, inside one fold-std (~0.002). CV is
well-calibrated.

## 2 · Original Irrigation Prediction dataset ablation ✓ done 2026-04-20

`scripts/benchmark_external.py` concats 10k original rows into each
training fold. Tuned OOF 0.97124 vs 0.97097 baseline →
Δ = **+0.00027**, within fold-std. `scripts/transfer_check.py` (train
on 8k original, eval on 630k synthetic) hit 0.96278 — DGPs overlap
almost completely, the small delta reflects data-volume cap (10k ≪ 630k).

## 3 · ~~Plug domain features into LGBM~~ — ruled out 2026-04-20

`scripts/benchmark_fe.py` with 8 engineered water-balance cols
(`ET0_proxy`, `Kc_stage`, `ETc_proxy`, `Soil_deficit`, `Is_Rainfed`,
`Eff_Rainfall_active`, `Crop_x_Stage`, `Season_x_Region`) moved tuned
OOF to 0.97045 vs baseline 0.97097 (Δ = −0.00052). Trees at 127 leaves
aren't leaf-limited — prebuilt water-balance interactions add no splits.

## 3b · DGP features into LGBM ✓ done 2026-04-20 (new best)

`scripts/benchmark_dgp.py` adds 15 DGP-derived cols (indicators +
`dgp_score` + signed/absolute distances to the 4 thresholds).
Tuned OOF **0.97271** (Δ = +0.00174, ~2σ, improves every fold).
`scripts/boundary_lgbm.py` (model trained on boundary-band rows)
ties at **0.97284** within 1σ. The right features are the ones the
generator actually uses, not generic water-balance terms — that's why
step 3 failed and this succeeded.

## 4 · ~~Meta-stack / hard-gate the flip detector~~ — ruled out 2026-04-20

`scripts/gated_v3.py` evaluated four decision rules on the saved OOFs:

| Rule                          | OOF tuned bal_acc |
|---                            |              ---: |
| LGBM+DGP tuned (baseline)     |       **0.97271** |
| Hard-gate τ=0.95              |           0.95893 |
| Soft(rule + main) tuned       |           0.97249 |
| Meta-LGBM stacking tuned      |           0.97245 |

None beats LGBM+DGP. The "99.4% bal_acc on flipped rows" headline is
degenerate — on that subset the true label is anti-rule by
construction, so the specialist just learns "predict ¬rule". Deployed
via `P_flip > τ` gating, the selection set is polluted with false
positives (clean rows near boundaries where P_flip happens to be
high), and on those the specialist systematically predicts the wrong
label. Meta-LGBM saw this and passed through P_main.

**Take-away**: LGBM+DGP (0.97271) is the ceiling from the DGP-features
architectural family — the flip signal has already been internalized.
The remaining ~0.01 gap to the pack must live somewhere else.

## 5 · Seed-bag LGBM+DGP — new top open bet

Cheapest remaining win. 3–5 seeds of `scripts/benchmark_dgp.py`,
average OOF + test probs, retune log-bias on the averaged OOF.
Fold-std ≈ 0.00068 → expected √5 reduction ≈ **+0.0005–0.001** over
0.97271. Cost: 3–5× `benchmark_dgp.py` runtime.

## 6 · XGBoost with DGP features + blend

Re-train XGBoost with the same 15 DGP-derived cols, then blend at
prob level (geometric mean) with LGBM+DGP. Retune bias on the blended
OOF. Expected **+0.001–0.002** from model diversity — XGB's argmax
historically sits at ~0.962 (vs LGBM's 0.961), so it's strong enough
to contribute orthogonal signal (unlike MNLogit which was a null).

## 7 · Neural-net tabular model

`brief.md:74` explicitly states the synthetic labels were generated
by a deep-learning model. No NN has been tried here — no MLP,
TabNet, or torch/keras code in the repo. An MLP fit to the same
feature space might recover the near-threshold noise band in a way
axis-aligned trees structurally can't. Uncertain upside, non-trivial
setup cost. This is the only *untried model family*. Promote if
steps 5–6 leave us below 0.974.

## 8 · Ordinal-aware loss / Medium↔High sample-weighting

Residual confusion mass lives at Medium↔High. LGBM `multiclassova`
with sample weights that upweight adjacent-class misclassifications
is a small change. Expected +0.001 or null.

## 9 · LGBM HP refresh — ruled out 2026-04-20

Optuna TPE, 10 dims / 47 trials / 200k subsample, landed at 0.97047
prior-reweight — plateau, not ridge. Extrapolated full-630k delta
≤ +0.001.

## Suggested immediate action

Run step 5 (seed-bag LGBM+DGP). ~1 hour of compute, essentially free
insurance toward the next LB submit. If the averaged OOF breaks 0.9734
(= 0.97271 + 1σ), submit it and spend another LB slot to calibrate
the DGP-features delta. In parallel queue step 6 (XGB+DGP) — it's
the last major lever inside the tree family before we'd have to
reach for a neural net.
