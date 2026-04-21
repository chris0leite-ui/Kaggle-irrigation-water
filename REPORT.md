# Work Report

A structured readout of what was done, what was observed, and what was
concluded. Lives alongside `CLAUDE.md` (the running log),
`DOMAIN.md` (the agronomy primer) and `LEARNINGS.md` (the portable
patterns).

## 1. Problem summary

- **Task**: 3-class classification (`Low` / `Medium` / `High`) of
  `Irrigation_Need` on 19 tabular features (11 numeric, 8 categorical).
- **Metric**: balanced accuracy (macro-recall).
- **Data**: 630k train rows, 270k test rows, no missingness, no
  train/test categorical-vocabulary drift.
- **Priors**: Low 58.7% / Medium 37.9% / High 3.3% — badly imbalanced;
  under balanced accuracy, `High`-class recall drives the score.

## 2. Data observations

- **Missingness**: none in train or test.
- **Drift**: test shares train's categorical vocabulary exactly. A
  formal numeric-drift (KS) pass is still TODO.
- **Top numeric signals (F-stat on a stratified 50% EDA subsample)**:
  `Soil_Moisture` (41k) ≫ `Wind_Speed_kmh` (11k) ≈ `Temperature_C`
  (11k) > `Rainfall_mm` (3.5k). All other numerics are < 100.
- **Top categorical signals (chi²)**: `Crop_Growth_Stage` (97k) ≫
  `Mulching_Used` (28k) ≫ everything else (< 1k).
- So ~6 features carry the bulk of the signal; the rest look like
  either noise or near-uniform structure.
- EDA uses a stratified 50% subsample (seed=42); other 50% held out.
- EDA report: `plots/eda/report.html` (self-contained, base64 images).

## 3. Models tried

All 5-fold stratified CV, seed 42. OOF balanced accuracy.

| Tier                     | Model / rule                             | OOF bal_acc |      Δ vs prev |
|---                       |---                                       |         ---:|           ---: |
| Floor                    | Majority-class (all Low)                 |     0.33333 |             –  |
| Floor                    | Stratified random                        |     0.33384 |      +0.00001  |
| Heuristic                | H1 — Soil_Moisture alone                 |     0.62911 |      +0.29527  |
| Heuristic                | H2 — raw water balance, equal z-weights  |     0.60606 |      −0.02305  |
| Heuristic                | H3 — H2 + Kc + mulch + soil capacity     |     0.63041 |      +0.00130  |
| Linear (MNLogit)         | F1 — minimal balance, 4 feats, argmax    |     0.52337 |             –  |
| Linear (MNLogit)         | F1 — tuned log-bias                      |     0.64721 |      +0.12384  |
| Linear (MNLogit)         | F2 — balance + Kc + deficit + mgmt, 19f  |     0.66429 |             –  |
| Linear (MNLogit)         | F2 — tuned log-bias                      |     0.78074 |      +0.11645  |
| Linear (MNLogit)         | F3 — full structural, 48 feats           |     0.61680 |             –  |
| Linear (MNLogit)         | F3 — tuned log-bias                      |     0.73294 |      +0.11614  |
| Tree (LGBM)              | argmax                                   |     0.96135 |             –  |
| Tree (LGBM)              | prior-reweight argmax                    |     0.97065 |      +0.00930  |
| **Tree (LGBM)**          | **tuned log-bias**                       | **0.97097** |      +0.00032  |
| Tree (XGBoost)           | argmax (per-fold ~0.961–0.964)           |      ~0.962 |             –  |
| Tree (LGBM+FE)           | argmax (8 engineered cols)               |     0.96133 |      −0.00002  |
| Tree (LGBM+FE)           | tuned log-bias (8 engineered cols)       |     0.97045 |      −0.00052  |
| Transfer check (orig→syn)| LGBM trained on 8k original, eval 630k syn, tuned | 0.96278 |    −0.00819 |
| Tree (LGBM+EXT)          | argmax (concat 10k original)             |     0.96208 |      +0.00073  |
| Tree (LGBM+EXT)          | prior-reweight argmax (concat 10k orig)  |     0.97097 |      +0.00032  |
| **Tree (LGBM+EXT)**      | **tuned log-bias (concat 10k original)** | **0.97124** |   **+0.00027** |
| Blend                    | LGBM + MNLogit Fk, sweep w ∈ [0, 0.5]    |     0.97097 |   +0.00000 (null) |
| Orthogonal model         | Heuristic (8-signal z-sum + 2 cuts, 630k, learned thresholds) | 0.60012 |    – |
| Orthogonal model         | Gaussian NB (FE cols, 630k)              |     0.75172 |      +0.15160  |
| Orthogonal model         | Multinomial LR balanced (FE cols, 630k)  |     0.83009 |      +0.07837  |
| Orthogonal model         | EBM with pairwise interactions (200k)    |     0.96106 |      +0.13097  |
| Tree (LGBM HP-tuned)     | prior-reweight argmax (200k, TPE best)   |     0.97047 |  −0.00050 vs baseline |
| Tree (LGBM+DGP)          | argmax (15 DGP-derived cols + distances) |     0.96349 |             –  |
| Tree (LGBM+DGP)          | prior-reweight argmax                    |     0.97250 |      +0.00901  |
| **Tree (LGBM+DGP)**      | **tuned log-bias (new best, 2026-04-20)**| **0.97271** |   **+0.00021** |
| Imbalanced-ensemble      | BalancedRandomForest + DGP (tuned)       |     0.96535 |      −0.00736  |
| Imbalanced-ensemble      | EasyEnsemble + DGP (tuned)               |     0.96932 |      −0.00339  |
| Imbalanced-ensemble      | RUSBoost + DGP (tuned)                   |     0.96666 |      −0.00605  |
| Blend                    | LGBM+DGP ⊗ {BRF, Easy, RUS}, sweep       |     0.97272 |      +0.00001 (null) |
| NN (MLP+DGP v1)          | plain CE, argmax (3×256→128→64, embed)   |     0.96184 |             –  |
| NN (MLP+DGP v1)          | plain CE, tuned log-bias                 |     0.96437 |      +0.00253  |
| **NN (MLP+BalSoft v3)**  | **Balanced Softmax (Menon 2021) tuned**  | **0.96596** |   **+0.00159 vs v1** |
| NN (MLP+LDAM-DRW v4)     | LDAM + eff-num CB weights (fold 1 only)  |      ~0.962 |   killed — see §5 |
| Blend                    | LGBM+DGP ⊗ MLP+BalSoft, geometric w=0.15 |     0.97276 |      +0.00005 (null) |
| Routed/spec (hybrid_v3)  | routed{0,1,2}+XGB + spec-{6,7,8} override|    0.97352 |      LB **0.97271**  |
| 3-way log-blend (greedy) | 0.45 hybrid_v3 + 0.40 routed_v3 + 0.15 spec_678 |  0.97375 | LB **0.97296** |
| Binary-High head (overfit)| hybrid_lgbmxgb + binhigh logit-add λ=+0.60 |   0.97398 | LB 0.97212 (gap blew up) |
| Binhigh on greedy (fixed bias) | same head stacked on greedy, no retune | 0.97375 (λ=0) | — (no submit) |
| **Non-rule-only blend**  | **greedy + XGB(non-rule features only) @ α=0.15** | **0.97421** | **LB 0.97352 ← best** |
| LGBM variant of nonrule  | same 13 features, different tree style   |     0.97415 |      null vs base    |
| Feature-subset bag       | 5×XGB on 4-of-7 non-rule subsets         |     0.97383 |      null vs base    |
| Nonrule + rule_pred      | 13 non-rule + rule_pred + dgp_score      |     0.97382 |      null vs base    |
| Weighted-shift (w=100)   | upweight flip rows 100×, shift-5 XGB     |     0.97375 (α=0) | — (null)    |
| LB reference             | LB tied pack (~100 teams)                |     0.98114 |             –  |
| LB reference             | LB leader (Chris Deotte)                 |     0.98219 |             –  |

Key read-outs

- **Soil_Moisture is the single dominant feature.** H1 (just signed
  Soil_Moisture + 2 thresholds) already covers 2/3 of the distance from
  random (0.333) to competitive (0.971). H2 is worse than H1 because
  equal-weight z-scoring dilutes a dominant signal with noisier axes.
- **Nonlinear interactions are where the real lift lives.** H3 vs
  LGBM = +0.34 of bal_acc, on the same underlying physics features.
  That gap isn't "the equation is wrong"; it's "additive combinations
  of the equation's terms miss the Medium↔High decision surface".
- **Bias tuning is model-agnostic and big when the base model is
  uncalibrated.** +0.12 on each MNLogit formula, +0.010 on LGBM. The
  leaderboard pack almost certainly already applies some form of it;
  our LGBM-tuned 0.971 is not a surprise to them.
- **LGBM × MNLogit blending is a null.** Δ = 0 at every mixing weight.
  MNLogit at 0.78 tuned is simply too far below LGBM's 0.971 to
  contribute orthogonal signal.
- **Hand-engineered domain features add nothing to LGBM.** Injecting
  `ET0_proxy`, `Kc_stage`, `ETc_proxy`, `Soil_deficit`, `Is_Rainfed`,
  `Eff_Rainfall_active`, `Crop_x_Stage`, `Season_x_Region` (+8 cols,
  27 total) moved tuned OOF from 0.97097 → 0.97045 (Δ = −0.00052,
  smaller than the 0.00088 fold std). Trees already discover these
  interactions; prebuilt features were the hypothesised fallback for
  near-leaf-limit splits, and that hypothesis doesn't hold at the
  current leaf count. Bias solution was essentially unchanged
  (Low +0.23, Medium +0.57, High +3.40).
- **Original Irrigation Prediction dataset overlaps the synthetic DGP
  almost completely, but concat only adds +0.00027.** Transfer check
  (LGBM on 8k original rows → predict 630k synthetic, tuned bias)
  hits 0.96278, just 0.00819 below the 5-fold baseline trained on
  63× more data. Concatenating the full 10k into each training fold
  moves tuned OOF to 0.97124 (Δ = +0.00027, < 1σ fold std = 0.00068).
  Tiny positive, not the silver bullet — the pack at 0.98114 is not
  getting there through this lever. Bias solution Low +0.13 /
  Medium +0.67 / High +3.40 (Low slightly relaxed vs baseline's
  +0.23, otherwise identical).
- **Confusion-matrix mass lives at Medium↔High.** LGBM tuned still
  flips ~4k Medium→High and ~875 High→Medium on OOF; the heuristic
  makes that error 50× more often. This is where any further gain must
  come from.
- **Independence-to-interaction gap is ~0.22.** A controlled ladder on
  identical 5-fold folds: heuristic 0.600 → Gaussian NB 0.752 → LR 0.830
  → EBM 0.961 → LGBM 0.971. Every +0.08 step is bought by letting the
  model represent more interaction structure. Rules out any
  independence-based or linear stacking candidate as a source of
  orthogonal signal worth the compute.
- **LGBM hyperparameter optimization did not beat default-ish.**
  60-trial Optuna TPE sweep (10-dim search: lr, num_leaves,
  min_data_in_leaf, feature/bagging fractions, freq, λ₁/λ₂, max_depth,
  min_gain) on a 200k subsample found best `num_leaves=46, max_depth=3,
  lr=0.064` at 0.97047 prior-reweight — roughly level with the baseline
  (num_leaves=127, defaults) after scale-up. TPE preferred shallow +
  regularized, but that's a different *shape* of optimum reaching the
  same plateau. Extrapolated full-630k delta ≤ +0.001. Baseline HPs
  are near-optimal at this feature set; gains need a different lever.
- **MLP plateaus at ~0.966 tuned OOF — capacity-bound, not loss-bound
  (2026-04-21).** A 3-layer tabular MLP (256-128-64, 50k params, BN +
  dropout 0.15, Adam + cosine LR, embedded cats, all 26 DGP-enriched
  numerics) hits 0.96437 under plain CE + post-hoc log-bias (v1) and
  0.96596 under Balanced Softmax (v3, Menon 2021 — training-time
  logit shift `z + log π` so raw argmax is Bayes-optimal). v3's
  residual post-hoc bias shift collapses to `{Low: +0.3, 0, 0}` vs
  v1's `{+1.33, +1.57, +3.40}` — BalSoft successfully substitutes
  for coord-ascent bias tuning. But the absolute ceiling is
  **persistently 0.007 below LGBM+DGP's 0.97271** across every fold
  (std ≈ 0.0005), so the bottleneck is model capacity / axis-aligned
  vs smooth boundary, not loss calibration. LDAM-DRW (v4, Cao 2019)
  with effective-number class weights (β=0.9999) degenerated to
  ~uniform at our 370k/239k/21k sample sizes — only the margin term
  was active, fold 1 landed at 0.96240 argmax, killed after fold 1.
- **MLP × LGBM+DGP blend adds +0.00005 — null (2026-04-21).**
  Arithmetic and geometric sweeps over `w ∈ [0, 0.5]` with coord-ascent
  bias retune per weight. Best: geometric `w=0.15 → 0.97276` vs LGBM
  alone 0.97271, well below the 0.00068 fold-std noise floor. Confusion
  diff: +44 High recalled, +459 Medium→High mistakes. The MLP's errors
  are **not** orthogonal to LGBM's; both models miss the same
  boundary-band flips. Consistent with the balanced-ensemble blend
  null — a second model of similar bias-tuned calibration but lower
  standalone accuracy can't contribute diversity. New rule:
  demonstrate per-row error orthogonality (Jaccard over OOF error
  sets) before investing in a full blend sweep.

## 3.x Soft-blend ensemble (2026-04-21, NEW LB BEST 0.97296)

Greedy forward-selection over 5 regenerated OOF pipelines. Starts from
the best standalone (xgb_hybrid_v3 at OOF 0.97352) and iteratively adds
the component whose log-blend at the OOF-best α most improves tuned
bal_acc. Winner after 2 additions:

    w = (0.45, 0.40, 0.15) on (hybrid_v3, routed_v3, spec_678)
    log-blend, tuned bias [0.132, 0.569, 3.401]
    OOF bal_acc = 0.97375, LB public = 0.97296

Δ LB vs prior best = +0.00025, matching the +0.00023 OOF prediction
(OOF→LB gap 0.00079, consistent with 0.00081 on standalone hybrid_v3).
No OOF-selection overfit — the greedy log-blend found real
model-diversity signal.

**Three independent blend strategies converged to the same ceiling**
(OOF ~0.9738 ± 0.00002):
  - greedy log-blend (this section): 0.97375
  - class-asymmetric High-prob mixing on hybrid_v3 (`blend_high_weighted`
    mean_other_high γ=+0.40): 0.97377
  - cross-lineage with main branch's `hybrid_lgbmxgb_blend` (OOF 0.97362):
    pairwise best w_ours=0.95 → 0.97376

Log-space blending on this problem has saturated. Further LB lift
needs an orthogonal model class (MLP retry with larger architecture,
CatBoost with DGP features) or a new structural lever outside the
current tree-ensemble basin. Logistic-regression meta-stack on
concat(P_hv3, P_routed, P_dgp, P_xgbdist) with class_weight=balanced
gave 0.97348 — underperformed the simple greedy blend because the
component probs are too correlated for 12-feature logistic to add signal.

## 4. Strategy and next steps (updated end-of-day 2026-04-21)

Current best: **greedy log-blend OOF 0.97375 / LB 0.97296**
(+0.00025 LB vs prior hybrid-v3 best 0.97271, −0.00818 below the
0.98114 tied pack, −0.00923 below the 0.98219 leader). The pack
ceiling is now known to be a public-notebook CSV-blend artifact,
not a modeling breakthrough available through our own pipelines.

Since the morning session, a broad sweep of tree-family + blending
+ training-data-quality levers converged on an OOF ceiling of
**~0.9738** across three independent strategies (greedy log-blend,
class-asymmetric High-prob mixing, cross-lineage blend with main's
OOFs). Training-data-quality levers (heavy original augmentation,
score-stratified CV) are net-negative. This signature — **multiple
independent approaches landing within ±0.00002** — is an
architectural bottleneck, not a tuning problem. The remaining
productive lever is model-class diversity outside the tree-ensemble
basin.

Ranked by expected ROI / effort:

1. **[TOP BET] Large-capacity tabular NN.** Rationale: the DGP is
   a deterministic NN (confirmed by the 2026-04-21 residuals EDA),
   and every non-NN lever we've tried plateaus at 0.9738. The 50k-
   param MLP we tested was 1-2 orders of magnitude below a serious
   tabular NN — capacity-bound, not structurally wrong.

   Candidates in priority order:
   - **FT-Transformer** (1-3M params; per-feature tokens + self-
     attention). Structural match to "generator learned a function
     of all feature interactions."
   - **NumEmb + wide MLP** (~500k params, learnable numeric
     embeddings per feature, 3×512 hidden). Cheaper than
     FT-Transformer, still 10× the capacity of the prior plateaued
     attempt.
   - **Tabular-ResNet** (skip connections, GELU, strong
     regularisation).

   Pre-check gate after fold 1: compute OOF error Jaccard vs
   `oof_xgb_hybrid_v3.npy`. Decision:
   - Jaccard ≥ 0.90: kill, NN is mimicking the tree ensemble.
   - Jaccard < 0.85: finish all 5 folds, blend into greedy.
   - 0.85–0.90: finish folds but cap blend-lift expectation at
     +0.00015 (same rule that identified the MLP / balanced-
     ensemble / MNLogit nulls).

   Expected: **+0.001 to +0.003 LB** if the NN is genuinely
   orthogonal; 0 if it plateaus at the tree ceiling. Compute budget
   ~1-2 h (GPU strongly preferred).

2. **Seed-bag the greedy log-blend** (3 seeds of the LGBM and XGB
   components with the same blend weights). Variance reduction on
   the current best; expected **+0.0001–0.0003 LB**. ~60 min
   compute. Guaranteed-safe hedge if (1) is killed or plateaus.

3. **Spec on score {3}** (102 k rows, 95 % Low / 5 % Medium,
   4.8 % rule-error rate). Parallel to spec-{6,7,8}. Expected
   **+0.0001–0.0003** if fused into the greedy hybrid. ~15 min.

4. **Per-score log-bias tuning** (30 params: 10 score bins × 3
   classes vs 3 global). Nested CV to avoid overfit; higher risk
   but higher ceiling. Expected +0.0003–0.0008. ~30 min.

5. **Ordinal-aware loss** for Medium↔High confusion (LGBM
   `multiclassova` + adjacent-class upweighting, or cumulative-link
   objective). Still untested, low priority — log-bias already
   nearly saturates macro-recall post-hoc.

6. **DGP archaeology (parked)**. The rule structure is already
   fully reverse-engineered; the remaining 10,304 flips are NN-
   deterministic and approximated by (1) above. No independent
   payoff to further archaeology.

### Final-selection candidates (pick 2 for Kaggle's final lock)

- **Primary**: `submission_blend_greedy_w045_040_015.csv`
  (OOF 0.97375 / LB 0.97296). Verified best.
- **Safe fallback**: `submission_xgb_hybrid_v3_routed012_spec678.csv`
  (OOF 0.97352 / LB 0.97271). Single-pipeline, minimal variance,
  private-LB robust.

If (1) lands a verified LB lift before close, swap the fallback for
the NN-augmented blend.

### Ruled out in today's session (short list)

- Hard-vote plurality/weighted/Borda/veto blends: pairwise committee
  agreement 0.99+ caps the lift at ~0.003; most plurality designs
  also demote the rare class and are LB-negative.
- Logistic meta-stacker (class-weight=balanced): 0.97348, below
  greedy. Components too correlated for LR to add signal on 12
  probability features.
- Cross-lineage blend with main's hybrid_lgbmxgb_blend: OOF 0.97376
  at w_ours=0.95 (null). Shared anchor on hybrid_v3 = no compounding.
- Heavy-weight original augmentation (w=20): −0.00026 on xgb_dist;
  Medium recall drops because rule-perfect external data biases
  training away from the NN flip signal present in both train and
  test.
- (target × dgp_score) stratified CV: unchanged tuned OOF; default
  StratifiedKFold is already well-balanced at 630k rows.

## 4.1 Strategy update — late-night 2026-04-21 (LB 0.97352)

**Session snapshot 2026-04-21 late-night**: after a 13-experiment
session on branch `claude/improve-balanced-accuracy-dXEHv`, current
best is **greedy 3-way log-blend + non-rule-features-only XGB
@ α=0.15 → OOF 0.97421 / LB 0.97352** (gap 0.00069 — tightest of
any submission). Pack at 0.98114 remains +0.00762 above; leader
0.98219 +0.00867. LB budget 6/10 used today, 4 remaining.

**One architectural win**: a 3-class XGB trained only on the 13
non-rule features (Soil_Type, Soil_pH, Organic_Carbon, EC, Humidity,
Sunlight_Hours, Crop_Type, Season, Irrigation_Type, Water_Source,
Field_Area_hectare, Previous_Irrigation_mm, Region) — standalone
bal_acc just 0.570 (barely above random) — log-blended at 15 %
weight into the greedy stack lifts LB +0.00056. Validates the
hypothesis that the host's NN label generator uses non-rule features
to perturb labels from the rule's 0.96097-ceiling prediction.

**Six nulls confirming the lever is cornered within its architecture**:

| Attempt | Fixed-bias lift vs base |
|---|---|
| LGBM variant of nonrule | +0.00000 (best α=0.05 ≈ base) |
| EBM variant of nonrule | aborted (fold 1 = 0.424, 29 min/fold) |
| Feature-subset bag (5×4/7) | +0.00009 (below XGB's +0.00047) |
| Nonrule + rule_pred+dgp_score | +0.00007 (loses orthogonality) |
| Shift-5 target + non-rule | monotonic negative |
| Shift + weight=100 | standalone below rule, blend negative |

**Methodology rules earned this session** (now in LEARNINGS.md):

1. **Fixed-baseline-bias sweep is the pre-LB filter.** If a new
   component doesn't lift OOF with the baseline's bias reused
   as-is, retuning bias manufactures fake lift that vanishes on LB.
2. **Real LB delta ≈ 1/3 OOF delta** when stacking tuned blends on
   tuned baselines. Architectural levers have 1:1 transfer.
3. **Gap shrinkage is the signature of honest architectural
   signal.** Greedy → greedy+nonrule: gap went 0.00079 → 0.00069.
4. **Rank/Borda blending is strictly dominated** by prob/log-space
   for 3-class log-bias-tuned decision rules.
5. **Diversity from "ignoring a feature class"** is more powerful
   than diversity from "using it differently" when the baseline
   already ensembles over the dominant class. XGB-nonrule wins by
   being rule-free, not by being a tree.
6. **Shift-target framings collapse** when any target class owns
   ≥95% of rows; boosted trees early-stop on the trivial majority.

Ranked by expected ROI / effort for the remaining days. Current
best: greedy+nonrule blend, OOF = 0.97421 / LB = 0.97352.

1. **Seed-bag XGB-nonrule** (5 seeds, ~20 min). Cheapest remaining
   insurance on the only architecturally-diverse leg. Expected
   **+0.00005 – 0.0002** LB, boring but not zero.
2. **Pseudo-labeling via current LB-best.** Previously null at
   τ=0.95 on the weaker hybrid_lgbmxgb base. Retry with the
   stronger greedy+nonrule base. Estimate: **+0.0002 – 0.001**
   if the stronger base produces cleaner high-confidence labels.
3. **Self-distillation on greedy+nonrule predictions.** Train a
   fresh XGB to match the current ensemble's probabilities on train
   rows, then blend back in. Forces the model to replicate the
   ensemble's decision surface via a different inductive path.
   Estimate: +0.0001 – 0.0005.
4. **Rule × non-rule pairwise FE on greedy base models.**
   Previously null on hybrid_lgbmxgb_blend (blend weight collapsed
   from 0.45 to 0.05 when FE cols added). Retest on the greedy base
   now that we have the fixed-bias methodology. Expected 0.0002 or
   null; confirms whether the null was architecture-specific.
5. **Ordinal-aware loss / Medium↔High sample-weighting.** EDA
   confirmed flips are *always* to the adjacent class. LGBM
   `multiclassova` with upweighted adjacent-class mistakes or a
   cumulative-link objective is a structural match to the noise
   pattern. Estimate: +0.001 or null.
6. **cRT on a fresh MLP backbone.** The 2026-04-21 MLP work
   plateaued at 0.966; not yet re-tried with non-rule features as
   the primary input stream. FT-Transformer or wide-MLP with
   non-rule features only might match the XGB-nonrule lift in a
   different architecture — orthogonal blend candidate.
7. **DGP archaeology (parked).** Reverse-engineer the synthetic
   generator. High effort, unclear payoff — revisit only if stuck
   above 0.975 after §1-§6 land.

Ruled out (see §5 + LEARNINGS.md methodology rules): balanced-
ensemble blends, hand-crafted water-balance cols in LGBM, standalone
MLP as replacement for LGBM, MLP × LGBM prob-level blend, LGBM HP
refresh, MNLogit blend, rank/Borda blending, binary-High head
stacking on tuned blends, per-cell logistic, shift-5 framing, LGBM/
EBM/feature-subset/+rule_pred variants of nonrule.

Minimum-viable first submission: the current
`submissions/baseline_lgbm_tuned.csv` (LGBM + tuned log-bias, OOF
0.97097). Sending it is cheap information — it tells us the LB gap
more accurately than our guess. Decision pending: burn one of the 10
daily submissions to calibrate, or wait until features/blend are
ready.

## 5. Rejected ideas

- **Equal-weight z-score fusion of water-balance axes (H2).** Worse
  than using Soil_Moisture alone (H1) because it dilutes the dominant
  signal. If we come back to hand-weighted scores, weights must be
  proportional to per-axis informativeness (F-stat or similar), not
  uniform.
- **LGBM + MNLogit blending.** Zero contribution from the linear
  model at any mixing weight. Parked as a stacking option only (use
  MNLogit OOF probs as additional features inside LGBM) — and even
  that is low expected value given how much weaker the linear model
  is.
- **CatBoost standalone.** Fold 1 argmax 0.96000 ≈ LGBM/XGB — no model-
  level edge, 23 min/fold training cost. Killed after fold 1. Could
  revisit only for a full 4+ model blend late in the competition if
  compute budget allows.
- **Hand-engineered water-balance features inside LGBM.** Eight cols
  from MNLogit-F2 / heuristic-H3 (ET0_proxy, Kc_stage, ETc_proxy,
  Soil_deficit, Is_Rainfed, Eff_Rainfall_active, Crop_x_Stage,
  Season_x_Region) ran in `scripts/benchmark_fe.py` — tuned OOF
  0.97045 vs baseline 0.97097 (Δ = −0.00052, within 1σ fold noise of
  0.00088). LGBM at 127 leaves / 200 min_data_in_leaf was clearly not
  leaf-limited on this dataset, so prebuilt interactions add no new
  splits. Artefacts: `scripts/artifacts/oof_lgbm_fe.npy`,
  `scripts/artifacts/test_lgbm_fe.npy`,
  `submissions/submission_lgbm_fe_tuned.csv`. Could revisit only if
  we ever retrain with a much smaller leaf budget or a tiny subset.
- **Balanced-ensemble methods on DGP features (2026-04-21).**
  BalancedRandomForest, EasyEnsemble, and RUSBoost (all from
  `imbalanced-learn`) run under identical 5-fold CV on the 34-col
  DGP-enriched feature set, each with base-learner configs tuned to
  avoid the known 3-class SAMME stump-collapse failure mode. Tuned
  OOF bal_acc: Easy 0.96932, RUSBoost 0.96666, BRF 0.96535 — all
  below LGBM+DGP 0.97271. Pairwise and 3-way blends with LGBM+DGP
  saturate at Δ ≤ +0.00008 (within fold noise); BRF gets zero weight
  in every blend config. These methods produce pre-balanced
  probabilities so log-bias has nothing to correct — argmax and tuned
  scores are within 0.002 of each other, vs LGBM's +0.0092 log-bias
  lift. The mechanism overlap means per-tree majority undersampling
  is not a distinct lever from post-hoc log-bias at this feature set;
  both pick the same balanced-accuracy operating point. Code was not
  retained (null result); full methodology and numbers live in this
  document and `CLAUDE.md` 2026-04-21 session entry.
- **Standalone MLP + DGP features as replacement for LGBM+DGP
  (2026-04-21).** Three variants on identical 5-fold seed=42 folds,
  3-layer tabular MLP (256-128-64, 50k params, BN, dropout 0.15,
  Adam + cosine LR, 8 embedded cats + 26 numerics including 15
  DGP-derived cols): v1 plain CE 0.96437 tuned, v3 Balanced Softmax
  (Menon 2021 / Ren 2020, loss `CE(z + log π, y)`) 0.96596 tuned
  (+0.00159, ~2σ), v4 LDAM-DRW (Cao 2019) killed at fold 1 = 0.96240
  because the effective-number class weights (β=0.9999) degenerate
  to ~uniform at our n_c ≫ 10⁴. Plateau at ~0.966 across loss
  changes — bottleneck is **MLP capacity vs LGBM's implicit 10⁶-
  param axis-aligned ensemble on rule-threshold features**, not loss
  function. Balanced Softmax did verify cleanly as the training-time
  equivalent of post-hoc log-bias (residual bias `{+0.3, 0, 0}` vs
  plain-CE's `{+1.33, +1.57, +3.40}`) — keep as a pattern. Artefacts:
  `scripts/mlp_{dgp,balsoft,ldam}.py`,
  `scripts/artifacts/oof_mlp_{dgp,balsoft}.npy`,
  `submissions/submission_mlp_{dgp,balsoft}_tuned.csv` on branch
  `claude/improve-balanced-accuracy-v1UtX`. Revisit only if a much
  larger/deeper architecture (FT-Transformer, NumEmb + wide MLP) is
  tried.
- **MLP+BalSoft × LGBM+DGP prob-level blend (2026-04-21).**
  `scripts/blend_lgbm_mlp.py` — arithmetic + geometric sweep over
  `w ∈ [0, 0.5]` with coord-ascent bias retune per weight. Best:
  geometric w=0.15 → 0.97276 tuned OOF vs LGBM alone 0.97271
  (Δ = +0.00005, well below fold std ≈ 0.00068). Confusion diff:
  +44 High recalled, +459 Medium→High mistakes. The MLP's errors
  are **not orthogonal** to LGBM's — both miss the same boundary-
  band flips. Combined with the earlier MNLogit and balanced-
  ensemble blend nulls, this establishes the pattern: any second
  model that's a weaker approximator on the same feature set will
  share LGBM's errors and not contribute in a blend. Rule added:
  demonstrate per-row error orthogonality (Jaccard over OOF error
  sets, or equivalently McNemar test) *before* running a full blend
  sweep; standalone OOF ≥ 0.965 is necessary but not sufficient.
- **Hinge-loss / max-margin tie-breaker over integer separating rules
  (2026-04-21).** Triggered by community discussion
  [692754](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/692754)
  showing the 10k original is linearly separable in a 9-binary-feature
  space (`Soil<25, Temp>30, Rain<300, Wind>10, Mulching=Yes,
  Crop=Flowering/Harvest/Sowing/Vegetative`) with many integer
  solutions differing in hinge loss. `scripts/enumerate_integer_models.py`
  reproduces the OR-Tools CP search; 743 distinct integer models with
  `|w|≤10, 1≤θ≤10` all achieve 100 % on the 10k, hinge-loss range
  0.0000 → 0.2981. **All 743 produce identical predictions on the
  630k synthetic** — agreement rate 1.0000 across top-50, bal_acc
  0.96097, raw_acc 0.98364 for every solution. The 2⁵ × 4 = 128
  discrete cells are fully labeled by the 10k, so any separating
  linear classifier is forced to the same cell-labeling; wider margin
  (`(w, θ) → (2w, 2θ)`) is pure scale and doesn't move any cell
  across the boundary. Max-margin / VC-bound extrapolation argument
  doesn't give a usable tie-breaker in discrete-feature regimes where
  every test row maps to a training cell. Ceiling for any linear rule
  in this representation = 0.96097. Artefacts:
  `scripts/artifacts/integer_separating_models.csv`,
  `integer_models_topk_{pred_syn,pred_test,ids}.npy`,
  `integer_models_summary.json`. Adjacent rule: **scale/shift
  ambiguities inside a single model family are not diversity** — do
  not ensemble over them.

## 6. Open questions

- Public LB calibration: is 0.98114 already tuned-log-bias, or raw
  argmax? Sending one submission would resolve this.
- Does the original Irrigation Prediction dataset improve CV?
- How much of the 0.00105 gap top↔pack is systematic vs noise? Our
  fold std is ~0.002 — the gap is within one seed's worth of variance.
- Is there an ordinal structure lurking in the synthetic DGP that an
  ordinal loss would exploit?

## 7. Original-dataset DGP — closed-form, 6 features, 100%

Reverse-engineered the generator of `data/irrigation_prediction.csv`
(all 10,000 rows, no exceptions). Code: `scripts/dgp_formula.py`.

Six indicators:

| Indicator | Definition                                         |
|---        |---                                                 |
| `dry`     | `Soil_Moisture < 25`                               |
| `norain`  | `Rainfall_mm   < 300`                              |
| `hot`     | `Temperature_C > 30`                               |
| `windy`   | `Wind_Speed_kmh > 10`                              |
| `nomulch` | `Mulching_Used == "No"`                            |
| `Kc`      | `2` if `Crop_Growth_Stage ∈ {Flowering, Vegetative}` else `0` |

Weighted water-need score:

```
score = 2·(dry + norain) + (hot + windy + nomulch) + Kc
```

Binning:

```
Low     if score ≤ 3
Medium  if 4 ≤ score ≤ 6
High    if score ≥ 7
```

How it was found

- RF feature importance on the 10k original collapses to 6 dominant
  features with a sharp cliff (Mulching_Used 0.087 → Humidity 0.021).
- An unconstrained DT on all 19 features reaches 100% train accuracy
  with only 66 leaves at depth 11; on the 6-feature subset, same 66
  leaves / depth 11 / 100%.
- Split thresholds cluster on round numbers: 25 (Soil_Moisture), ~300
  (Rainfall_mm), ~30 (Temperature_C), ~10 (Wind_Speed_kmh).
- Applying those four thresholds plus the two categoricals yields a
  2⁵ × 4 = 128-cell lookup table; **every cell is pure** (0 of 128
  mixed-label cells).
- Inspecting the pure table shows water-supply axes (`dry`, `norain`)
  carry 2× the weight of demand axes (`hot`, `windy`, `nomulch`), and
  crop stage acts as a +2 bump when the crop is actively transpiring.

Implications

- The original dataset is **fully deterministic** on 6 features — it
  is NOT a noisy physical simulation, it is an integer rule written
  by the host. That closes §6's "does the original improve CV" for
  a different reason than we thought: the original is a clean
  target, but its rule is so simple that any competent tree (or even
  a lookup table) reproduces it perfectly, so adding it contributes
  information only where it disagrees with the synthetic DGP.
- **Synthetic train/test likely uses the same or a near-identical
  rule**, given the earlier transfer-check finding (8k-original →
  630k-synthetic tuned bal_acc 0.96278, and categorical vocab +
  numeric distributions align within ~1%). The ~3.7% that doesn't
  transfer is probably label noise injected by the synthetic
  generator, or a slight perturbation of the thresholds.
- **Concrete next bet**: score the synthetic train with this formula
  (pending a new `data/train.csv` download) and measure exact
  per-row agreement. If it's near-perfect, the pack at 0.98114 is
  almost certainly running this rule (or an equivalent one) and the
  remaining gap is entirely label noise. If it's, say, 80%, then
  either thresholds were shifted or weights were tweaked, and a
  small grid search over rule parameters should recover the
  synthetic version.

## 8. DGP is NN-generated, not rule + noise (2026-04-21)

The rule matches 630k synthetic with raw acc 0.98364 (10,304 flips).
We initially modeled the flips as a near-threshold label-flip noise
process, but `brief.md:74` (host states labels come from a deep
learning model trained on the 10k original) plus the 2026-04-21 EDA
force a different interpretation.

### Evidence the "flips" are deterministic

1. **Zero exact feature-vector duplicates** in 630,000 rows. A
   rule + Bernoulli-flip DGP would naturally produce duplicate rows
   (it only has finitely many continuous values in each synthetic
   sample). Continuous-feature generators (VAE / diffusion) produce
   unique rows — which is what we see.
2. **Non-rule features differ significantly between flipped and
   non-flipped rows at score=3** (4,899 flips / 102,157 rows,
   t-test on mean difference):

   | feature                 | d     | mean flipped | mean non-flipped | p       |
   |---                      |---:   |---:          |---:              |---:     |
   | Previous_Irrigation_mm  | +0.107 |  64.87      |  61.26           | 5e-14   |
   | Humidity                | +0.076 |  62.05      |  60.57           | 8e-8    |
   | Electrical_Conductivity | +0.037 |   1.77      |   1.74           | 1e-2    |
   | Field_Area_hectare      | +0.035 |   7.60      |   7.46           | 2e-2    |
   | Soil_pH                 | −0.013 |   6.47      |   6.48           | n.s.    |
   | Organic_Carbon          | −0.008 |   0.92      |   0.92           | n.s.    |
   | Sunlight_Hours          | −0.004 |   7.52      |   7.53           | n.s.    |

   A Bernoulli-flip noise process gives d ≈ 0 on every non-rule
   feature. Instead, rows that "flipped" to Medium have systematically
   higher Humidity and Previous_Irrigation_mm — an agronomically
   plausible shift consistent with a NN that absorbed subtle
   correlations from the 10k original during training.

3. **Per-cell majority on the 64 rule-cells gives raw 0.98384 /
   bal 0.95983** — essentially identical to the rule. Only 1 cell
   has a synthetic majority different from the rule's assignment
   (308 rows, 0.05%). So the "noise" is not cell-level flipping;
   it is within-cell variation driven by continuous position and
   non-rule features.

4. **Flips are always to the adjacent class**, confirmed by the
   per-score breakdown: score=3 flips to Medium (never High),
   score=6 flips to High (never Low), score=7 flips to Medium
   (never Low), etc. The DGP's decision boundary is smooth and
   local, exactly like a NN's.

### Properties deducible from "labels come from a NN"

- Labels are a **deterministic** function of the feature vector
  (`argmax(NN(x))`). No stochastic process. No irreducible error.
- Theoretical ceiling is 100 %. LB leader at 0.98219 implies nobody
  has fully recovered the generator yet, but there is no "noise
  floor" argument against trying.
- **Axis-aligned trees are structurally handicapped.** A NN's
  decision boundary is a smooth curved manifold in the full feature
  space. LGBM needs O(many) axis-aligned splits to approximate
  each NN neuron's contribution, and tree regularization prunes
  small-effect-size signals (d=0.1) that the NN kept.
- Non-rule features (Previous_Irrigation_mm, Humidity,
  Electrical_Conductivity, Field_Area_hectare) carry deterministic
  signal. They are inputs to the current LGBM+DGP but the model
  has not fully integrated them into boundary decisions.
- **The pack at 0.98114 is almost certainly reproducing the NN
  (via FE that captures NN-friendly interactions, or an actual DL
  model), not denoising a stochastic process.**

### Implications for strategy

- **Reframe**: "how do I denoise labels?" → "how do I approximate
  the label-generating NN?" Different question, different toolbox.
- **MLP promoted to top open bet** (see NEXT_STEPS §5). Structural
  match to the DGP.
- **Pairwise FE of rule × non-rule features** (Humidity × Soil_Moisture,
  Prev_Irrigation × Rainfall_mm, Field_Area × dgp_score) may let
  LGBM capture NN-learned interactions without the full setup cost
  of a neural model. See NEXT_STEPS §6.
- LGBM+DGP's 0.97271 is a strong tree baseline but not the ceiling.
  The remaining 0.01 gap to the pack is recoverable signal, not
  irreducible noise.
