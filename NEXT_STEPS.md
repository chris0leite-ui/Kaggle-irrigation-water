# Next steps

**Current LB best**: `submission_blend_greedy_w045_040_015.csv`
(greedy 3-way log-blend hybrid_v3 0.45 + routed_v3 0.40 + spec_678
0.15) → OOF 0.97375 / **LB 0.97296** (gap 0.00079). Our binhigh
experiment (OOF 0.97398) LB-submitted at 0.97212 — overfit
(gap 0.00186). Pack 0.98114 (+0.00818 above LB-best), leader 0.98219
(+0.00923). LB budget: **5/10 used today**, 5 remaining.

## Calibration ladder (OOF → LB)

| Model | OOF | LB | Gap |
|---|---|---|---|
| Baseline LGBM tuned | 0.97097 | 0.96972 | −0.00125 |
| LGBM+DGP tuned | 0.97271 | 0.97137 | −0.00134 |
| Bag × XGB blend | 0.97327 | 0.97170 | −0.00157 |
| hybrid_v3 (routed {1,2}) | 0.97352 | 0.97224 | −0.00128 |
| hybrid_v3 (routed {0,1,2}) | 0.97352 | 0.97271 | −0.00081 |
| **greedy 3-way log-blend** | **0.97375** | **0.97296** | **−0.00079** |
| hybrid + binhigh logit-add | 0.97398 | 0.97212 | −0.00186 ← **overfit** |

**Selection overfit lesson (2026-04-21):** the binhigh experiment
added +0.00036 OOF but *lost* 0.00084 LB vs the greedy blend. Layering
a tuned component (75-point sweep + log-bias retune) on top of an
already-OOF-tuned stack compounds selection bias ~5.2× (gap blew up
from 0.00079 to 0.00186). **Rule: expect real LB delta ≈ 1/3 of OOF
delta** when stacking tuned blends on tuned baselines. Prefer
architectural levers (new feature sets, orthogonal models) over more
tuning.

## Open bets (ranked by ROI / effort)

### 2026-04-21 brainstorm — fresh angles (next batch)

Grouped by lever. Each bet sized to ≤45 min of compute unless noted.

**High-class lever** (3× leverage under balanced accuracy):

1. ~~**Binary "is High?" head.**~~ **FALSIFIED 2026-04-21.** Two-stage
   test: first `hybrid_lgbmxgb_blend + binhigh (75-point sweep +
   bias retune)` showed +0.00036 OOF / −0.00084 LB (overfit). Second
   test `greedy + binhigh (fixed bias, 9-point sweep)` showed
   monotonic decrease — λ=0.05 → −0.00002, λ=0.30 → −0.00129. The
   binary head (AUC 0.9987) carries information that is already
   fully absorbed by the greedy blend's High column; any additional
   injection + bias-retune manufactures fake OOF lift. **Lever dead.**
   See `scripts/greedy_binhigh_minimal.py` + CLAUDE.md entry.

2. **High-only focal loss.** Custom XGB objective: γ=2 focal on High,
   standard CE on Low/Medium. Targets ~21 k High rows without
   data-starvation (unlike per-class specialist, which failed). ~40 min.

3. **Upweight flip-candidate rows.** Re-train XGB-dist with
   `sample_weight = 1 + 2·P(flip|x)` using the saved AUC-0.90 flip
   detector. Forces capacity onto exactly the rows where High↔Medium
   confusion lives. ~20 min.

**"Sum" / aggregation we haven't tried**:

4. ~~**Rank-sum / Borda blend across saved OOFs.**~~ **FALSIFIED
   2026-04-21.** All rank-avg / rank-wavg / Borda variants landed at
   0.96739–0.96810 on tuned OOF (−0.0055 to −0.0062 below current
   best). Row-softmaxed ranks destroy the absolute-probability signal
   log-bias tuning needs. Full mix sweep rank↔prob monotonically
   worsens with rank weight. Rule: **prob/log-space blends strictly
   dominate rank-space blends** for 3-class log-bias-tuned
   decision rules. See `scripts/rank_blend.py` + CLAUDE.md entry.

5. **Hard-vote over tuned variants.** Majority vote of hybrid_v3,
   LGBM+DGP, XGB-dist, LGBM×XGB blend, spec-{6,7,8}. Captures "all
   models agree" as a confidence signal for threshold tuning. ~15 min.

6. **Sum-of-one-hot submissions → final LGBM meta-stack.** One-hot
   labels from N saved submissions, sum, feed as features to a final
   LGBM alongside raw features. Mimics the pack's CSV-blending trick
   with our own diversity. ~30 min.

**NN-flip hypothesis** (non-DGP features drive label flips):

7. **Non-rule-features-only flip predictor.** Train LGBM/XGB restricted
   to `Humidity, Prev_Irrig, EC, Soil_pH, Organic_C, Sunlight,
   Field_Area, Region, Crop_Type, Soil_Type` predicting either
   `flip direction` or full `y`. Orthogonal by construction to
   LGBM+DGP. ~30 min.

8. **Two-stage rule-base + non-rule-correction.** Rule gives ordinal
   base; XGB predicts signed shift `y − rule ∈ {−2,−1,0,+1,+2}`
   using only non-rule features. Matches the "NN perturbed labels on
   extra features" story. ~45 min.

9. **Ordinal multiclass (CORN / Frank-Hall).** Two binaries:
   `P(y≥Medium)` and `P(y≥High)`. Different gradient direction than
   `multi:softprob`; matches the sum-of-indicators DGP structure.
   ~40 min.

10. **TTA with non-rule feature noise.** Predict test 5× with small
    Gaussian noise on non-rule continuous cols, average. If labels
    are smooth-NN outputs, TTA denoises the surface. ~15 min on saved
    XGB.

**Structural long-shot**:

11. **Tiny additive NN with explicit rule × non-rule crosses.** 2-layer
    MLP, layer 1 forced to `[rule_emb, non_rule_emb, rule⊗non_rule]`.
    Matches brief.md:74's "DL model" note without the generic-MLP
    plateau from branch `v1UtX`. ~2 h.

### Carried over from earlier sessions

1. **Rule × non-rule pairwise FE for LGBM-dist AND XGB-dist.**
   `Humidity × Soil_Moisture`, `Previous_Irrigation × Rainfall_mm`,
   `Humidity × Temperature_C` (≈ VPD), `EC × Soil_Moisture`,
   `Field_Area × dgp_score`. Target the 4 non-rule features with d>0.03
   on flips (2026-04-21 EDA). Re-run bag + blend. ~30-line patch.
   Expected **+0.0005–0.002**.
   **NOTE (2026-04-21)**: falsified on branch
   `claude/improve-balanced-accuracy-v1UtX` — full 8-FE set delivered
   Δ = −0.00007 on blend, optimal blend-weight collapsed from α=0.45
   to α=0.05. See CLAUDE.md "rule × non-rule pairwise FE (null
   result)" entry. **Deprioritize.**

2. **CatBoost-dist as 3rd blend leg.** Model-family diversity compounds
   (LGBM → XGB was +0.00038). Pre-check: Jaccard overlap between
   CatBoost and (LGBM ∪ XGB) OOF errors; skip if >0.8. Expected
   **+0.0002–0.0008** on 3-way blend.
   **NOTE (2026-04-21)**: falsified — CatBoost 0.97128 standalone, 3-way
   blend hurt by 0.00007. Low Jaccard (0.74) was necessary but not
   sufficient — error magnitudes weren't complementary. **Deprioritize.**

3. **Seed-bag XGB-dist** (3–5 seeds), mirror LGBM bag before the 3-way
   blend. Expected **+0.00015**.

4. **Meta-stack**: feed the 3 component OOF probs (LGBM-bag, XGB,
   CatBoost) into a small LGBM meta-model. Expected **+0.0001–0.0005**.

5. **Within-cell per-cell logistic / small MLP** on 7 non-rule
   continuous features, one model per rule-cell (128 cells × ~5k rows).
   **NOTE (2026-04-21)**: LR variant falsified — standalone 0.96280,
   rule ⊗ LR blend 0.96286. Within-cell continuous signal is
   exhausted at linear capacity; MLP on same data unlikely to rescue.
   **Deprioritize.**

6. **LB recalibration sub** on any of the above once OOF gains
   compound. 7 remaining LB subs today.

7. **Ordinal-aware loss** for Medium↔High. Untested; see brainstorm
   #9 above (CORN / Frank-Hall).

## Ruled out this session

- Hinge-loss / max-margin tie-breaker over 743 integer rules: all
  produce identical synthetic predictions (cell-labeling forced by
  CP constraints). See `scripts/enumerate_integer_models.py` and
  `CLAUDE.md` 2026-04-21 entry.

## Suggested immediate action

Brainstorm #4 (rank-sum) falsified 2026-04-21. Next: #1 (binary
"is High?" head) → #7 (non-rule-features-only predictor). One covers
the High-class lever directly, the other covers the NN-flip
hypothesis. ~1.5 h total, one lever each.

See `REPORT.md` §4 for the full ranked plan with per-experiment
deltas; this file is the action-oriented short list.
