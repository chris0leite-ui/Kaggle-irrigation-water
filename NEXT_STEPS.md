# Next steps

**Current best**: LGBM-dist 5-seed bag ⊗ XGBoost-dist log-blend α=0.45
→ OOF **0.97327**, **LB public 0.97170**. Pack 0.98114 (+0.00944),
leader 0.98219 (+0.01049). LB budget: 4 subs spent cumulative, 6 left
today.

## Calibration ladder (OOF → LB)

| Model | OOF | LB | Gap |
|---|---|---|---|
| Baseline LGBM tuned | 0.97097 | 0.96972 | −0.00125 |
| LGBM+DGP tuned | 0.97271 | 0.97137 | −0.00134 |
| **Bag × XGB blend** | **0.97327** | **0.97170** | −0.00157 |

Gap is growing ~+0.00032/tier (selection overfit). Discount predicted
LB by ~0.0015 above OOF 0.972.

## Open bets (ranked by ROI / effort)

### 2026-04-21 brainstorm — fresh angles (next batch)

Grouped by lever. Each bet sized to ≤45 min of compute unless noted.

**High-class lever** (3× leverage under balanced accuracy):

1. **Binary "is High?" head + geo-mean merge.** Dedicated XGB binary
   classifier on all 630 k. Blend its `P(High|x)` with hybrid's High
   posterior (geo-mean or additive). Expected **+0.0005–0.002** if
   High is under-modeled relative to Low/Medium.

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
