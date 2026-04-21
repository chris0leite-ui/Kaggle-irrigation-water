# Next steps

**Current LB best**: `submission_greedy_nonrule_blend.csv`
(greedy 3-way + non-rule-features-only XGB log-blend α=0.15, fixed
greedy bias) → OOF **0.97421** / **LB 0.97352** (gap 0.00069 —
shrunk from greedy's 0.00079, confirming honest architectural
signal). Pack 0.98114 (+0.00762 above LB-best), leader 0.98219
(+0.00867). LB budget: **6/10 used today**, 4 remaining.

## Calibration ladder (OOF → LB)

| Model | OOF | LB | Gap |
|---|---|---|---|
| Baseline LGBM tuned | 0.97097 | 0.96972 | −0.00125 |
| LGBM+DGP tuned | 0.97271 | 0.97137 | −0.00134 |
| Bag × XGB blend | 0.97327 | 0.97170 | −0.00157 |
| hybrid_v3 (routed {1,2}) | 0.97352 | 0.97224 | −0.00128 |
| hybrid_v3 (routed {0,1,2}) | 0.97352 | 0.97271 | −0.00081 |
| greedy 3-way log-blend | 0.97375 | 0.97296 | −0.00079 |
| hybrid + binhigh logit-add | 0.97398 | 0.97212 | −0.00186 ← **overfit** |
| **greedy + nonrule α=0.15** | **0.97421** | **0.97352** | **−0.00069 ← NEW BEST** |

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

7. ~~**Non-rule-features-only flip predictor.**~~ **CONFIRMED
   2026-04-21. NEW LB BEST.** XGB 3-class on 13 non-rule features,
   log-blended into greedy at α=0.15 (fixed greedy bias). Standalone
   bal_acc 0.430 (near-random) but lifts greedy from OOF 0.97375 →
   **0.97421 (+0.00047)**. **LB: 0.97352 (+0.00056 vs greedy
   0.97296)**, OOF→LB gap shrunk to 0.00069. High recall +0.38pp
   via non-rule-driven flips. See `scripts/nonrule_features_only.py`
   + CLAUDE.md entry. Lever ALIVE.

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

Non-rule lever validated (LB 0.97352, +0.00056). Stack follow-ups
ranked cheapest-first:

**2026-04-21 update**: four stacking follow-ups all null. XGB-nonrule-
full on 13 features is the single best expression of the non-rule
lever; LGBM/EBM/feature-subset/weighted-shift all track or underperform
it. Diversity has to come from OUTSIDE "different model on same
non-rule features".

1. **Seed-bag the non-rule model** (5 seeds, ~20 min). Variance
   reduction on the only-architecturally-diverse leg. Expected
   +0.00005–0.0002 on LB. Cheapest remaining insurance.
2. ~~Brainstorm #8 shift-correction.~~ **FALSIFIED.**
3. ~~LGBM variant of nonrule.~~ **NULL 2026-04-21.** Tracks XGB to 3
   decimals on 13 features.
4. ~~EBM variant of nonrule.~~ **ABORTED 2026-04-21.** Fold-1 argmax
   parity with XGB (0.424), 29 min/fold compute not justified.
5. ~~Feature-subset bagging on top-7 non-rule.~~ **NULL 2026-04-21.**
   5 subsets overlap too much; ensemble below XGB-full.
6. **Non-rule + rule_pred as 1 extra feature** — previously flagged as
   risky. Fixed-bias sweep will give honest answer. If lifts, rule-
   aware posterior + non-rule features stacks. Cheap (~15 min).
7. **Pseudo-labeling via LB-best**. τ=0.95 was null earlier on
   hybrid_lgbmxgb_blend base; retry with the stronger greedy+nonrule
   base. ~40 min.
8. **Self-distillation** — train XGB to match LB-best predictions on
   all train rows. ~40 min.
9. **Rule × non-rule pairwise FE applied to greedy base models**
   (ruled out on hybrid_lgbmxgb_blend; worth re-checking on greedy).

Methodology: every follow-up uses the **fixed-greedy-bias sweep
first**; only LB-probe if fixed-bias OOF lifts ≥ +0.0003.

See `REPORT.md` §4 for the full ranked plan with per-experiment
deltas; this file is the action-oriented short list.
