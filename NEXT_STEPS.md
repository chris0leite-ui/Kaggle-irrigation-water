# Next steps

**Current LB best**: `submission_greedy_nonrule_blend.csv`
(greedy 3-way + non-rule-features-only XGB log-blend α=0.15, fixed
greedy bias) → OOF **0.97421** / **LB 0.97352** (gap 0.00069 —
shrunk from greedy's 0.00079, confirming honest architectural
signal). Pack 0.98114 (+0.00762 above LB-best), leader 0.98219
(+0.00867). LB budget: **7/10 used cumulative**, 3 remaining for
today's session (1 burned on 2026-04-22 seed-bag null).

**Completed this session (2026-04-22)**:
- `scripts/catboost_optuna.py` — **DONE, null**. Phase 1 + Phase 2
  on full 630k: OOF tuned **0.97179** (below LGBM-dist 0.97266,
  below XGB-dist 0.97304). Best HPs: depth=5, lr=0.067, l2=1.08,
  rs=2.74. Jaccard vs LB-best = 0.7376; fixed-bias blend peak
  α=0.05 → +0.00005 (non-signal). CatBoost lever closed.

**Completed this session (2026-04-22) — both swings closed**:
- `scripts/lgbm_competitor_baseline.py` — **DONE, null**. OOF
  tuned **0.97195** vs claim 0.97943 (**−0.00748**, not
  reproducible). Argmax per fold is real (~0.969, +0.008 over
  our LGBM-dist), but digit FE + multiclass TE + inverse-freq
  sample_weight probs are near-uniform, so log-bias tuning only
  adds +0.003 vs our usual +0.009. Most probable cause of the
  0.97943 claim: leaky CV (TE fit on full train before CV loop,
  val-fold labels leak into its own encoding). Fixed-bias blend
  peak α=0.05 → +0.00016 (non-signal, below fold-std 0.00037).
  Lever closed.

## Path to +0.010 LB (target LB 0.9835)

**Gap math**: +0.00762 to the pack, +0.00867 to the leader. The
"+0.0005 insurance" category (seed-bag, variance reduction) is
exhausted — compounding it gives +0.001 at best. **Reaching +0.010
requires at least one swing-sized win, not ten small ones.**

### A. Swing bets — one of these has to hit

1. ~~**Competitor LGBM reproduction (digit FE + multiclass TE)**~~
   **(DONE, null 2026-04-22)**. OOF tuned 0.97195 vs claim 0.97943
   (−0.00748, not reproducible under our honest 5-fold shuffle=True
   seed=42 protocol). Most probable cause of the claim: leaky CV
   (fit TE once on full train, then CV on encoded features → val-
   fold labels leak into their own encoding). Fixed-bias blend
   peak α=0.05 → +0.00016 (non-signal). Moved to ruled-out. The
   2026-04-21 "0.98114 pack = public-CSV blending" conclusion
   stands — it is NOT this recipe + seed-bag.
2. ~~**Serious CatBoost with native Ordered TS**~~ **(DONE, null
   2026-04-22)**. Phase 2 full-630k OOF tuned 0.97179 (below
   LGBM-dist 0.97266 by 0.00087). Best HPs: depth=5, lr=0.067,
   l2=1.08, rs=2.74 — plateau not ridge (top 5 trials 0.968–0.969).
   Jaccard with LB-best 0.7376 (diverse), but fixed-bias blend peak
   α=0.05 → +0.00005 (non-signal). Moved to ruled-out.
3. **Public-notebook CSV ensemble** (the pack's actual recipe,
   IF competitor reproduction plateaus). The 0.98114 pack pulls
   other competitors' submission CSVs as Kaggle Dataset inputs
   and prob-averages them (confirmed 2026-04-21 rival-analysis).
   Blending 3–5 top public CSVs IS the pack's lever.
   **Expected: closes most of the +0.008 gap** because we'd be
   applying the documented method. Ethical question — confirm
   with user before pulling external CSVs. ~30 min.
4. **DGP NN reversal.** Fit an MLP or transformer directly to
   mimic the label-generation function on the 10 k original + the
   synthetic flip patterns, then predict test directly (bypassing
   any ensemble). **Expected: +0.005 to +0.020 if it clicks, 0 if
   not learnable at available capacity.** ~4 h, high variance.

### B. Compounding bets — +0.001–0.003 combined if several land

4. **Pseudo-labeling v2** on the stronger greedy+nonrule base,
   τ=0.99, class-stratified (Low only — the class with best
   calibration; skip Medium/High where high-confidence errors
   exist per the 2026-04-21 residuals EDA). Previous τ=0.95
   attempt on the weaker base compounded boundary errors.
   Expected +0.0002 to +0.001. ~40 min.
5. **Error-analysis → targeted specialist.** Sample 200 rows
   from each off-diagonal cell of greedy+nonrule's confusion
   matrix; identify per-feature patterns; build a specialist on
   the 1–2 biggest error clusters. Expected +0.0002 to +0.001.
   ~1 h.
6. **Self-distillation**: train a fresh XGB-distill to match
   greedy+nonrule's output probabilities on all 630 k rows,
   then log-blend. Forces a new inductive path toward the
   ensemble consensus. Expected +0.0001 to +0.0005. ~40 min.
7. **Rule × non-rule pairwise FE on the greedy base** (ruled
   out previously on `hybrid_lgbmxgb_blend`; untested on
   greedy). Fixed-bias sweep. Expected +0.0001 to +0.0005. ~30
   min.
8. **Ordinal-aware loss** (CORN / Frank-Hall decomposition) for
   Medium↔High. Structural match to the "flips are always
   adjacent-class" finding. Expected +0.0001 to +0.0005. ~45 min.
9. **Seed-bag XGB-nonrule** (5 seeds, ~20 min). Cheapest
   insurance on the only architecturally-diverse leg. Expected
   +0.00005–0.0002 LB. Default fallback if nothing else is
   running.

### Recommended order (post-2026-04-22 null sweep)

Both own-pipeline swings closed this session. #A1 (competitor
reproduction) null, #A2 (CatBoost proper) null. Remaining options
in descending expected value:

1. **#A3 (public-CSV blend)** — the only remaining path to +0.008
   LB. The pack's documented mechanism. Requires user authorization
   to pull public-notebook submission CSVs. ~30 min.
2. **Compounding bets (B.4, B.6, B.7, B.8)** — pseudo-label v2,
   self-distillation, rule × non-rule FE retry, ordinal loss.
   Each is ≤ +0.0005 LB standalone; combined +0.001-0.002 if
   several land. Fall-back if #A3 is off-limits. ~3 h total.
3. **#A4 (DGP NN reversal)** — fit MLP/transformer directly to
   label-gen function on 10k original + synthetic flips. High
   variance. ~4 h on GPU. Only if #A3 is off-limits and all
   compounding bets plateau.

Methodology: fixed-greedy-bias sweep first; LB-probe only if
fixed-bias OOF lifts ≥ +0.0003.

Methodology: every non-swing follow-up uses the **fixed-greedy-bias
sweep first**; only LB-probe if fixed-bias OOF lifts ≥ +0.0003.

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

## Ruled out this competition

Full reasoning in `CLAUDE.md` session log. Short list:

- **Binary "is High?" head** (both hybrid + greedy stacks) — LB overfit,
  then monotonic-negative with fixed bias.
- **Rank-sum / Borda blends** — strictly dominated by prob/log space.
- **LGBM variant of XGB-nonrule** — tracks to 3 decimals, zero diversity.
- **EBM variant of XGB-nonrule** — fold-1 parity with XGB, aborted.
- **Feature-subset bagging on top-7 non-rule** — subsets overlap,
  ensemble below XGB-full.
- **Nonrule + rule_pred + dgp_score** — loses orthogonality that
  made the rule-free version work.
- **Shift-target framings (vanilla + weighted)** — 98% majority
  class collapses the loss.
- **Pairwise FE on hybrid_lgbmxgb_blend** — optimal blend weight
  collapsed from α=0.45 to α=0.05 (retry on greedy still open).
- **CatBoost-dist** — 0.97128 standalone, 3-way blend hurt −0.00007.
- **CatBoost-optuna (2026-04-22)** — OOF 0.97179 after proper Optuna
  HP sweep + raw 8 cats via `cat_features=` + minimal DGP feat set.
  Still below LGBM-dist/XGB-dist. Jaccard 0.7376 with LB-best but
  blend peak α=0.05 → +0.00005 (non-signal). Lever exhausted.
- **Competitor LGBM (digit FE + multiclass TE, 2026-04-22)** — OOF
  tuned 0.97195, Δ vs claim 0.97943 = −0.00748 (not reproducible
  at honest CV). Sample-weight training + 288 TE cols produced
  near-uniform probs, required bias [+2.23, +2.07, +3.20] vs our
  usual [+0.13, +0.57, +3.40]. Jaccard 0.6622 with LB-best
  (genuinely diverse) but blend peak α=0.05 → +0.00016 — below
  fold-std 0.00037 and LB-probe threshold 0.0003. Lever exhausted.
  Sharpest possible reframe: if even digit FE + multiclass TE +
  sample-weight training doesn't clear our pipeline, no own-model
  recipe will. The +0.008 gap to the pack is CSV-blend shaped.
- **Per-cell logistic** — rule-cell information saturated at 0.963.
- **Hinge-loss / max-margin tie-breaker over 743 integer rules** —
  all produce identical synthetic predictions.
- **Balanced-ensemble wrappers (BRF, EasyEnsemble, RUSBoost)** — same
  operating point as post-hoc log-bias via a different mechanism.
- **50k-param MLP (CE / BalSoft / LDAM)** — plateaus at 0.966.
- **Large-capacity tabular NN (5 MLP variants, 2026-04-22)** —
  v5 full 1M / v6 nonrule 150k / v7 top-3 numerics 15k / v8
  specialist {6,7,8} 200k / v9 training-data-routed 1M all null
  standalone + blend-null vs greedy and greedy+nonrule. NN lever
  closed as a capacity-or-optimizer problem; if anything remains
  it's tree-distilled features (leaf embeddings), not more NN
  capacity.
- **Pseudo-labeling τ=0.95 on weaker hybrid base** — compounded
  boundary errors.
- **LGBM HP refresh (Optuna 47 trials)** — +0.001 on 200k proxy,
  plateau on full 630k.
- **Seed-bag greedy (2 seeds, 2026-04-22)** — OOF +0.00010, LB
  −0.00012. Below-1-fold-std lifts from near-deterministic bags
  are non-signal.
- **Spec-{3} (2026-04-22)** — 95/5/0 class mix below the 20–80%
  specialist threshold; hybrid override null.

## Final-submission candidates (pick 2 at competition close)

Primary (keep): **`submission_greedy_nonrule_blend.csv`** — LB 0.97352,
OOF→LB gap 0.00069 (narrowest of any submission). Proven.

Safe fallback: **`submission_xgb_hybrid_v3_routed012_spec678.csv`** —
LB 0.97271, minimal tuning, clean pipeline. Hedge against any
overfitting on the greedy+nonrule blend.

If a genuinely stronger model lands before deadline (from A.1–A.3
above), swap the fallback for it.

See `REPORT.md` §4.1 for the full ranked plan with per-experiment
deltas; this file is the action-oriented short list.
