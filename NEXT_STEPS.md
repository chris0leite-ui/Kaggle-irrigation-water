# Next steps

**Current LB best**: `submission_greedy_nonrule_blend.csv`
(greedy 3-way + non-rule-features-only XGB log-blend α=0.15, fixed
greedy bias) → OOF **0.97421** / **LB 0.97352** (gap 0.00069 —
shrunk from greedy's 0.00079, confirming honest architectural
signal). Pack 0.98114 (+0.00762 above LB-best), leader 0.98219
(+0.00867). LB budget: **6/10 used today**, 4 remaining.

## Path to +0.010 LB (target LB 0.9835)

**Gap math**: +0.00762 to the pack, +0.00867 to the leader. The
"+0.0005 insurance" category (seed-bag, variance reduction) is
exhausted — compounding it gives +0.001 at best. **Reaching +0.010
requires at least one swing-sized win, not ten small ones.**

### A. Swing bets — one of these has to hit

1. **Large-capacity tabular NN on full feature set.** DGP is a
   host NN (`brief.md:74` + 2026-04-21 residuals EDA). Prior 50 k-
   param MLP plateaued at 0.966 — capacity-bound, not structurally
   wrong. Try FT-Transformer (1–3M params) or NumEmb + wide MLP
   (~500 k, per-feature learnable embeddings + 3×512 hidden). Pre-
   check after fold 1: error-Jaccard with greedy; ≥0.90 kill,
   <0.85 commit to full run. **Expected: +0.001 to +0.003 LB
   standalone, or 0 if it plateaus like before.** ~2 h on GPU.
2. **Public-notebook CSV ensemble** (the pack's actual recipe).
   The 0.98114 pack pulls other competitors' submission CSVs as
   Kaggle Dataset inputs and prob-averages them (confirmed 2026-
   04-21 rival-analysis). Blending 3–5 top public CSVs IS the
   pack's lever. **Expected: closes most of the +0.008 gap**
   because we'd be applying the documented method, not trying to
   beat it. Ethical question — confirm with user before pulling
   external CSVs. ~30 min.
3. **DGP NN reversal.** Fit an MLP or transformer directly to
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

### Recommended order

1. **Dispatch #1 (large tabular NN) in background** — 2 h, orthogonal
   to everything tried, real upside.
2. **Foreground: #5 (error analysis)** on the current best to
   surface hidden levers. ~30 min.
3. If #1 doesn't produce +0.003: combine **#4 + #6 + #7 + #8** for
   compounding +0.001–0.002.
4. If still stuck at ~0.975 after that: escalate to **#2 (public-
   CSV blend)** as the deliberate last resort that's documented to
   work. Requires user OK.
5. If #2 is off-limits: try **#3 (DGP reversal)** as a high-
   variance swing.

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
- **Per-cell logistic** — rule-cell information saturated at 0.963.
- **Hinge-loss / max-margin tie-breaker over 743 integer rules** —
  all produce identical synthetic predictions.
- **Balanced-ensemble wrappers (BRF, EasyEnsemble, RUSBoost)** — same
  operating point as post-hoc log-bias via a different mechanism.
- **50k-param MLP (CE / BalSoft / LDAM)** — plateaus at 0.966.
- **Pseudo-labeling τ=0.95 on weaker hybrid base** — compounded
  boundary errors.
- **LGBM HP refresh (Optuna 47 trials)** — +0.001 on 200k proxy,
  plateau on full 630k.

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
