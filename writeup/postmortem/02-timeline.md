# 02 — Timeline

10 days, 109 commits on `main`. Phases below are anchored to the
git log.

## Phase 1 — Kickoff & calibration (2026-04-20)

- First submission of a tuned LGBM baseline: OOF 0.97097 → LB
  **0.96972** (rank 726 / 2357). The tight 0.00125 gap confirmed
  CV is well-calibrated and future deltas are trustable from OOF.
- Read-out: the LB tied pack at 0.98114 is *not* running raw argmax.
  They have structural advantages (FE / external data / seed bagging
  / log-bias tuning).

## Phase 2 — DGP archaeology (2026-04-20 to 04-21)

- Reverse-engineered the closed-form rule on 6 features
  (`scripts/dgp_formula.py`). 98.4% of synthetic rows match the rule;
  1.6% are NN-flipped to a neighbouring class.
- Added `LGBM+DGP` (15 DGP-derived columns + distances) to the
  stack: OOF 0.97271, +0.00021 over plain LGBM.
- Decision: physics-faithful FE was deleted (`benchmark_fe.py` showed
  Δ = −0.00052). The trees discover the rule's interactions on their
  own; hand FE adds noise.

## Phase 3 — Recipe ladder (2026-04-21 to 04-25)

LB climb keyed to recipe components:

```
0.97097  baseline LGBM tuned
0.97296  3-way log-blend (0.45 hybrid + 0.40 routed + 0.15 spec_678)
0.97352  greedy + XGB(non-rule features only) @ α=0.15
0.97468  recipe_full_te (target encoding + recipe)
0.97581  + multi-seed bagging
0.97939  recipe production with full pipeline
```

The lift came from target encoding on the categorical-rich subset,
plus seed bagging, plus a routed/specialist split for the rare High
class. Each step was OOF-honest by 5-fold StratifiedKFold and
calibrated via the first-submission gap.

## Phase 4 — Stacking era (2026-04-25 to 04-27)

- 3-stack meta: OOF 0.98061 → LB 0.98008.
- 4-stack `tier1b_greedy_meta`: OOF 0.98084 → LB **0.98094**.
- First leakage warnings: stacking-inflation ceiling. Multiple metas
  hit OOF 0.98030 but LB ~0.97995. The gap diagnosed as cross-
  component memorization on a saturated bank.
- The 4-gate filter was developed this week (see
  `LEARNINGS.md` § Leakage). All four gates must pass before LB-probe:
  G1 standalone OOF, G2 blend lift, G3 net-rare-class-flip ratio,
  G4 direction-asymmetry.

## Phase 5 — NN expedition (parallel, 2026-04-22 to 04-29)

- 18 NN architectures tested: TabPFN-10k, RealMLP n_ens={1,2,4},
  FT-Transformer, KAN, Mamba, Trompt, TabM, ExcelFormer, narrow
  sklearn MLPs, etc.
- All ran null on the blend gate. The plateau was not a capacity
  limit (see `LEARNINGS.md`: NN-as-structural-match-to-DGP collapse).
- One GPU kernel (RealMLP via pytabkit) ate 3h 34min of CPU
  preprocessing before training and was killed without producing
  output. This generated the **GPU 1-hour cap** rule in CLAUDE.md.

## Phase 6 — Override era (2026-04-29 to 04-30)

The breakthrough phase. Stacking saturated; lift came from a
different mechanism — selective per-row overrides on top of the
LB-best primary, gated by 14-bank-majority consensus.

```
0.98094  4-stack (Phase 4 ceiling)
0.98129  v1 RF natural standalone (sklearn RF meta-stacker)
0.98134  k=4 unanimous override
0.98140  2-OTHER raw+tier1b k=2 unanimous (became "B")
0.98150  Idea 4b triple-consensus (LB-BEST, PRIMARY)
```

Idea 4b: 108 selective flips (105 H→M, 2 L→M, 1 M→L), gated by
"bagged_v1' disagrees with B + {raw, tier1b} unanimous + 14-bank
majority all agree".

## Phase 7 — Saturation thrashing (2026-04-30, deadline day)

Final-day ideas all regressed or held:

```
Idea 5 anchor-switch                  0.98148   −0.00002
98150 minus 176 L→M flips             0.98148   −0.00002
4b + W5(M→H) + strict90               0.98143   −0.00007
T6 directional compose                0.98121   −0.00029   (40th sat.)
L5c K=100 union                       LB regress −0.00145  (47th sat.)
bagginglr_natural standalone          0.98106   −0.00044   (48th sat.)
```

48 independent saturation confirmations at LB 0.98150. The day spent
testing whether the structural ceiling was real (it was, modulo
mechanisms not yet tried).

## Phase 8 — Comp closes

Final selection: PRIMARY = 0.98150 (`submission_idea4b_*`),
HEDGE = 0.98129 (`submission_sklearn_rf_meta_natural_v1_*`). PI's
call, not agent's.
