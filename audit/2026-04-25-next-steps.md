# 2026-04-25 — Next steps after audit (without giving up)

**Context**: senior-engineer audit closed F2 (iso-on-full-OOF) GREEN
and confirmed OOF honesty under both Region- and Crop-grouped KFold
(Δ -0.00029 / -0.00056, both inside the 0.002 honesty threshold).
The current primary at LB 0.98094 is **mostly genuine signal**, not
inflation. That means we have honest LB headroom to push for —
0.98114 (pack) and 0.98219 (leader) remain reachable.

5 days to deadline, 7 LB probes left today (10/day fresh tomorrow).
Hedge swap to `submission_3way_recipe025_s1035_s7040.csv` recommended
per audit F1.

This document is the "we're not done" plan: ranked by EV-per-hour,
each item written to be runnable from a one-line invocation.

---

## Tier A — cheap CPU, can run today/tomorrow (no GPU needed)

### A1. τ sweep on stage-1 pseudo (3 × ~50 min)

Stage-1 pseudo at τ=0.98 is the load-bearing component of every
LB-best stack. blamerx's τ=0.92 nulled (boundary contamination).
The team picked 0.98 by instinct and never swept the band between.

Run:
```bash
for TAU in 0.95 0.97 0.99; do
  PSEUDO_TAU=$TAU PSEUDO_SUFFIX="tau$(echo $TAU | tr -d .)" \
    python3 scripts/recipe_pseudolabel.py
done
```

Then for each, build a 2-way blend with recipe at fixed bias and
compare standalone OOF + Jaccard vs the LB-best 2-way. If any τ has:
- standalone tuned OOF > 0.97993 (stage-1 baseline) AND
- errors ≤ 10039 AND
- Jaccard < 0.85 with stage-1

…it's a drop-in upgrade. Add to the meta-stacker bank, re-run greedy,
LB-probe if blend Δ ≥ +0.0002.

Expected upside: +0.00005 to +0.00030 LB. Floor: 0 (null).

### A2. GroupKFold-by-Crop OOF as meta-stacker input (~15 min)

We just generated `oof_b2_groupkfold_crop.npy` (tuned 0.97910). Its
errors are structurally different from StratifiedKFold OOFs (different
fold split → different fold-adapted decisions per row). Add it to the
meta-stacker bank as a candidate component, re-run `tier1b_greedy_with_meta.py`.

Run:
```bash
# Make sure b2_groupkfold_crop is NOT in the EXCLUDE_FROM_POOL list
sed -i 's/"b2_groupkfold_region",.*$/"b2_groupkfold_region",/' \
  scripts/tier1b_greedy_with_meta.py  # no-op safeguard
# Then:
python3 scripts/tier1b_xgb_metastack.py    # bake new OOF into meta features
python3 scripts/tier1b_greedy_with_meta.py
```

If new greedy picks `b2_groupkfold_crop`, the lift is real. If it
doesn't, no harm. **No LB probe unless final OOF > 0.98090.**

### A3. Per-fold iso variants in greedy pool (~15 min, no retraining)

The per-fold-iso experiment chose `xgb_metastack_bag3__iso` α=0.350
(OOF 0.98080) over `xgb_metastack__iso` α=0.300 (current primary,
OOF 0.98084). The choice is noise-level but suggests
`xgb_metastack_bag3` (the 3-seed bagged meta) might be a better
backbone than the single-seed meta in a per-fold-iso pool.

Cross-experiment idea: re-run `tier1b_greedy_with_meta.py` but use
`iso_cal_perfold` from `tier1b_greedy_perfoldiso.py` for the pool
copies. If greedy picks a different combination than the original
primary AND the OOF lifts ≥ +0.0002 over current primary, LB-probe.

### A4. Hedge swap (zero-compute, audit F1)

Manual action on Kaggle: change marked final submission from
`submission_recipe_full_te.csv` → `submission_3way_recipe025_s1035_s7040.csv`.
Premium drops from −0.00155 to −0.00089. Sidesteps meta-stacker layer.

---

## Tier B — Kaggle GPU (queue overnight)

### B1. SMOTE-NC on Kaggle kernel (~3h Kaggle wall)

Smoke green locally (+0.00174 OOF lift over recipe smoke). Production
env-blocked twice locally (container rehydrate). Kaggle has 9h cap and
is rehydrate-immune. The scaffold (`scripts/recipe_smote_high.py` +
`scripts/blend_gate_smote.py`) is committed.

Action: wrap into `kaggle_kernel/kernel_smote/` with inlined deps,
push, pull artifacts back, blend-gate locally. The only training-data-
level lever that targets High recall directly. Strong upside if it
produces orthogonal errors.

Expected upside: +0.0005 to +0.0020 LB if blend gate passes.

### B2. RealMLP n_ens=2 with n_epochs=40 (~50 min Kaggle GPU)

CLAUDE.md log notes n_ens=4 nulled and attributes it to
under-convergence (n_epochs=25 forced by 1h cap). n_ens=2 with full
n_epochs=40 directly tests this: if it beats n_ens=1 in the
LB-best stack, ensembling at full epoch budget helps and we should
push n_ens=4 with longer wall (e.g. local CPU overnight, ~3h).

Expected upside: +0.0001 to +0.0005 LB.

### B3. Trompt push — CLOSED NULL (commit 87726f0 on main)

Sibling session pushed Trompt as a 1-fold full-data probe; result was
**13th NN null**: lowest Jaccard ever (0.53) with the LB-best stack
but errors +169 vs anchor. Magnitude trap defeated the orthogonality.
Don't retry as a full 5-fold run.

Replacement candidate: **TabM (ICLR 2025 BatchEnsemble MLP via
pytorch_frame)** flagged in the parallel session's Tier-1b cross-
pollinate write-up as the only architecturally novel NN family
remaining. Reuses Trompt kernel scaffold with a single-line model swap.
~1h GPU. Same magnitude-trap risk; gate at fold-1 errs ≤ +5% over
LB-best 4-stack.

### B4. OvR-XGB on V10 recipe (~80 min CPU)

From parallel session's kernel audit round 3: include4eto's
OvR-XGB recipe — 3 binary:logistic XGB heads on the FULL V10
feature set, concat → softmax-renormalize → multiplicative class-
weight Optuna (200-trial, bounds [0.5, 3.0]³). Different gradient
than multi-class softmax CE; different boundary geometry. Adds a
genuinely new component to the meta-stacker bank.

Expected upside: +0.00010 to +0.00030 OOF if binary CE produces
materially different boundary geometry. Magnitude-trap rule applies
(Jaccard < 0.80 AND errs ≤ anchor required for blend transfer).

---

## Tier C — speculative but ceiling-breaking

### C1. Per-bin blend at lower-resolution bins (~30 min CPU)

Earlier per-bin sweep (5 bins × 2 weights = 10 free params) overfit
under nested CV. The fix: 3 bins (`{0..2}` easy-Low, `{3..6}` boundary,
`{7..9}` easy-Med-or-High) with 2 weights each = 6 free params on the
same 5-fold split. Lower variance per bin.

Run a fresh `scripts/per_bin_blend_3way.py` (new — to be written) on
the LB-best 3-stack components. If nested-CV OOF ≥ +0.00020 over
current primary, LB-probe.

### C2. 4-component "clean meta-stacker" (~5 min)

The current 30+ component meta-stacker over-parameterizes. A meta with
ONLY {recipe, pseudo_s1, pseudo_s7, RealMLP, nonrule_iso, GroupKFold-crop}
as inputs (6 features × 3 classes = 18 dims) might generalize better
than 30+ component bank. Cleaner architecture; probably less LB-probing
selection-risk.

### C3. Public/private split ratio verification (~5 min)

`brief.md` doesn't explicitly state the ratio. Read the Kaggle
competition page or pull from the API. Affects every variance estimate.
50/50 vs 30/70 changes "true LB" math by ~30%. Cheap diagnostic that
informs every other decision.

---

## What NOT to do (already exhausted, would burn budget)

- HP tuning (LB-regressed twice; structurally null).
- Model-seed bagging (LB-regressed; XGB is near-deterministic at our HPs).
- Cleanlab label-noise interventions (mechanism wrong for deterministic
  flips).
- Retune log-bias on primary (binhigh rule violation).
- More NN-from-scratch MLPs (12 nulls, architectural ceiling on raw
  feature set).
- Public-CSV blending (banned by top-of-CLAUDE.md rule).

---

## Ranked execution plan (concrete, this session + next)

If you have ~2h CPU now:
1. Run **A1** (τ sweep) and **A2** (GroupKFold-Crop in meta) in parallel.
2. While they run, do **A4** (hedge swap on Kaggle, manual).

If overnight Kaggle GPU available:
3. Push **B1** (SMOTE-NC) — highest single EV remaining.
4. Run **B4** (OvR-XGB) on the 16-core box (~80 min CPU); doesn't
   compete with GPU jobs.

If A1/A2 produce a candidate:
4. Single LB probe (≤1 of remaining 7), only if blend OOF Δ ≥ +0.0002
   over current primary.

If everything nulls:
5. Lock primary + hedge as final. We end at LB ≥ 0.98094 (±private
   shake-up). That's top-100-tier in a comp where the pack is at 0.98114
   via public-CSV blending we explicitly chose not to use. **That's a win,
   not a defeat.**

---

## Posture

Audit says primary is honest. OOF passes both leakage-axis checks.
Iso wasn't inflating. We have 5 days, 50 LB probes available across
the rest of the comp, and 3-4 untried levers ranked above.

The right move is to keep pushing the cheapest open levers (A1+A2 today),
queue the GPU-required ones (B1 first), and reserve LB probes for
candidates that pass the +0.0002 blend gate. Lock at the end of
day 5, not before.
