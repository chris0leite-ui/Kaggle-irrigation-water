# Next steps

## 📌 Current status (2026-04-29 11:18 UTC)

- **LB best**: `submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv`
  → **LB 0.98129** (sklearn RF meta-stacker, 7-component natural-cal bank,
  bootstrap=True, class_weight=None). v1 is preserved as `_v1_lb98129`
  backup; the standard `submission_sklearn_rf_meta_natural_standalone.csv`
  was overwritten by v2 (LB 0.98098 regression — see `2026-04-29` CLAUDE.md
  entry).
- **Hedge**: `submission_rawashishsin_2600_standalone.csv` → **LB 0.98109**
  (single XGB + sklearn TargetEncoder(cv=5), naturally-calibrated).
  Different model class from primary → orthogonal failure modes.
- **Pack**: 0.98148 (+0.00019 above primary, public-CSV blend — banned).
- **Leader**: 0.98219 (+0.00090 above primary).
- **Deadline**: 2026-04-30 (1 day).

### Final-selection lock recommendation (CONFIRM ON KAGGLE UI)

| slot | submission | LB |
|---|---|---:|
| **Primary** | `submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv` | **0.98129** |
| **Hedge** | `submission_rawashishsin_2600_standalone.csv` | **0.98109** |

Different model classes, ~620 row diff on test. Orthogonal failure
modes for private-LB protection.

---

## A1 closure (2026-04-29)

Bank-extension v2 (11 components: + Pick 2b CB skte + XGB clone +
LGBM skte + xgb_dist_routed_v3) → **LB 0.98098 (regression -0.00031)**.

Confirmed 3× now (a1lgbm parallel, v2 this branch, plus v1 baseline):
**bank extension on saturated natural-cal RF meta-stacker doesn't
reliably translate OOF lift to LB.** v1's 7-component bank is the
LB sweet spot.

Key insight: more natural calibration ≠ better LB transfer. v2's
drift [0, -0.10, 0] is "more natural" than v1's [-0.10, -0.10, -0.20]
but transfers worse on LB by 0.00031. v1's slight imperfection
fortuitously aligns with LB test distribution.

---

## Tier B — Mechanism-novel L2 architectures on v1 bank (in progress)

**Hypothesis:** different L2 architectures (ExtraTrees, HistGBM) on
the SAME 7-component bank that produced v1 LB 0.98129 may extract
orthogonal signal via different bagging/boosting variance structure.
Cleanest comparison since v1 bank is LB-validated.

### B1 — sklearn ExtraTreesClassifier on v1 bank

- `scripts/sklearn_extratrees_natural_v1bank.py`
- Diff vs RF v1: `bootstrap=False` + random feature thresholds.
  Same n_estimators=500, max_depth=12, class_weight=None.
- Bank: identical 7-component v1 (rawashishsin + cb_natural + cb +
  recipe + realmlp + xgb_corn + xgb_dist_digits).
- SMOKE result: tuned 0.98014, drift [-0.40, 0.00, 0.00] (better
  Med/H drift than RF, slightly worse Low).
- Wall: ~10 min CPU. Status: production running.

### B2 — sklearn HistGradientBoostingClassifier on v1 bank

- `scripts/sklearn_histgbm_natural_v1bank.py`
- Diff vs RF v1: gradient-boosted level-wise tree growth (vs bagged).
  HPs: lr=0.05, max_depth=3, max_iter=1000, l2_reg=0,
  class_weight=None, early_stopping=True.
- Same v1 7-component bank.
- Wall: ~15-20 min CPU. Status: production running.

### Decision rule for B1/B2

Same 4-gate filter as A1:
- G1: standalone OOF Δ ≥ +2e-4 vs RF v1's 0.98063 (LB-validated)
- G2: per-class recall within −5e-4 floor each
- G3: dual-α stability (1.0x to 2.0x linear scaling)
- G4: net rare-class flip > 0 AND |asymmetry| ≥ 0.5

If standalone OOF passes G1 AND PCR drift not catastrophic, candidate
for LB probe (1 slot).

If both pass: **B3 — ensemble RF v1 + B1 + B2 at meta-output level**
(geomean or weighted). Different architectures preserve different
errors → variance reduction without bank inflation.

---

## Tier C — Speculative ensembles (deferred)

If Tier B produces a LB-positive candidate:

### C1 — RF v1 + B-best meta-output ensemble
Geomean of two LB-validated RF natural variants (v1 + best of B1/B2).
Mechanism: variance reduction across DIFFERENT meta architectures, NOT
across more bank components. Untested.

### C2 — Per-row gating between v1 and Tier-B variant
Train small classifier on row features → which meta-stacker is more
trustworthy per row. If RF v1 better on score=3 rows but B1 better on
score=6 rows, per-row routing extracts both. Risk: overfit on routing
signal.

### C3 — Tier-B on EXPANDED v2 bank
If B1/B2 on v1 bank works, optionally test on v2's 11-component bank
to see if the new components help when paired with a different L2
architecture.

If Tier B is null:

### Lock + stop
- Primary v1 (LB 0.98129) + hedge rawashishsin (LB 0.98109) is the
  optimal locked pair.
- Reserve remaining 8 LB submissions for end-of-comp variance check.
- 1 day to deadline (2026-04-30).

---

## Skipped levers (already exhausted)

- **More bank-extension** — 3× confirmed non-transfer (v1, a1lgbm, v2).
- **More NN-family attempts** — 15+ nulls form structural pattern.
- **More meta-stacker variants** (LR-meta, mlp-meta, deeper RF, more
  trees) — saturation confirmed.
- **HP / model-seed bagging** — LB-regressed in past runs.
- **Public-CSV blending** — banned by top-of-file rule.
- **More natural-cal base components** — sklearn TE on V10 recipe FE
  doesn't transfer rawashishsin's calibration property
  (drift +1.30 vs rawashishsin +1.10 on Low).

---

## Reference: full LB ladder (sorted by LB)

| Submission | OOF tuned | LB | Gap | Notes |
|------------|-----------|-----|-----|-------|
| **RF natural v1** (PRIMARY) | 0.98063 | **0.98129** | -0.00066 | 7-component bank, LB-best |
| rawashishsin v3 | 0.98010 | 0.98109 | -0.00099 | HEDGE, single XGB |
| RF natural v2 (A1) | 0.98067 | 0.98098 | +0.00031 | 11-component bank, regression |
| RF natural a1lgbm | 0.98078 | 0.98097 | +0.00019 | 10-component bank, regression |
| LB-best 4-stack | 0.98084 | 0.98094 | -0.00010 | tier1b stack |
| 3-way multi-seed | 0.98029 | 0.98005 | +0.00024 | recipe family |
