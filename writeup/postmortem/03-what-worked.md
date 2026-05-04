# 03 — What worked

Five mechanisms moved the LB. Each is reusable on similar comps.

## 1. DGP reverse-engineering

The synthetic label was generated from a closed-form integer rule on
6 features. We found it by:

- Inspecting the original 10k irrigation dataset (the basis the host
  used to train their NN).
- Brute-forcing thresholds on candidate features.
- Confirming on the 630k synthetic train (98.4% rule-match).

```python
dry     = (Soil_Moisture < 25)
norain  = (Rainfall_mm   < 300)
hot     = (Temperature_C > 30)
windy   = (Wind_Speed_kmh > 10)
nomulch = (Mulching_Used == "No")
Kc      = 2 if Crop_Growth_Stage in {Flowering, Vegetative} else 0
score   = 2*(dry + norain) + (hot + windy + nomulch) + Kc
→ Low if score <= 3 ; Medium if 4 <= score <= 6 ; High if score >= 7
```

Implementation: `scripts/dgp_formula.py`.

**Lesson**: research the problem domain *as a hypothesis seeder*
before training anything. We deleted physics-faithful FE once the
DGP was rule-shaped; but the agronomy primer pointed us at the rule.

## 2. The recipe (`recipe_full_te`)

A single 5-fold pipeline that strung together:

- Target encoding with leave-one-out / smoothing on the categorical-
  rich subset.
- Multi-seed bagging (5 seeds × 5 folds) for variance reduction.
- A routed/specialist split for the rare High class.
- LGBM tuned via prior-reweight + log-bias.

OOF 0.97939, LB 0.97939 (gap +0.00028). This was the workhorse;
every later mechanism stacked on top of it.

## 3. 14-bank natural-cal meta family

A bank of 14 calibrated probability vectors from different model
families (LGBM/XGB/CatBoost/RF) under different reweighting schemes,
all aligned to a "natural" calibration target. Used as inputs to:

- Sklearn RandomForest meta-stacker (`v1 RF natural standalone`)
  → LB 0.98129.
- Selective-override decision rules (Idea 4b, see below).

The 14 components were chosen for *error orthogonality*, not OOF
score. We verified pairwise Jaccard < 0.85 on the OOF error set
before committing the bank.

## 4. The override mechanism (Idea 4b → LB 0.98150)

Selective per-row flips on top of the LB-best primary, gated by
multi-rule consensus:

> Flip primary's prediction at row i iff:
> bagged_v1'_pred[i] ≠ B_pred[i]
> AND raw_pred[i] == tier1b_pred[i] (other-pair unanimous)
> AND 14-bank-majority agrees with the flip direction.

108 selective flips total: 105 H→M, 2 L→M, 1 M→L. Implementation:
`scripts/T1_compose_override.py` family + `build_l1_override_*`.

**Why this worked**: the override is *not* a probabilistic blend.
It's a hand-coded decision rule that fires only when N independent
signals all agree. This decorrelates the override decision from the
saturated stacking bank — the same reason +6bp showed up here while
the 14-bank meta-stacker family had stopped lifting.

## 5. The 4-gate leakage filter

By 04-27 we had 7 leakage incidents costing ~0.0045 LB. Every
candidate now passes four gates before it gets an LB probe:

1. **G1 — Standalone OOF**: candidate alone clears the prior LB-best
   anchor at the recipe-bias operating point.
2. **G2 — Blend lift**: candidate + anchor blend at α* > anchor.
3. **G3 — Net rare-class flip ratio**: ratio of correct-direction
   flips on the High class ≥ 0.5.
4. **G4 — Direction asymmetry**: more correct flips than incorrect
   in the rare-class direction (asymmetric, not just net).

Plus: a **minimal-input meta test** — train the candidate meta with
ONLY 2 components (anchor + new). If the 2-component OOF lands below
anchor, the N-component lift was cross-component memorization, not
orthogonal signal. Don't deploy.

Implementation in `LEARNINGS.md` § Leakage. Subsequent stacking
candidates that *passed* the 4-gate filter consistently held their
OOF→LB gap. Candidates that *failed* G4 reshuffle but were probed
anyway always regressed by 1×–3× their predicted carryover.

## Honourable mentions

- **OOF-honest GroupKFold sanity check**: 30-second test that catches
  fold-leak bugs in pipeline plumbing.
- **`scripts/lb_status.py`**: parses Kaggle CLI output into a queryable
  list of submitted CSVs with scores. Used as the single source of
  truth for "what has been probed" — eliminates the
  re-recommend-already-tested failure mode.
- **Atomic submission builders**: every `build_*_submission.py` script
  produces exactly one CSV from a deterministic recipe. Easy to diff
  two builders to understand a candidate.
