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

1. **Rule × non-rule pairwise FE for LGBM-dist AND XGB-dist.**
   `Humidity × Soil_Moisture`, `Previous_Irrigation × Rainfall_mm`,
   `Humidity × Temperature_C` (≈ VPD), `EC × Soil_Moisture`,
   `Field_Area × dgp_score`. Target the 4 non-rule features with d>0.03
   on flips (2026-04-21 EDA). Re-run bag + blend. ~30-line patch.
   Expected **+0.0005–0.002**.

2. **CatBoost-dist as 3rd blend leg.** Model-family diversity compounds
   (LGBM → XGB was +0.00038). Pre-check: Jaccard overlap between
   CatBoost and (LGBM ∪ XGB) OOF errors; skip if >0.8. Expected
   **+0.0002–0.0008** on 3-way blend.

3. **Seed-bag XGB-dist** (3–5 seeds), mirror LGBM bag before the 3-way
   blend. Expected **+0.00015**.

4. **Meta-stack**: feed the 3 component OOF probs (LGBM-bag, XGB,
   CatBoost) into a small LGBM meta-model. Expected **+0.0001–0.0005**.

5. **Within-cell per-cell logistic / small MLP** on 7 non-rule
   continuous features, one model per rule-cell (128 cells × ~5k rows).
   Orthogonal to LGBM-dist and XGB-dist by construction. Expected
   **+0.0005–0.002** if residuals concentrate in a few cells.

6. **LB recalibration sub** on any of the above once OOF gains
   compound. One of 6 remaining LB subs today.

7. **Ordinal-aware loss** for Medium↔High. Untested; lower priority.

## Ruled out this session

- Hinge-loss / max-margin tie-breaker over 743 integer rules: all
  produce identical synthetic predictions (cell-labeling forced by
  CP constraints). See `scripts/enumerate_integer_models.py` and
  `CLAUDE.md` 2026-04-21 entry.

## Suggested immediate action

Run step 1 (pairwise FE) + step 2 (CatBoost leg) in one session. Both
stack on the existing bag + blend pipeline. If step 1 clears +0.001
OOF, submit one LB to recalibrate before step 4/5.

See `REPORT.md` §4 for the full ranked plan with per-experiment
deltas; this file is the action-oriented short list.
