#!/usr/bin/env bash
# Option 1 / A1+LGBM pipeline driver.
#
# Sequential phases (each phase exits on failure of its child):
#   1. Pick 2b production  (recipe_catboost_skte.py, ~50 min CPU)
#   2. LGBM-skte production (recipe_lgbm_skte.py,    ~30 min CPU)
#   3. RF natural meta retrain on extended bank
#      (META_SUFFIX=_a1lgbm sklearn_rf_meta_natural.py)
#   4. Standalone CSV emit  (emit_rf_natural_a1lgbm_standalone.py)
#   5. Blend gate analysis  (blend_gate_rf_natural_full.py — TODO: param SUFFIX)
#
# Idempotent: each step skips if its outputs exist.
# Resume-aware: per-fold checkpoints inside production scripts mean
# rehydrate + re-run starts only from the last incomplete fold.
set -e

ART=scripts/artifacts

phase() { echo "===== $1 =====  $(date -u +%H:%M:%S)"; }

phase "Phase 1: Pick 2b production"
if [ ! -f $ART/oof_recipe_full_te_catboost_skte.npy ]; then
    python3 scripts/recipe_catboost_skte.py
else
    echo "  ✓ Pick 2b output exists; skip"
fi

phase "Phase 2: LGBM-skte production"
if [ ! -f $ART/oof_recipe_full_te_lgbm_skte.npy ]; then
    python3 scripts/recipe_lgbm_skte.py
else
    echo "  ✓ LGBM-skte output exists; skip"
fi

phase "Phase 3: RF natural meta retrain (META_SUFFIX=_a1lgbm)"
if [ ! -f $ART/oof_sklearn_rf_meta_natural_a1lgbm.npy ]; then
    META_SUFFIX=_a1lgbm python3 scripts/sklearn_rf_meta_natural.py
else
    echo "  ✓ Extended RF meta exists; skip"
fi

phase "Phase 4: Standalone CSV emit"
python3 scripts/emit_rf_natural_a1lgbm_standalone.py

phase "Done. Run blend_gate_rf_natural_full.py manually for 4-gate analysis."
