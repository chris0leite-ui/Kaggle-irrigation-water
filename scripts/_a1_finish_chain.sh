#!/bin/bash
set -e
cd /home/user/Kaggle-irrigation-water
echo "[a1-finish] starting at $(date)"

# Step 1: XGB fold 5
echo "[a1-finish] launching XGB fold 5"
RUN_FOLD=5 python scripts/recipe_xgb_skte.py 2>&1 | grep -v "PerformanceWarning\|fragmented\|^\s\+orig" | tail -5
if [ ! -f "scripts/artifacts/oof_xgb_skte_fold5.npy" ]; then
  echo "[a1-finish] FATAL: XGB fold 5 did not save"
  exit 1
fi
echo "[a1-finish] fold 5 saved at $(date)"

# Step 2: XGB aggregate
echo "[a1-finish] aggregating XGB"
python scripts/recipe_xgb_skte.py 2>&1 | grep -v "PerformanceWarning\|fragmented\|^\s\+orig" | tail -10
if [ ! -f "scripts/artifacts/oof_recipe_full_te_xgb_skte.npy" ]; then
  echo "[a1-finish] FATAL: XGB aggregate did not save"
  exit 1
fi
echo "[a1-finish] XGB aggregate at $(date)"

# Step 3: RF natural rebuild on expanded 10-component bank
echo "[a1-finish] running RF natural rebuild"
python scripts/sklearn_rf_meta_natural.py 2>&1 | tail -30

# Step 4: full blend-gate analysis
echo "[a1-finish] running blend gate analysis"
python scripts/blend_gate_rf_natural_full.py 2>&1 | tail -50

echo "[a1-finish] DONE at $(date)"
