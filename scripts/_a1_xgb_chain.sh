#!/bin/bash
set -e
cd /home/user/Kaggle-irrigation-water
echo "[xgb-seq] starting at $(date)"
until [ -f scripts/artifacts/oof_recipe_full_te_catboost_skte.npy ]; do sleep 60; done
echo "[xgb-seq] CB aggregate ready at $(date); sleeping 30s then sequential XGB"
sleep 30
for f in 1 2 3 4 5; do
  echo "[xgb-seq] launching XGB fold $f at $(date)"
  RUN_FOLD=$f python scripts/recipe_xgb_skte.py 2>&1 | grep -v "PerformanceWarning\|fragmented\|^\s\+orig" | tail -3
  if [ ! -f "scripts/artifacts/oof_xgb_skte_fold${f}.npy" ]; then
    echo "[xgb-seq] WARN: fold $f did not save"
    exit 1
  fi
  echo "[xgb-seq] fold $f saved"
done
echo "[xgb-seq] aggregating XGB at $(date)"
python scripts/recipe_xgb_skte.py 2>&1 | grep -v "PerformanceWarning\|fragmented\|^\s\+orig" | tail -10
echo "[xgb-seq] DONE at $(date)"
