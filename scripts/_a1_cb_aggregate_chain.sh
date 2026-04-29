#!/bin/bash
set -e
cd /home/user/Kaggle-irrigation-water
echo "[cb-agg] starting at $(date)"
until [ -f scripts/artifacts/oof_catboost_skte_fold5.npy ]; do sleep 30; done
echo "[cb-agg] fold 5 saved at $(date); aggregating CB"
python scripts/recipe_catboost_skte.py 2>&1 | grep -v "PerformanceWarning\|fragmented\|^\s\+orig" | tail -10
if [ -f scripts/artifacts/oof_recipe_full_te_catboost_skte.npy ]; then
  echo "[cb-agg] CB AGGREGATE COMPLETE at $(date)"
else
  echo "[cb-agg] WARN: CB aggregate file not produced"
fi
