#!/bin/bash
set -e
cd /home/user/Kaggle-irrigation-water
echo "[cb-chain-v2] starting at $(date)"
until [ -f scripts/artifacts/oof_catboost_skte_fold2.npy ]; do sleep 30; done
echo "[cb-chain-v2] fold 2 ok at $(date); folds 3-5 sequential"
for f in 3 4 5; do
  echo "[cb-chain-v2] launching fold $f at $(date)"
  RUN_FOLD=$f python scripts/recipe_catboost_skte.py 2>&1 | grep -v "PerformanceWarning\|fragmented\|^\s\+orig" | tail -3
  if [ ! -f "scripts/artifacts/oof_catboost_skte_fold${f}.npy" ]; then
    echo "[cb-chain-v2] WARN: fold $f did not save"
    exit 1
  fi
  echo "[cb-chain-v2] fold $f saved"
done
echo "[cb-chain-v2] aggregating CB at $(date)"
python scripts/recipe_catboost_skte.py 2>&1 | grep -v "PerformanceWarning\|fragmented\|^\s\+orig" | tail -10
echo "[cb-chain-v2] DONE at $(date)"
