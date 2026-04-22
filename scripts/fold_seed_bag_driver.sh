#!/usr/bin/env bash
# Sequential driver: run fold_seed_bag_pipeline.py across fold seeds.
# Usage: bash scripts/fold_seed_bag_driver.sh [seed1 seed2 ...]
# Defaults to the 5-seed spec [42, 7, 123, 2024, 9999].
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(42 7 123 2024 9999)
fi

mkdir -p scripts/artifacts/fold_seed_logs

for seed in "${SEEDS[@]}"; do
    LOGFILE="scripts/artifacts/fold_seed_logs/pipeline_fs${seed}.log"
    echo "===== fold_seed=${seed}  log=${LOGFILE} ====="
    FOLD_SEED="${seed}" python3 scripts/fold_seed_bag_pipeline.py 2>&1 | tee "${LOGFILE}"
    echo "SEED_DONE fs=${seed}"
done

echo "ALL_SEEDS_DONE seeds=${SEEDS[*]}"
