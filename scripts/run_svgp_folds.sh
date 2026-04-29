#!/usr/bin/env bash
# Runs SVGP folds sequentially in foreground. Per-fold checkpoints
# are saved to scripts/artifacts/oof_xgb_metastack_svgp_fold{N}.npy.
# After all 5 folds complete, runs the aggregator + blend gate.
#
# Usage:
#   ./scripts/run_svgp_folds.sh        # runs all 5 folds + aggregate + gate
#   START_FOLD=3 ./scripts/run_svgp_folds.sh   # resume from fold 3
set -e
START_FOLD=${START_FOLD:-1}
END_FOLD=${END_FOLD:-5}
echo "[run_svgp_folds] folds $START_FOLD..$END_FOLD"
for f in $(seq $START_FOLD $END_FOLD); do
    if [ -f scripts/artifacts/oof_xgb_metastack_svgp_fold${f}.npy ]; then
        echo "[run_svgp_folds] fold $f already done, skipping"
        continue
    fi
    echo "[run_svgp_folds] === starting fold $f ==="
    PCA_DIM=50 RUN_FOLD=$f python scripts/svgp_metastack.py 2>&1 | \
        tee scripts/artifacts/svgp_fold${f}.log
    if [ ! -f scripts/artifacts/oof_xgb_metastack_svgp_fold${f}.npy ]; then
        echo "[run_svgp_folds] FOLD $f FAILED — check scripts/artifacts/svgp_fold${f}.log"
        exit 1
    fi
    echo "[run_svgp_folds] === fold $f done ==="
done
echo "[run_svgp_folds] all folds done; aggregating"
python scripts/svgp_aggregate.py
echo "[run_svgp_folds] running blend gate"
python scripts/svgp_blend_gate.py
echo "[run_svgp_folds] DONE"
