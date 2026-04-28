#!/usr/bin/env bash
# Self-restarting DART production loop. Survives container rehydrate via
# per-fold checkpoints (recipe_full_te.py saves oof_recipe_full_te_dart_foldN
# after each fold; on restart, scans for cached folds + skips them).
#
# Loop terminates when the final aggregate file exists OR an explicit
# error is detected. Each iteration runs the python script; if killed
# mid-fold, the next iteration resumes from completed folds.
#
# Usage (foreground via Monitor or Bash):
#   ./scripts/run_dart_resilient.sh
set -e
ART=scripts/artifacts
FINAL=${ART}/oof_recipe_full_te_dart.npy
RESULTS=${ART}/recipe_full_te_dart_results.json

attempt=0
while [ ! -f "$FINAL" ] || [ ! -f "$RESULTS" ]; do
    attempt=$((attempt + 1))
    echo "[run_dart_resilient] === attempt $attempt at $(date +%H:%M:%S) ==="
    n_cached=$(ls ${ART}/oof_recipe_full_te_dart_fold*.npy 2>/dev/null | wc -l)
    echo "[run_dart_resilient] cached folds: $n_cached / 5"
    XGB_BOOSTER=dart python scripts/recipe_full_te.py 2>&1 | \
        grep -E "(fold|argmax|tuned|wrote|Error|Traceback|cached|resume|elapsed)" \
        || echo "[run_dart_resilient] python exit code $? — will retry"
    if [ -f "$FINAL" ] && [ -f "$RESULTS" ]; then
        echo "[run_dart_resilient] === COMPLETE at $(date +%H:%M:%S) ==="
        break
    fi
    echo "[run_dart_resilient] iteration $attempt finished without final; retry in 5s"
    sleep 5
done

echo ""
echo "[run_dart_resilient] === running auto-analysis ==="
python scripts/dart_blend_gate.py
echo ""
python scripts/dart_minimal_check.py
echo ""
echo "[run_dart_resilient] DONE"
