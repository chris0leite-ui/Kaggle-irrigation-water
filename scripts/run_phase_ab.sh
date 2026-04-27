#!/usr/bin/env bash
# Sequential foreground runner for Phase A (residual TE) + Phase B (base-margin).
# Per-fold checkpointing means each invocation resumes where it left off.
#
# Usage:
#   bash scripts/run_phase_ab.sh phase_a       # run all 5 Phase A folds + aggregate
#   bash scripts/run_phase_ab.sh phase_b       # run all 5 Phase B folds + aggregate
#   bash scripts/run_phase_ab.sh phase_a 3     # run only Phase A fold 3
#   bash scripts/run_phase_ab.sh both          # serial: Phase A all → Phase B all
#
# After both phases complete, run:
#   python scripts/blend_gate_4gate.py --candidate residte
#   python scripts/blend_gate_4gate.py --candidate residte --use-iso
#   python scripts/blend_gate_4gate.py --candidate basemargin_K4
#   python scripts/blend_gate_4gate.py --candidate basemargin_K4 --use-iso
set -euo pipefail
cd "$(dirname "$0")/.."

phase=${1:-both}
single_fold=${2:-}

run_phase_a() {
    if [ -n "$single_fold" ]; then
        echo "=== Phase A fold $single_fold ==="
        RUN_FOLD=$single_fold python scripts/residual_te.py 2>&1 | grep -vE 'PerformanceWarning|^  df\['
    else
        for f in 1 2 3 4 5; do
            echo "=== Phase A fold $f ==="
            RUN_FOLD=$f python scripts/residual_te.py 2>&1 | grep -vE 'PerformanceWarning|^  df\['
        done
        echo "=== Phase A aggregate ==="
        python scripts/residual_te.py 2>&1 | grep -vE 'PerformanceWarning|^  df\['
    fi
}

run_phase_b() {
    if [ -n "$single_fold" ]; then
        echo "=== Phase B fold $single_fold ==="
        RUN_FOLD=$single_fold K_MARGIN=4.0 python scripts/recipe_basemargin.py 2>&1 | grep -vE 'PerformanceWarning|^  df\['
    else
        for f in 1 2 3 4 5; do
            echo "=== Phase B fold $f (K=4) ==="
            RUN_FOLD=$f K_MARGIN=4.0 python scripts/recipe_basemargin.py 2>&1 | grep -vE 'PerformanceWarning|^  df\['
        done
        echo "=== Phase B aggregate ==="
        K_MARGIN=4.0 python scripts/recipe_basemargin.py 2>&1 | grep -vE 'PerformanceWarning|^  df\['
    fi
}

case "$phase" in
    phase_a) run_phase_a ;;
    phase_b) run_phase_b ;;
    both)
        run_phase_a
        run_phase_b
        ;;
    *)
        echo "Usage: $0 {phase_a|phase_b|both} [fold]" >&2
        exit 1
        ;;
esac
echo "=== run_phase_ab.sh DONE ==="
