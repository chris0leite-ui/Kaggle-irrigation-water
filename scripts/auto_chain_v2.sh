#!/usr/bin/env bash
# Self-restarting auto-chain. Runs leakfree teacher in a retry loop —
# every rehydrate kill is auto-resumed since the script is checkpoint-aware.
# Once the teacher OOFs are all 5 saved, runs distill + blend gate.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_TEACHER="logs/leakfree_teacher_prod.log"
LOG_DISTILL="logs/leakfree_distill_prod.log"
LOG_GATE="logs/blend_gate_3way_prod.log"

is_teacher_done() {
    [ -f scripts/artifacts/oof_recipe_leakfree_outer1.npy ] && \
    [ -f scripts/artifacts/oof_recipe_leakfree_outer2.npy ] && \
    [ -f scripts/artifacts/oof_recipe_leakfree_outer3.npy ] && \
    [ -f scripts/artifacts/oof_recipe_leakfree_outer4.npy ] && \
    [ -f scripts/artifacts/oof_recipe_leakfree_outer5.npy ] && \
    [ -f scripts/artifacts/test_recipe_leakfree_outer1.npy ] && \
    [ -f scripts/artifacts/test_recipe_leakfree_outer2.npy ] && \
    [ -f scripts/artifacts/test_recipe_leakfree_outer3.npy ] && \
    [ -f scripts/artifacts/test_recipe_leakfree_outer4.npy ] && \
    [ -f scripts/artifacts/test_recipe_leakfree_outer5.npy ]
}

attempt=0
while ! is_teacher_done; do
    attempt=$((attempt + 1))
    echo "[$(date +%H:%M:%S)] auto_chain_v2: teacher attempt $attempt"
    python3 scripts/leakfree_teacher_oof.py >> "$LOG_TEACHER" 2>&1 || \
        echo "[$(date +%H:%M:%S)] auto_chain_v2: teacher exited (rc=$?), will retry"
    sleep 10
done

echo "[$(date +%H:%M:%S)] auto_chain_v2: teacher done; running distill"
python3 scripts/leakfree_distill.py > "$LOG_DISTILL" 2>&1
echo "[$(date +%H:%M:%S)] auto_chain_v2: distill done; running blend gate"
python3 scripts/blend_gate_3way_v2.py > "$LOG_GATE" 2>&1
echo "[$(date +%H:%M:%S)] auto_chain_v2: ALL DONE"
