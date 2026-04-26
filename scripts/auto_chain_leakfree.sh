#!/usr/bin/env bash
# Wait for multitask to finish, then auto-launch leakfree teacher production,
# then leakfree distill student. Logs to logs/<step>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

# Wait for multitask completion artifact.
echo "[$(date +%H:%M:%S)] auto_chain: waiting for multitask completion"
until [ -f scripts/artifacts/oof_multitask_xgb.npy ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] multitask done; launching leakfree teacher"

python3 scripts/leakfree_teacher_oof.py > logs/leakfree_teacher_prod.log 2>&1
echo "[$(date +%H:%M:%S)] leakfree teacher done; launching leakfree distill"

python3 scripts/leakfree_distill.py > logs/leakfree_distill_prod.log 2>&1
echo "[$(date +%H:%M:%S)] leakfree distill done; running blend gate"

python3 scripts/blend_gate_3way_v2.py > logs/blend_gate_3way_prod.log 2>&1
echo "[$(date +%H:%M:%S)] all done"
