#!/usr/bin/env bash
# Auto-triggered analysis pipeline once DART production completes.
# Polls for oof_recipe_full_te_dart.npy + results JSON, then runs:
#   1. dart_blend_gate.py (4-gate vs LB-best primary)
#   2. dart_minimal_check.py (CLAUDE.md leakage defense)
set -e

ART=scripts/artifacts
TARGET=${ART}/oof_recipe_full_te_dart.npy
RESULTS=${ART}/recipe_full_te_dart_results.json

echo "[dart_analysis] waiting for production complete: $TARGET"
while [ ! -f "$TARGET" ] || [ ! -f "$RESULTS" ]; do
    sleep 60
done
echo "[dart_analysis] $TARGET ready, running analyses"

echo "=== production summary ==="
python -c "
import json, numpy as np
r = json.load(open('${RESULTS}'))
print(f'argmax_oof = {r.get(\"argmax_oof\", \"?\")}')
print(f'tuned_oof  = {r.get(\"tuned_oof\", \"?\")}')
print(f'fold scores: {r.get(\"fold_scores\", \"?\")}')
"

echo ""
echo "=== 4-gate analysis ==="
python scripts/dart_blend_gate.py

echo ""
echo "=== minimal-input check ==="
python scripts/dart_minimal_check.py

echo ""
echo "[dart_analysis] DONE"
