#!/usr/bin/env bash
# UserPromptSubmit hook — inject latest git log + Kaggle LB state into context
# so the model isn't relying on the (monotonically-stale) injected CLAUDE.md
# snapshot when reasoning about LB-best, recently-probed submissions, etc.
#
# Triggered each time the user submits a prompt. Reads stdin (we ignore it),
# writes a JSON envelope on stdout with `additionalContext` for the model.

set -u
cd /home/user/Kaggle-irrigation-water 2>/dev/null || true

GIT_OUT=$(git log --oneline -10 2>/dev/null || echo "(not a git repo)")
LB_OUT=$(python scripts/lb_status.py --top 10 2>/dev/null || echo "(kaggle CLI unavailable; check git for ground-truth recent LB results)")

CONTENT=$(printf "## Latest git commits (last 10) — ground truth, not the CLAUDE.md snapshot\n%s\n\n## Kaggle LB submissions top 10\n%s\n" "$GIT_OUT" "$LB_OUT")

jq -Rn --arg c "$CONTENT" '{hookSpecificOutput: {hookEventName: "UserPromptSubmit", additionalContext: $c}}'
