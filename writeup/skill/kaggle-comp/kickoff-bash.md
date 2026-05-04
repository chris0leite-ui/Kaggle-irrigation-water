# Kickoff bash batches — companion to kickoff-runbook.md

Each section below is one batch the agent runs at the corresponding
runbook step. Variables (`$SLUG`, `$DIR`, etc.) are extracted from
the chat Q&A turns and persist across batches in the same session.

## pre-flight

```bash
which kaggle || pip install -q kaggle
[[ -n "$KAGGLE_API_TOKEN" ]] || echo "MISSING_TOKEN"
curl -s -o /dev/null -w "%{http_code}" https://www.kaggle.com --max-time 5
```

## batch-A-scaffold

```bash
mkdir -p "$DIR" && cd "$DIR"
SKILL=~/.claude/skills/kaggle-comp
cp -r "$SKILL/templates/." .
mv gitignore-template .gitignore
mkdir -p data submissions audit plots/eda scripts/artifacts notebooks
touch data/.gitkeep submissions/.gitkeep audit/.gitkeep
sed -i "s/{{COMP_SLUG}}/$SLUG/g" \
    bootstrap.sh CLAUDE.md comp-context.md scripts/lb_status.py
chmod +x bootstrap.sh
git init -q && git add . && git commit -q -m "kickoff scaffold"
```

## batch-B-context

```bash
kaggle competitions view "$SLUG" > /tmp/_view.txt
kaggle competitions leaderboard "$SLUG" --download -p /tmp/_lb -q
LB_CSV=/tmp/_lb/${SLUG}.zip
N_TEAMS=$(unzip -p "$LB_CSV" | tail -n +2 | wc -l)
LB_BEST=$(unzip -p "$LB_CSV" | awk -F, 'NR==2{print $5}')
LB_R100=$(unzip -p "$LB_CSV" | awk -F, '$1==101{print $5}')
RANK_5PCT=$(python -c "print(int($N_TEAMS * 0.05))")
SCORE_5PCT=$(unzip -p "$LB_CSV" | awk -F, -v r="$RANK_5PCT" '$1==r{print $5}')
# Substitute {{LB_BEST}}, {{LB_RANK_100}}, {{N_TEAMS}}, {{RANK_5PCT}},
# {{SCORE_5PCT}}, {{TITLE}}, {{TASK}}, {{METRIC}}, {{DEADLINE}},
# {{TEAM_LIMIT}}, {{DAILY_LIMIT}}, {{FINAL_LIMIT}}, {{LICENSE}},
# {{N_TRAIN}}, {{N_TEST}} in comp-context.md from /tmp/_view.txt parse.
```

## batch-C-data

```bash
./bootstrap.sh
HEADER=$(head -1 data/sample_submission.csv)
ID_COL=$(echo "$HEADER" | cut -d, -f1)
TARGET_COL=$(echo "$HEADER" | cut -d, -f2)
sed -i "s/{{ID_COL}}/$ID_COL/; s/{{TARGET_COL}}/$TARGET_COL/" comp-context.md
python scripts/eda.py
```

## batch-D-baseline

```bash
python scripts/baseline_lgbm.py
RES=scripts/artifacts/baseline_lgbm_results.json
OOF=$(python -c "import json; print(json.load(open('$RES'))['oof_score'])")
STD=$(python -c "import json; print(json.load(open('$RES'))['fold_std'])")
DAILY=$(grep '^submission_budget' comp-context.md | grep -oE '[0-9]+' | head -1)
```

## batch-D2-submit

```bash
kaggle competitions submit "$SLUG" \
    -f submissions/submission_baseline_lgbm.csv \
    -m "baseline LGBM 5-fold OOF $OOF (kickoff Day-1)"
sleep 10
LB=$(kaggle competitions submissions "$SLUG" \
     -v 2>/dev/null | awk -F, 'NR==2{print $7}' | tr -d '"')
```

## batch-E-audit

```bash
GAP=$(python -c "print($LB - $OOF)")
TWO_STD=$(python -c "print(2 * $STD)")
VERDICT=$(python -c "print('CALIBRATED' if abs($LB - $OOF) < $TWO_STD else 'DRIFT')")
DATE=$(date +%Y-%m-%d)

cat > "audit/${DATE}-day-1-kickoff.md" <<EOF
# Day 1 — kickoff

## Comp context
- slug: $SLUG  task: $TASK  metric: $METRIC
- LB-best at kickoff: $LB_BEST  rank-100: $LB_R100
- top-5% cutoff (rank $RANK_5PCT): $SCORE_5PCT

## Baseline LGBM
- OOF: $OOF  fold-std: $STD
- LB:  $LB
- Gap: $GAP  → $VERDICT

## Day-2 queue (PI can edit)
1. Domain hypothesis seeder — read existing knowledge of this problem.
2. Heuristic baselines — H1 single-feature threshold, H2 hand rule.
3. DGP archaeology if synthetic — brute-force candidate rules.
EOF

sed -i "s/our_lb_best: null/our_lb_best: $LB/" CLAUDE.md
sed -i "s/submissions_used_today: 0/submissions_used_today: 1/" CLAUDE.md
git add . && git commit -q -m "Day 1 kickoff: OOF $OOF → LB $LB ($VERDICT)"
```

## On any batch failure

1. Capture the error to a 1-line summary.
2. Surface to PI in chat with the fix-path question.
3. Do NOT retry automatically. PI decides.
4. Append to `audit/friction.md`:
   `<date>  tool-missing/<batch-name>  <one-line summary>`.
