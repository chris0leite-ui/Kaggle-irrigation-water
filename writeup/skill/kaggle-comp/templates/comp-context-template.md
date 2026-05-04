# comp-context.md — settled-once facts

Filled out once on Day 1 by the kickoff agent (API auto-fill +
batch confirm with PI). NEVER re-asked. Loaded by every subagent
on session start.

## Facts (auto-filled by kaggle CLI)

```yaml
slug: {{COMP_SLUG}}
url: https://www.kaggle.com/competitions/{{COMP_SLUG}}
title: {{TITLE}}
task: {{TASK}}                  # binary | multiclass | regression
metric: {{METRIC}}              # bal_acc | log_loss | rmse | auc | ...
train_rows: {{N_TRAIN}}
test_rows: {{N_TEST}}
deadline: {{DEADLINE}}
team_size_limit: {{TEAM_LIMIT}}
submission_budget: {{DAILY_LIMIT}}/day
final_submissions: {{FINAL_LIMIT}}
data_license: {{LICENSE}}
external_data_allowed: {{EXTERNAL}}    # yes | no | conditional
```

## Schema (auto-filled by kickoff EDA)

```yaml
target_col: {{TARGET_COL}}
id_col: {{ID_COL}}
feature_count:
  numeric: {{N_NUM}}
  categorical: {{N_CAT}}
class_priors: {{PRIORS}}        # if classification, else null
missingness_train: {{MISS_TRAIN}}
missingness_test: {{MISS_TEST}}
```

## LB context (auto-filled from leaderboard download)

```yaml
lb_best_at_kickoff: {{LB_BEST}}
pack_score_at_rank_100: {{LB_RANK_100}}
total_teams_at_kickoff: {{N_TEAMS}}
top_5pct_rank: {{RANK_5PCT}}    # 0.05 * N_TEAMS
top_5pct_score: {{SCORE_5PCT}}  # rank top_5pct_rank's score
public_split_pct: 20            # default for Playground; confirm for Featured
probe_resolution_floor: 0.00005 # 80/20 split × N_TEST (re-derive if Featured)
```

## Strategic decisions (PI-answered, batched Q4 in kickoff)

```yaml
lb_stability: {{STABILITY}}               # stable | per-row-seeded | unknown
external_data_strategy: {{EXT_STRATEGY}}  # use | skip | depends_on_rules
time_budget_per_day_h: {{TIME_DAY}}
time_budget_total_days: {{TIME_TOTAL}}
compute_budget: {{COMPUTE}}               # cpu_only | gpu_kaggle | gpu_local
model_preferences: {{MODELS}}             # e.g. "trees first, NN if obvious"
```

## Daily-updated facts

These do change. The Bookkeeper updates them in CLAUDE.md
current-state, NOT here.

```yaml
# In CLAUDE.md, not comp-context.md:
# - lb_best_today
# - our_lb_best
# - submissions_used_today
# - saturation_count
```

## Anti-patterns

- Don't re-ask any field above. Read this file instead.
- Don't add new strategic-decision fields after Day 1 without
  flagging it as a friction event (`audit/friction.md`,
  `tag: settled-once`).
- Don't put per-experiment OOF/LB results here. They go in
  `audit/YYYY-MM-DD-*.md` and the calibration ladder.
