# Predicting Irrigation Need — Playground Series S6E4

<https://www.kaggle.com/competitions/playground-series-s6e4>

3-class classification (`Low` / `Medium` / `High`) on 19 tabular agronomy
features (soil chemistry, weather, crop/irrigation metadata). Train 630,000
rows, test 270,000 rows. Metric: **balanced accuracy** (macro-recall).
Severe class imbalance: 58.7 / 37.9 / **3.3** %.

Best public LB: **0.98094** — Tier-1b 4-stack:
`lb3 ⊗ realmlp@0.20 ⊗ xgb_nonrule_iso@0.075 ⊗ xgb_metastack_iso@0.30`
(log-space) with fixed log-bias `[1.4324, 1.4689, 3.4008]`, where
`lb3 = recipe_full_te 0.25 + recipe_pseudolabel 0.35 + recipe_pseudolabel_seed7labeler 0.40`.
Submission at `submissions/submission_tier1b_greedy_meta.csv`. Build by
running the per-component pipelines (recipe + pseudolabel chains, RealMLP
on Kaggle GPU, xgb_nonrule, xgb_metastack) then `scripts/tier1b_greedy_with_meta.py`.

OOF→LB gap is **−0.00010** (LB above OOF — meta-stacker CV-pessimism).
Pack 0.98114 is +0.00020 above; leader 0.98219 is +0.00125 above.

> **For any fresh Claude session / new container**: run `./bootstrap.sh`
> first. It installs deps and downloads the competition data. The
> `KAGGLE_API_TOKEN` env var is configured at the container level, so
> no interactive prompt is needed. Do **not** use `download_data.py`
> for the competition data — it targets an optional extra dataset.


## Reproduce

```bash
./bootstrap.sh                 # installs deps, prompts for Kaggle token, downloads data

# Build the final submission(s)
python scripts/<builder>.py
# or walk the narrative:
jupyter notebook notebooks/<final>.ipynb
```

`bootstrap.sh` reads the Kaggle API token via an interactive prompt (no
echo, not written to disk except as `~/.kaggle/kaggle.json`) and downloads
`data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.

> Competition data is **not** committed to git — it's covered by the
> competition Rules § 2.4.b (no redistribution). Re-run `bootstrap.sh`
> after each container restart to fetch it.

## Leaderboard ladder (selected)

| Submission file | OOF tuned | Public LB | Gap | Notes |
|---|---:|---:|---:|---|
| `submission_recipe_full_te.csv`                   | 0.97967 | 0.97939 | +0.00028 | Single-model recipe-XGB baseline |
| `submission_recipe_full_te_catboost.csv`          | 0.97936 | 0.97935 | +0.00001 | CatBoost twin (tightest gap) |
| `submission_recipe_greedy_recipe_pseudolabel.csv` | 0.98012 | 0.97998 | +0.00014 | 50/50 recipe × pseudo_s1 |
| `submission_3way_recipe025_s1035_s7040.csv`       | 0.98029 | 0.98005 | +0.00024 | 3-way multi-seed (RECOMMENDED HEDGE) |
| `submission_lb3_realmlp_nonruleiso.csv`           | 0.98061 | 0.98008 | +0.00053 | LB-best 3-stack (primary's base) |
| **`submission_tier1b_greedy_meta.csv`**           | **0.98084** | **0.98094** | **−0.00010** | **PRIMARY (LB best)** |

## Final-selection lock (deadline 2026-04-30)

- **Primary**: `submission_tier1b_greedy_meta.csv` → LB 0.98094.
- **Hedge**: `submission_3way_recipe025_s1035_s7040.csv` → LB 0.98005
  (premium −0.00089). Sidesteps the meta-stacker layer — the most-tuned,
  most-likely-private-LB-overfit element of primary.

The hedge swap from `submission_recipe_full_te.csv` (LB 0.97939, premium
−0.00155) to the 3-way multi-seed was accepted on 2026-04-25 per audit
F1 (`audit/2026-04-25-senior-engineer-audit.md`). Confirm the swap is
locked on Kaggle's final-selection UI before deadline.

## Repo layout

```
notebooks/          Final narrative notebooks.
scripts/            Every submission and analysis reproducible from here.
scripts/artifacts/  Cached models/OOF preds/features — survives restart.
data/               Competition data (gitignored, re-fetched via bootstrap.sh).
submissions/        Built submission CSVs.
plots/              High-signal diagnostics.
legacy/             Archive of exploratory code, stale plots, dead ends.
brief.md            Verbatim host material.
CLAUDE.md           Development log.
LEARNINGS.md        Portable patterns.
REPORT.md           Work report.
bootstrap.sh        One-shot environment re-hydration.
```

## Key findings

- `Irrigation_Need` is severely imbalanced (3.3% `High`) → balanced accuracy
  is dominated by minority-class recall → per-class log-bias tuning on OOF
  probabilities is the first lever (we use coord-ascent → fixed bias
  `[1.4324, 1.4689, 3.4008]`).
- **DGP is a closed-form rule on 6 features** (see `scripts/dgp_formula.py`).
  Rule reaches 100% on the 10k original; synthetic train has ~10,304
  rule-flipped rows (~1.6%). The rule alone scores LB 0.95835.
- **Synthetic flips are a deterministic NN function**, not Bernoulli noise
  (per 2026-04-21 EDA). Within-cell flip signal is feature-correlated
  (Humidity, Previous_Irrigation_mm, EC) but only at small effect sizes.
- **Saturation evidence at LB 0.98094**: 13+ independent attacks (greedy
  expansion, meta-stacker variants, LR meta, isotonic, bootstrap-bag,
  cross-poll, SMOTE-NC, soft-distill, perturbed-meta, score=6 boundary
  deep-dive, advanced ensembling, OvR-feature stack, bucket FE specialists,
  etc.) all land at OOF 0.98030–0.98090 and LB 0.97950–0.98010. Pareto
  frontier closure on rare-class High recall verified.
- **15 NN-family nulls** (MLP v5–v9, FT-Transformer, TabPFN-1.5k,
  pretrain-FT, DAE, RealMLP n_ens={1,2,4}, Trompt, Mambular SSM, KAN,
  TabPFN-10k, TabM). Magnitude trap is structural at this feature set —
  NN errors are orthogonal but in larger absolute count.
- The 0.98114 pack uses **public-CSV blending** (banned by repo rule).
  Within own-pipeline approaches the LB ceiling appears to be 0.98094.
