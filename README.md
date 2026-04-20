# Predicting Irrigation Need — Playground Series S6E4

<https://www.kaggle.com/competitions/playground-series-s6e4>

3-class classification (`Low` / `Medium` / `High`) on 19 tabular agronomy
features (soil chemistry, weather, crop/irrigation metadata). Train 630,000
rows, test 270,000 rows. Metric: **balanced accuracy** (macro-recall).
Severe class imbalance: 58.7 / 37.9 / **3.3** %.

Best public LB: **—** (not yet submitted).

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

## Leaderboard tiers

| Submission file | CV | Public LB | Notes |
|---|---:|---:|---|
| `submission_<name>.csv` | — | — | — |

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
  is dominated by minority-class recall → per-class threshold tuning on
  OOF probabilities is the first lever to try before any model ensembling.
- The 0.98114 tied pack (~100+ teams exactly tied as of 2026-04-20) looks
  like a ceiling from the default public baseline. Real movement above
  0.98114 probably comes from threshold tuning, careful class weighting,
  or adding the original Irrigation Prediction dataset as extra training
  signal.
