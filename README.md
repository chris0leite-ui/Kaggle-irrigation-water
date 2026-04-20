# <Competition Name>

<kaggle URL>

<1–2 sentence task description: rows, features, metric.>

Best public LB: **<score>** with <short description>.

## Reproduce

```bash
pip install -r requirements.txt

# Place competition files into data/ (not shipped with the repo)
kaggle competitions download -c <slug> -p data/ && unzip -o data/*.zip -d data/

# Build the final submission(s)
python scripts/<builder>.py
# or walk the narrative:
jupyter notebook notebooks/<final>.ipynb
```

## Leaderboard tiers

| Submission file | CV | Public LB | Notes |
|---|---:|---:|---|
| `submission_<name>.csv` | — | — | — |

## Repo layout

```
notebooks/    Final narrative notebooks.
scripts/      Every submission and analysis reproducible from here.
data/         Competition data (gitignored).
submissions/  Built submission CSVs.
plots/        High-signal diagnostics.
legacy/       Archive of exploratory code, stale plots, dead ends.
brief.md      Verbatim host material.
CLAUDE.md     Development log.
LEARNINGS.md  Portable patterns.
REPORT.md     Work report.
```

## Key findings

- (fill in as the competition progresses)
