# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Competition

- **Name**: <competition name>
- **URL**: <kaggle URL>
- **Task**: <e.g. tabular regression, MAE metric>
- **Deadline**: <YYYY-MM-DD>
- **LB submission budget**: <daily> / <total>, <n> spent

See `brief.md` for the full host material (description, rules,
evaluation, data description, host forum posts).

## Commands

```bash
pip install -r requirements.txt

# Download competition data into data/
kaggle competitions download -c <slug> -p data/ && unzip -o data/<slug>.zip -d data/
```

## Architecture

```
notebooks/     Narrative notebooks. Final submission notebook lives here.
scripts/       Reproducible analysis and submission-builder scripts.
data/          Competition data (gitignored).
submissions/   Built submission CSVs (only submission_*.csv committed).
plots/         Diagnostics, organised by topic subfolder.
legacy/        Archived exploratory code, stale plots, dead ends.
brief.md       Verbatim host material (description, rules, eval, data).
CLAUDE.md      This file — development log and session guidance.
LEARNINGS.md   Portable patterns for future competitions.
REPORT.md      Work report: observations, models, results, rejected ideas.
README.md      TL;DR + reproduction instructions.
```

## Session log

### YYYY-MM-DD — kickoff

- Goal:
- Changed:
- LB delta:
- Next bet:

## Hypothesis board

- **Open**:
- **Ruled out**:
- **Parked**:

## Playbook

The reusable Kaggle playbook lives at
<https://github.com/chris0leite-ui/kaggle-claude-code-setup> (branch
`claude/kaggle-playbook`). Kickoff steps, workflow norms, and
methodology are maintained there — update that repo when a transferable
lesson surfaces.
