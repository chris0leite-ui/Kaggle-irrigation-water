# Templates — drop-in files for a fresh Kaggle comp repo

The kickoff agent (`../kickoff-runbook.md`) copies these into a new
comp directory. Files with `{{COMP_SLUG}}` / `{{TARGET_COL}}` / etc.
placeholders are sed-substituted during copy.

## Layout copied to the new repo

```
<new-comp-dir>/
├── bootstrap.sh            ← bootstrap.sh             (placeholder: COMP_SLUG)
├── requirements.txt        ← requirements.txt
├── .gitignore              ← gitignore-template       (renamed)
├── CLAUDE.md               ← CLAUDE-template.md       (placeholder: COMP_SLUG)
├── comp-context.md         ← comp-context-template.md (many placeholders)
├── brief.md                ← brief-template.md
├── scripts/
│   ├── lb_status.py        ← scripts/lb_status.py     (placeholder: COMP_SLUG)
│   ├── common.py           ← scripts/common.py
│   ├── baseline_lgbm.py    ← scripts/baseline_lgbm.py
│   └── eda.py              ← scripts/eda.py
└── tests/
    └── test_oof_invariants.py ← tests/test_oof_invariants.py
```

After copy, the kickoff agent also creates empty:
`data/`, `submissions/`, `audit/`, `plots/eda/`,
`scripts/artifacts/`, `notebooks/` with `.gitkeep`.

## Placeholders

Substituted by the kickoff agent during copy:

- `{{COMP_SLUG}}` — kaggle competition slug
- `{{TITLE}}`, `{{TASK}}`, `{{METRIC}}`, etc. — from
  `kaggle competitions view <slug>` output
- `{{TARGET_COL}}`, `{{ID_COL}}`, etc. — from
  `data/sample_submission.csv` header

The agent does substitution via `sed -i` per file; placeholders
that aren't filled remain as `{{...}}` (visible TODO).

## What's NOT in templates/

- `meta_common.py` — too comp-specific to ship as a template; users
  add stacking utilities later when there's a bank to stack.
- The 14 framework guardrail files — those live in the skill itself
  (`../guardrails.md`, `../personas.md`, etc.) and are loaded by
  the agent during the day-loop, not copied into the new repo.
- Any per-fold artifact, OOF, or LB result.

## Maintenance

When this irrigation-water repo evolves a generic-enough utility
(better CV helper, calibration scaffold, override-detection check),
copy it here. Keep each file ≤150 lines.
