# 13 — Repo template

The skeleton to clone for the next comp. Lift list at the bottom.

## Folder skeleton

```
<comp-slug>/
├── data/                    # competition CSVs (gitignored)
├── scripts/                 # reproducible analysis + builders
├── submissions/             # built CSVs (only submission_*.csv tracked)
├── audit/                   # timestamped postmortems, decision log
├── notebooks/               # narrative notebooks (final-submission nb)
├── tests/                   # OOF invariant smoke tests
├── prompts/                 # LLM prompt templates (if used)
├── kaggle_kernel/           # one-shot kernel push scaffolding
├── plots/                   # diagnostics organised by topic
├── legacy/                  # archived dead-end work
├── CLAUDE.md                # running log + ⚠️ rules (≤50k tokens)
├── LEARNINGS.md             # portable patterns
├── REPORT.md                # work report
├── brief.md                 # verbatim host material
├── comp-context.md          # settled-once facts (Day 1)
├── README.md                # TL;DR + reproduction
├── bootstrap.sh             # auto-deps + data download
├── requirements.txt         # minimal
└── .gitignore               # inverted artifact policy (see below)
```

## What goes in `comp-context.md` (Day 1)

A short fixed-fact sheet. Filled out once on Day 1 from the comp
page; never re-asked.

```
- Competition slug:
- URL:
- Task: (binary / multiclass / regression / ...)
- Metric:
- Public-LB split %:
- LB stability assumption: (stable / probe-once / per-row-seeded)
- Train / test row counts:
- Feature count: (numeric, categorical breakdown)
- Class priors: (if classification)
- Deadline:
- Team-size limit:
- Submission budget: (typically 10/day, 2 final)
- Data license:
- External data allowed: (yes / no / conditional)
- LB-best at kickoff:
- Pack score at rank 100:
```

## Inverted .gitignore for artifacts

Default-track `.npy` and `.json` artifacts under `scripts/artifacts/`.
Ignore per-fold checkpoints and transient prefixes:

```gitignore
# (from this comp's working .gitignore — copy verbatim)
scripts/artifacts/*
!scripts/artifacts/oof_*.npy
!scripts/artifacts/test_*.npy
!scripts/artifacts/*_results.json
scripts/artifacts/*_fold[0-9]*.npy
scripts/artifacts/*.log
scripts/artifacts/*.db
scripts/artifacts/*.pkl
scripts/artifacts/_smoke_*.npy
scripts/artifacts/tmp_*.npy
scripts/artifacts/scratch_*.npy
```

This eliminates the prior 536-line per-file whitelist. Cross-branch
artifact sharing is now zero-friction.

## Lift list — files to copy verbatim

Verified by survey of this repo. Each is small, well-tested, and
generic enough to drop into a new comp with minimal edits.

| Path | What it does | Edit needed? |
|---|---|---|
| `bootstrap.sh` | Installs deps, fetches comp data via Kaggle CLI, falls back to interactive token prompt. | Change comp slug. |
| `scripts/lb_status.py` | Parses `kaggle competitions submissions` into a queryable list. Used as the single source of truth for "what's been probed". | Change comp slug. |
| `scripts/common.py` | Shared OOF/CV conventions (5-fold StratifiedKFold, class mapping, carrier detection). | Maybe change CV strategy / class mapping. |
| `scripts/meta_common.py` | Meta-learner utilities (load OOF bank, align rows, build Pareto frontier). | None. |
| `tests/test_oof_invariants.py` | Smoke tests on committed OOF artifacts (shape, sparsity, normalization). | None. |
| `scripts/T2_conformal_helpers.py` | Split-conformal calibration scaffold. | Optional; only if calibration matters. |
| `.gitignore` (artifact policy) | Inverted-tracking pattern. | None — copy block. |

## Engineering principles to keep

These are the implicit rules the irrigation repo followed. Make them
explicit in `CLAUDE.md` of the new comp:

1. **Files ≤150 lines.** Every script. Every doc.
2. **One submission CSV per builder script.** `build_*_submission.py`
   does exactly one job, deterministically.
3. **Atomic artifact writes.** Builders write to a `tmp_` path then
   rename; never leave half-written `.npy`.
4. **OOF-shape invariants in `tests/`.** Smoke-runnable in <30s.
   Catches fold-leak and shape regressions.
5. **Tracked artifacts (no whitelists).** Inverted `.gitignore`.
6. **Audit dir convention.** One timestamped `.md` per saturation
   or postmortem-worthy event. Format:
   `audit/YYYY-MM-DD-<topic>.md`.
7. **`CLAUDE.md` is the running log + ⚠️ rules**, not a brain dump.
   Archive when bloated.

## What NOT to copy

- The 536-script `scripts/` dir (lots of one-off candidates,
  comp-specific). Lift only `common.py`, `meta_common.py`,
  `lb_status.py`, `dgp_formula.py` if applicable, and the
  conformal helper.
- The 404 submission CSVs.
- The audit/ entries (comp-specific). The *convention* lifts;
  the *content* doesn't.
- The competition-specific override mechanism. It worked because
  the DGP was rule-shaped; a different comp will have a different
  DGP shape.
