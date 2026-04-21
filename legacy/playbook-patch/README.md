# Playbook update — manual-push handoff

The sandbox can't push to `chris0leite-ui/kaggle-claude-code-setup`
(local proxy denies non-project repos). The commit is prepared and
packaged here for manual landing.

## What's included

- `CLAUDE.md` — the full rewritten playbook methodology. Drop into
  repo root, replacing the existing file.
- `template-LEARNINGS.md` — updated template LEARNINGS stub with a
  new "Ensembling / blend methodology" section. Drop into
  `template/LEARNINGS.md`.
- `methodology-from-irrigation.patch` — `git format-patch` output;
  apply with `git am < methodology-from-irrigation.patch` for a clean
  commit on the destination repo.

## How to land (recommended)

```bash
# In a separate checkout of the playbook repo
cd /path/to/kaggle-claude-code-setup
git checkout -b claude/methodology-from-irrigation origin/main
git am /path/to/Kaggle-irrigation-water/legacy/playbook-patch/methodology-from-irrigation.patch
git push -u origin claude/methodology-from-irrigation
# then open a PR against main
```

Or manually replace files:

```bash
cp CLAUDE.md             /path/to/kaggle-claude-code-setup/CLAUDE.md
cp template-LEARNINGS.md /path/to/kaggle-claude-code-setup/template/LEARNINGS.md
```

## Content summary

The playbook rewrite adds the following sections on top of the
existing Day 1 / Workflow / Methodology scaffold:

1. **Git as the only communication channel** — every artifact
   commits, `oof_*.npy` + `test_*.npy` are first-class outputs.
2. **Session arc** (Setup → Floor baselines → EDA → Heuristics +
   domain → Out-of-box trees → Advance-the-front → Closeout) with
   time budgets per phase.
3. **Optimization gradient discipline** (simulated-annealing analogy)
   — monotonically decreasing improvement bar. Phase-by-phase table
   of "interesting lift" thresholds.
4. **Prune complexity between phases** — what to cut when moving
   to the next stage.
5. **Artifacts for blending** — OOFS.md manifest, gitignore allowlist
   pattern.
6. **Blending & ensemble methodology** — 12 rules from the irrigation
   competition:
   - Fixed-baseline-bias sweep as pre-LB filter
   - Real LB delta ≈ 1/3 OOF delta when stacking tuned-on-tuned
   - Gap shrinkage = honest architectural signal
   - "Ignoring a feature class" > "using it differently"
   - Rank/Borda dominated by prob/log space
   - Jaccard necessary but not sufficient
   - Specialists need 20–80 % minority class
   - Don't augment specialist training with clean data
   - Deterministic > learned at OOF parity
   - Training-distribution engineering ≠ inference routing
   - Shift-target framings collapse at ≥95 % majority
   - Feature-subspace bagging needs pool ≥ 3× subset size
7. **Daily-log format** (goal/changed/result/read-out/next-bet)
8. **Hypothesis board structure** (current best / open bets / ruled
   out / parked).
9. **Anti-patterns to refuse** (LB without OOF gate, retuning bias
   on layered blends, etc.).
10. **Session-close checklist.**
