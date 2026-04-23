# Code review — final-submission audit (2026-04-23)

Branch `claude/plan-code-review-NwqHE`. Scope: the two final-selection
candidates and every script / OOF array they depend on.

## TL;DR

Both candidate submissions are **safe to final-select**. Every P0 check
passed. Zero leaks, zero fold mismatches, zero NaN/inf in any OOF or
test array used by either candidate, and both CSVs reproduce
byte-exactly (MD5 match) from the committed `.npy` arrays.

- **LB-best** `submissions/submission_xgb_dist_digits_tuned.csv`
  (OOF 0.97449 / **LB 0.97468**) — MD5 `e0d024d92588fdea30fe104985128ea2`.
- **Safe fallback** `submissions/submission_greedy_nonrule_blend.csv`
  (OOF 0.97421 / LB 0.97352) — MD5 `37e18a2df7e648fb8240ccc2c6b02698`.

Nothing below is blocking. P1 / P2 items are hygiene.

## P0 — could break final submission

1. **Submission format parity vs `data/sample_submission.csv`** — clean.
   Both CSVs: 270 000 rows, columns `[id, Irrigation_Need]`, `id` in the
   same order as sample, labels `Low/Medium/High` with correct casing.
   Digit sub: 159 779 Low / 99 432 Medium / 10 789 High. Greedy+nonrule
   sub: 159 814 Low / 99 085 Medium / 11 101 High. Both within the
   expected distribution given prior (58.7/37.9/3.3 %) + log-bias push.
2. **Digit-extraction math** (`scripts/digit_features.py`, used by
   `scripts/xgb_dist_digits.py`) — correct. Unit-tested every digit
   position in ±3 range on 123.456 → all match the header spec. Float
   rounding defended by `+ 1e-9` before `floor` (0.3 → digit 3 at
   position −1 as expected, not 2). `drop_zero_variance` correctly
   drops based on train uniqueness and mirrors the drop on test; on the
   real 1000-row probe, kept 13 / 21 digit cols, train and test end
   with identical digit-column sets. All digit values land in `[0, 9]`.
3. **End-to-end reproducibility** — both CSVs regenerated from the
   committed `.npy` arrays and the log-bias stored in
   `xgb_dist_digits_results.json` / `greedy_binhigh_minimal_results.json`
   reproduce byte-exactly (see TL;DR MD5s). `test_greedy_blend.npy`
   also reconstructs with max abs diff 0 from its
   `routed_v3 + spec_678` components using `build_hybrid_v3` + 3-way
   log-blend weights (0.45, 0.40, 0.15). **One caveat (non-blocking):**
   `binary_high_head.py` (which produced `oof_xgb_bin_high.npy` →
   consumed by `greedy_binhigh_minimal.py` → `oof_greedy_blend.npy` →
   `submission_greedy_nonrule_blend.csv`) lives at
   `legacy/null/binary_high_head.py`, not `scripts/`. Full regeneration
   from raw data needs `git mv legacy/null/binary_high_head.py scripts/`
   first. Documented in `legacy/INDEX.md`. The committed OOF makes
   this purely theoretical; both candidates reproduce without it.
4. **Fold-seed parity** — every OOF producer feeding the two
   candidates uses `StratifiedKFold(n_splits=5, shuffle=True,
   random_state=42)` on the full 3-class `y`. Canonical va-idx hashes
   match across producers. Row-level verification:
   `xgb_dist_routed_v3` has exactly 271 444 rule-routed rows, which
   exactly equals the count of rows with `dgp_score ∈ {0,1,2}`;
   per-fold argmax bal_acc for `xgb_dist_digits` lies in
   `[0.96402, 0.96535]` (tight, consistent with a 5-fold split);
   `xgb_spec_678` per-fold in-domain bal_acc in `[0.9498, 0.9531]`.
   Nonrule uses the same split. **No fold mismatches.**

## P1 — could invalidate OOF conclusions

1. **Target-encoding leakage** — only one TE script exists in the
   current tree: `scripts/benchmark_te_orig.py`. TE stats are computed
   from the 10 k original dataset (`compute_te_from_source(orig,
   y_orig, …)` at line 173 / 187) and **never from synthetic train
   labels**, so leak-free by construction. Its output
   (`oof_lgbm_te_orig.npy` / `test_lgbm_te_orig.npy`) is NOT used by
   either candidate, so even a hypothetical bug there wouldn't affect
   final selection. Other TE scripts referenced in `CLAUDE.md`
   (`te_targets*.py`, `benchmark_te_oof.py`, `te_xgb_regression.py`)
   live on an un-merged branch — not present here.
2. **Log-bias coord-ascent scope** — correct. `common.tune_log_bias`
   optimises on the full OOF array (not per-fold), with wide High-class
   grid (`-3..+6`, optimum ≈ +3.4). Every usage in the critical path
   either (a) tunes once on a fixed baseline OOF and freezes the bias,
   or (b) passes that frozen bias into downstream α sweeps. Concrete
   flow for the LB-best pipeline:
   - `greedy_binhigh_minimal.py:108` tunes bias on `oof_greedy_blend`
     once → saved in `greedy_binhigh_minimal_results.json`.
   - `nonrule_features_only.py:189` loads that bias and uses it UNCHANGED
     across every α in the log-blend sweep (fixed-bias protocol).
   - `blend_digits.py:62` does the same (would-be digit × greedy+nonrule
     blend; produced a strictly-worse candidate not selected).
   - `xgb_dist_digits.py:62-86` tunes bias on its own OOF once for the
     standalone submission and writes it to the results JSON.
   - `blend_ensemble.py` retunes bias per candidate blend (known
     selection-bias risk documented in CLAUDE.md); it was superseded
     by the fixed-bias pipelines and does not feed either candidate.
3. **Routing / specialist OOF index alignment** — clean.
   `xgb_specialist_678.py` calls `skf.split(X, y)` on full `y`, then
   intersects `tr_idx`/`va_idx` with `np.isin(dgp_score, (6,7,8))` —
   fold boundaries stay aligned with the main XGB OOFs. OOF is
   zero-filled outside the spec domain and `build_hybrid_v3` in
   `greedy_binhigh_minimal.py` reads it via the same spec-mask.
   Committed `oof_xgb_spec_678.npy` has exactly 573 878 zero-rows,
   which equals the count of out-of-domain rows. `xgb_dist_routed_v3`
   predicts on ALL val rows, then routes `score ∈ {0,1,2}` to a
   soft-clipped `rule_prob_low = [1-2e-9, 1e-9, 1e-9]` at OOF time;
   the 271 444 routed rows match the dgp_score mask exactly. Every
   dense OOF sums to ~1 per row, no NaN/inf.
4. **Balanced-accuracy metric** — every tuner uses
   `balanced_accuracy_score` (or `common.fast_bal_acc`, a vectorised
   equivalent). No `accuracy_score` mis-use in any tuner loop.

## P2 — code-quality hygiene

1. **Dead scripts in `scripts/`.** 23 of 32 scripts in `scripts/` are
   not on the critical path for either candidate (see `Scripts NOT
   reachable` in the audit). Most are live diagnostics or session-B
   analysis, a few are stale (e.g. `enumerate_integer_models.py`,
   `verify_integer_models_identical.py`, `hybrid_v3_reconstruct.py`,
   `session_b_*`, `refit_thresholds_synthetic.py`, `transfer_check.py`,
   `hybrid_routed_spec.py`). Moving these to `legacy/` would reduce
   cognitive load for the final reviewer. `legacy/INDEX.md` is
   well-maintained so additions are cheap.
2. **Orphan reference in a JSON**. `xgb_dist_nn_orig_results.json`
   references `submissions/submission_xgb_dist_nn_orig_blend.csv`
   which no longer exists. Harmless (the NN-on-orig lever was closed
   and the submission CSV was cleaned up), but the JSON still points
   at it. Low priority.
3. **`common.py` drift**. `common.py` has the shared
   `add_distance_features` / `tune_log_bias` / `log_blend` /
   `fast_bal_acc`, but earlier scripts predate it and redefine these
   locally (`benchmark_dist.py`, `benchmark_xgb_dist.py`,
   `benchmark_te_orig.py`, `nonrule_features_only.py`,
   `xgb_dist_digits.py`, `xgb_specialist_678.py`, etc.). The locally-
   defined versions are all equivalent — confirmed by reproducibility
   of both candidates — but the duplication is a long-term
   maintenance hazard. Don't touch the critical-path scripts (would
   invalidate the committed OOFs); new scripts should use `common`.
4. **Categorical mapping assumes no unseen test values**. Several
   scripts (`nonrule_features_only.py:113-116`, `xgb_dist_digits.py:114-117`,
   `xgb_specialist_678.py:130-133`, `xgb_dist_routed_v3.py:153-156`)
   build category maps from `tr[c].unique()` then `te[c].map(mapping)`.
   Unseen test values yield NaN, then `.astype("int32")` raises. Safe
   for this competition (fixed categorical vocab) but not defensive
   for reuse.
5. **Data directory is ephemeral**. `data/train.csv`, `data/test.csv`,
   and `data/sample_submission.csv` are all gitignored and were
   missing on session start; `./bootstrap.sh` fetched them in one
   step via `kaggle competitions download`. Documented at the top of
   `CLAUDE.md`.

## Reproducibility recipe

To regenerate `submission_xgb_dist_digits_tuned.csv` end-to-end from raw
data:
```
./bootstrap.sh
python scripts/xgb_dist_digits.py      # ~4 min
```

To regenerate `submission_greedy_nonrule_blend.csv`:
```
./bootstrap.sh
git mv legacy/null/binary_high_head.py scripts/
python scripts/xgb_dist_routed_v3.py   # ~12 min
python scripts/xgb_specialist_678.py   # ~4 min
python scripts/binary_high_head.py     # ~6 min
python scripts/greedy_binhigh_minimal.py  # seconds
python scripts/nonrule_features_only.py   # ~6 min
```

From committed `.npy` arrays only (no training):
```python
# For digits (candidate 1)
import json, numpy as np, pandas as pd
test = np.load('scripts/artifacts/test_xgb_dist_digits.npy')
bias = np.array(json.load(open('scripts/artifacts/xgb_dist_digits_results.json'))['log_bias'])
preds = (np.log(np.clip(test, 1e-9, 1)) + bias).argmax(1)
pd.DataFrame({'id': pd.read_csv('data/test.csv')['id'],
              'Irrigation_Need': [['Low','Medium','High'][i] for i in preds]}
            ).to_csv('sub.csv', index=False)
```
