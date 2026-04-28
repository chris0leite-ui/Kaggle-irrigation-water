# 07 — Execution results: Options A, B', C

Three drift-leverage options proposed in `06_three_options.md`,
executed 2026-04-28 on `claude/analyze-distribution-shift-A4uIv`.

**Note**: Option B was simplified from "concat orig with density-ratio
weights" (which would require ~100 lines of new orig-FE pipeline
extension into the recipe) to **Option B'**: per-row sample weight
multiplier `sw *= (1 + β × (1 − P(synth)))` on existing synth-only
training. Same AV signal source, mechanism still distinct from
ANCHOR_WEIGHT_ALPHA (which uses primary's max_prob, not P(synth)).

## Preflight: full-train + full-test P(synth)

- `scripts/dist_shift/av_full_predict.py` builds leak-free P(synth |
  row) on the full 630k train + 270k test using the AV classifier
  (trained on orig 10k + train-subsample 10k).
- For the 10k subsample: AV-OOF predictions are leak-free.
- For the remaining 620k + the 270k test: full-fit AV predictions
  (the AV classifier never saw any of these rows).
- Train P(synth) percentiles [1/25/50/75/99]: `[0.142, 0.504, 0.590, 0.660, 0.795]`
- Test P(synth) percentiles: `[0.143, 0.503, 0.590, 0.660, 0.795]` —
  **identical to train** (consistent with J3's train↔test AV-AUC 0.50).

## Strengthened diagnostic on full 630k

`av_predicts_flip_full.py`:

```
AUC(P(orig), flip) on full 630k = 0.5746
Cohen's d (P(orig), flip vs clean) = +0.335
Top-K precision (sort by P(orig) desc):
  K=100   → 5.00% precision (3.06× base)
  K=500   → 7.40% precision (4.52× base)
  K=5000  → 8.12% precision (4.96× base)  ← right at M↔H break-even 8.1%
  K=20000 → 6.22% precision (3.80× base)
```

Per-score AUC of P(orig) for flip:

| score | n      | n_flip | AUC    | comment                |
|-------|--------|--------|--------|------------------------|
| 1     | 115457 | 5      | 0.4955 | n_flip too small       |
| 2     | 122220 | 365    | 0.5093 | weak                   |
| 3     | 102157 | 4899   | 0.5227 | weak (dominant cell)   |
| 4     | 117837 | 1520   | 0.5806 | moderate               |
| 5     | 79203  | 274    | **0.7169** | strong             |
| 6     | 38416  | 1549   | **0.6103** | moderate-strong    |
| 7     | 15026  | 1360   | **0.6258** | moderate-strong    |
| 8     | 2680   | 330    | 0.168  | inverted (n=330)       |

Signal concentrates at scores 5/6/7. Score=3 (the dominant flip
band, 4,899 of 10,304 flips) is essentially random. This **predicts**
Option C's NULL.

## Option C — score=3 specialist with AV-score: NULL

`scripts/dist_shift/optC_score3_specialist.py`:

```
domain    = score=3 ∩ teacher_argmax=Low (n=101,392)
target    = (y == Medium); prevalence 4.28%
features  = 37 (raw nums + cats + dist + AV-score + teacher meta)
5-fold OOF AUC = 0.8195
prior 2026-04-26 spec_lm_v3 (no AV-score) AUC was 0.827
```

AV-score did NOT lift the score=3 specialist (slightly lower AUC,
within noise). Top-K precision peaks at 43% (K=100), but Wilson 90%
lower bound 0.368 < 0.393 break-even. **No conformal-feasible
operating point.**

This was predicted by the per-score AV-AUC: at score=3, AV is
essentially random (0.52). The signal lives at scores 5/6/7 instead.

## Option A — AV-score as recipe FE feature

(IN FLIGHT at write time. Production launched 09:11 UTC, 5-fold
seed=42 with EXTRA_AV_PSYNTH=1. ETA ~75 min wall.)

(Results to be filled in when production completes.)

## Option B' — orig-weight sample multiplier on synth rows

(Pending Option A completion to free CPU. Will run with
`ORIG_WEIGHT_BETA=1.0`. SMOKE GREEN: tuned smoke OOF 0.96378 vs
vanilla 0.96381 — wiring works, β=1.0 produces sample weight range
[0.64, 19.5].)

## Summary at write time

- **Option C closed NULL** at expected pattern (no signal at score=3).
- **Option A in flight**, expected NULL based on diagnostic AUC 0.5746
  on full train (modest signal that recipe XGB at depth=4 may absorb
  natively via its existing splits on `norain` + `Rainfall_mm` +
  rule indicators — which are the AV classifier's top-3 gain features).
- **Option B' wiring built**, awaiting A completion.

If all three close NULL, this becomes the **31st-33rd structural
saturation confirmations** at LB 0.98094, with a clean closure of
the "AV-shift signature as flip information" mechanism family.

If A passes the 4-gate filter, **ASK USER** before LB probe.

(Final-selection lock unchanged regardless of outcome. Two days to
deadline 2026-04-30.)
