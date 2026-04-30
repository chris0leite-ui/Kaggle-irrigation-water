# 2026-04-30 — T6 directional compose LB result: 0.98121 (40th saturation)

## Mechanism

Caruana 2004 diversity-penalized forward selection over the 14-component
LB-validated bank, plus directional restriction to H→M direction only.

**Construction** (from `scripts/T6_diversity_greedy.py` +
`scripts/T6_directional_compose.py`):
- Greedy step: `score(m) = macro_recall(blend(ens, m)) − β · max_jaccard(m, ens)`
  with β = 0.005
- Path picked at β=0.005:
  v1 → +xgb_nonrule (α=0.15, jac 0.652) → +xgb_metastack (α=0.30) →
  +recipe_pseudolabel_seed7labeler (α=0.05) →
  +sklearn_rf_meta_natural_r10_with_tier1b (α=0.05) →
  +tier1b_greedy_meta (α=0.30) → +realmlp (α=0.10)
- TRAIN OOF tuned macro: 0.981105 (vs v1's 0.980646 = +0.000459)
- Direction-precision analysis on 1235 TRAIN OOF disagreement rows
  vs 4b OOF analog:
  - H→M: T6 wins 251/263 (**95.4% precision**, well above 92% break-even)
  - M→H: T6 wins 53/591 (9% precision; W3_MHonly already saturated this)
  - L→M: 57/103 (55% — tossup)
  - M→L: 161/278 (58% — tossup)
- Compose: take 4b base + T6's H→M flips only. 45 new H→M flips on test.
- TRAIN OOF projected macro lift: +0.000175 over 4b OOF analog (0.980828 → 0.981003)

## LB result

**LB public = 0.98121**

- Δ vs LB-best 4b 0.98150 = **−0.00029** (regression)
- OOF→LB carryover ratio: −1.66× (-0.00029 / +0.000175)
- Back-calculated test-side H→M precision: ~77% (well below 92% break-even)
  - macro_delta = -0.00029 = (45 × p − 45 × (1−p) × N_M/N_H) / (3 × N_M)
  - solving: p ≈ 0.77

## Diagnosis (40th saturation confirmation)

The TRAIN OOF→LB transfer asymmetry is now structurally explained:

**TRAIN OOF bank-precision overestimates test-side** because TRAIN OOF rows
have v1 IN the bank, biasing bank-majority/bank-mean toward v1's own
argmax. On TEST, independent rows are evaluated by the same bank, but
without the v1-anchoring bias.

This reproduces the failure mode documented across N5b family (LB -0.00039
to -0.00106), R2/R5 a045 (LB -0.00098), classw (LB -0.00011), D 3-meta
(LB -0.00021), mlp_metastack (LB -0.00021), and others.

**Portable rule** (LEARNINGS.md candidate): "Caruana diversity-penalized
forward selection on a saturated meta-stacker bank produces TRAIN OOF
direction-precision figures that are ~15-20pp inflated over test-side
precision when the anchor model is itself in the bank. The 95% TRAIN
OOF precision becomes ~77% test precision, well below the 92% H→M
break-even, producing a clean LB regression. To validate: LEAVE the
anchor OUT of the bank when computing direction-precision proxies on
TRAIN OOF — that's the only honest measurement on this problem."

## Files

- `scripts/T6_diversity_greedy.py` — diversity-penalized greedy
- `scripts/T6_emit_candidate.py` — test-side blend emission
- `scripts/T6_compare_to_4b_oof.py` — direction-precision analysis
- `scripts/T6_directional_compose.py` — H→M-only override + emit
- `submissions/submission_T6_directional_4b_plus_t6_hm.csv` (LB 0.98121)
- `scripts/artifacts/T6_*_results.json`

## Companion closures (T1-T5 from same session)

- T1 (LLM judge): blocked, no API key in env
- T2 (conformal-certified override): TRAIN OOF blind spot (only 7 analog rows)
- T3 (IP-argmax under prior): redundant with LP cap (already saturated 2026-04-26)
- T4 (perturbation stability): low-impact V1 + V2 same blind spot as T2
- T5 (2-stage pseudo): consensus filter too tight, stage-2 cannot learn boundary
