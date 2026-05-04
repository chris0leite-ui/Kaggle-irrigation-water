# 01 — Overview

## The result

| | Score | Notes |
|---|---|---|
| **Our final LB** | **0.98150** | `submission_idea4b_selective_override.csv` |
| Hedge candidate | 0.98129 | `submission_sklearn_rf_meta_natural_v1_lb98129.csv` |
| Pack at rank ~100 | 0.98114 | huge tie at exactly this score |
| Leader (Chris Deotte) | 0.98219 | +0.00069 above us |
| First submission | 0.96972 | tuned LGBM, rank 726 / 2357 |

We finished above the rank-100 cutoff — top ~5% public LB. The gap to
the leader is small in absolute terms (~7 parts in 10 000) but
qualitatively meaningful: it's the gap between a hand-built recipe +
override stack and whatever Cdeotte was doing.

## Calibration ladder

OOF (5-fold stratified CV) → public LB transfer table:

```
recipe_full_te                       OOF 0.97967 → LB 0.97939   gap +0.00028
3-way multi-seed                     OOF 0.98029 → LB 0.98005   gap +0.00024
LB-best 3-stack                      OOF 0.98061 → LB 0.98008   gap +0.00053
LB-best 4-stack (tier1b_greedy_meta) OOF 0.98084 → LB 0.98094   gap −0.00010
v1 RF natural standalone             OOF 0.98063 → LB 0.98129   gap −0.00066
2-OTHER raw+tier1b k=2 unanimous (B) OOF 0.98088 → LB 0.98140   gap −0.00052
Idea 4b triple-consensus (PRIMARY)   OOF ~0.98088 → LB 0.98150   gap −0.00062
```

Key reads from the ladder:

- **OOF-LB gap is tight (<10bp) once we calibrate.** First submission
  showed −0.00125, well within one fold-std. After that we trusted
  the 5-fold OOF to within a basis point.
- **Negative-gap entries (LB above OOF) are override-family.** They
  come from a different mechanism (14-bank majority + selective
  flips) than the stacking metas. The negative gap is *structural to
  the override decision rule*, not a margin we could spend on stacks.

## Top-line numbers

- **Days**: ~10 (2026-04-20 → 2026-04-30)
- **Commits on `main`**: 109
- **Submissions used**: ~50 of (10/day × 10 days) = 100 budget
- **Saturation events logged**: **48**
- **Leakage incidents (LB-paying)**: 7, total cost ~0.0045 LB
- **NN architectures tested null**: 18 (TabPFN, RealMLP, FT-T, KAN,
  Mamba, Trompt, TabM, ExcelFormer, etc.)

## What this comp was actually about

The host ran a closed-form integer rule on 6 features through a small
NN and labelled 900k synthetic rows. ~98.4% of synthetic rows match
the rule; ~1.6% are NN-flipped to a neighbouring class. Reverse-
engineering this DGP (closed-form, see `scripts/dgp_formula.py`) was
the single biggest piece of progress. The remaining 1–2% lift after
DGP came from:

1. **Recipe ladder** — feature engineering + target encoding moved
   us from raw LGBM (0.97097) to `recipe_full_te` (0.97939).
2. **Stacking** — 14-bank natural-cal meta family pushed to 0.98094.
3. **Override mechanisms** — hand-coded triple-consensus rules over
   committed CSVs took us the final 0.00056 to 0.98150.

NN families produced 18 nulls. None added orthogonal signal that
survived the blend gate.

## Where to read next

- Timeline of how we got there: [02-timeline.md](02-timeline.md)
- What actually moved the LB: [03-what-worked.md](03-what-worked.md)
- The expensive misses: [04-what-failed.md](04-what-failed.md)
- Where coordination broke: [05-coordination.md](05-coordination.md)
