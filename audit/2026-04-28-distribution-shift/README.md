# Distribution shift: 10k original vs 630k synthetic — research report

Written 2026-04-28 by `claude/analyze-distribution-shift-A4uIv`.
Question: how does the synthetic train distribution differ from the
10k original anchor dataset, and what does that imply for any further
"original-as-anchor" lever?

## TL;DR

The synthetic train has a **substantial distribution shift** from the
10k original, dominated almost entirely by **`Rainfall_mm` and the
rule-axis features it touches**. Class priors are unchanged; the shift
is in the joint feature space.

Three numbers tell the story:

| diagnostic                                | value           | comp ref                     |
|-------------------------------------------|-----------------|------------------------------|
| AV AUC, train ↔ test                      | **0.50247**     | J3, 2026-04-25 (no shift)    |
| AV AUC, **orig ↔ synth-train**            | **0.69690**     | this report                  |
| AV AUC, train ↔ test (cdeotte / public)   | ~0.50           | published kernels            |

The synth has shifted Rainfall up by **+210 mm mean** (Cohen's d =
+0.315). The shift propagates to every class (per-class d ≈ 0.25 Low,
0.42 Medium, 0.40 High) and reshapes the rule's score distribution
(synth has +5.4 pp at score=1 and **−6.9 pp at score=3** — the dominant
flip-prone band).

This is consistent with — and explains — the entire history of
"original-as-anchor" failures in the comp log:

- transfer-check (2026-04-20) bal_acc 0.96278 vs synth 0.97097 = 0.00819 gap.
- NN-on-original (2026-04-22) all 5 archs collapsed to rule ceiling.
- TE-from-original (2026-04-22) argmax-equivalence theorem.
- soft-distill recipeonly (2026-04-25) gap +0.00201, distill family closed.
- W7 NN-to-original (2026-04-26) min Euclidean distance 1.33, no
  near-duplicates → **first measurement** that synth is not a perturbation
  of orig anchors.

## Files in this folder

| file                       | what                                              |
|----------------------------|---------------------------------------------------|
| `01_marginal.md`           | per-column shift (KS + Cohen's d + chi-square)    |
| `02_av.md`                 | adversarial-validation result + gain ranking      |
| `03_class_conditional.md`  | class priors, per-class shifts, DGP-score histogram |
| `04_flip_manifold.md`      | per-score / per-cell flip rates, flip directions   |
| `05_implications.md`       | what this means for the current pipeline + LB push |

## Reproducibility

```
python3 -m scripts.dist_shift.marginal
python3 -m scripts.dist_shift.av
python3 -m scripts.dist_shift.class_conditional
python3 -m scripts.dist_shift.flip_manifold
```

Outputs land in `scripts/artifacts/dist_shift/`. Wall ~3 min total
on a 16-core CPU container.

## What this is NOT

- It is **not** a new lever to push past LB 0.98094. With 30+
  documented saturation confirmations and the leakage-defense rule
  on this branch, no proposal is made for an LB probe. This is
  diagnostic / explanatory.
- It is **not** a re-run of the 2026-04-21 DGP residuals EDA.
  That EDA was within-synth (score=3 flip rows vs clean rows). This
  is between-source (orig vs synth-train at the joint level).
