# 2026-04-30 — final-day follow-ups: variants A/B/C, X1a, L2, TabNet

After LB 0.98140 was set + TC1 saturated the override pool, three
mechanism-distinct attempts and three scaffolds. All probes NULL on
4-gate; scaffolds untested.

## Variants A / B / C — 3 NULLs (commit `16b90cd`)

**Variant A — soft router blend** (per-row weight, not argmax flip)
on v1 RF natural ⊗ a1lgbm:
- NULL on 4-gate at every α
- Best gate-pass: α=0.20, Δ_OOF = −0.000010 (G2 pass, G1+G4 fail)
- Best Δ_OOF: α=0.30 +0.000030 (G2 fail, PCR_H −0.0006)
- Same Pareto-frontier closure as hard routing. v1 wins 62.4% of 1858
  OOF disagreements; any positive routing weight removes correct H
  predictions faster than it adds correct M/L.

**Variant B — ExtraTrees natural on v1 bank** (drop-in RF→ET swap):
- NULL standalone — tuned 0.98029 vs v1 RF 0.98063 (Δ −0.00034)
- ET 8× faster (35s/fold vs 290s) due to random split thresholds
- Test diff vs v1: 535 rows (more diverse than fold-seed variants
  150-300 row diffs). Standalone too weak to clear blend gate.

**Variant C — router-as-feature** (which-model gate prob added as
recipe FE col, not as blend weight):
- Δ_OOF = +0.00002 (sub-threshold), G2 PASS but G1 sub-+2e-4
- Standalone tuned ties recipe baseline. Tree splits at depth=4 +
  reg_alpha=5 absorb the new feature with near-zero gain.

## X1a prob-level geomean (commit `1827e31`)
Sweep w_raw ∈ {0.20…0.80}, geomean of rawashishsin (LB 0.98109) with
reconstructed LB-best 4-stack (LB 0.98094), tune log-bias per weight.
- Best: w_raw=0.30 → tuned OOF 0.98102 (vs LB-best 4-stack 0.98090
  with retune, +0.00012)
- 419 rows differ from B (LB 0.98140), direction RESHUFFLE
  (155 ADD-H + 137 REMOVE-H, net_H +18)
- Per binhigh + RESHUFFLE rules: projected LB ≤ 0.98094, **negative
  vs B's 0.98140**. Not LB-probed.

## In-flight scaffolds (untested as of deadline-day open)

**L2 — SupCon-NCM** (commits `e9a3aba` + `9c2548d`):
- SupCon contrastive embedding on 443-feature recipe FE
  (reuses p3_embed_propagate model unchanged)
- Mahalanobis nearest-class-mean with LedoitWolf-shrunk per-class
  covariance
- Decision rule: argmax of `softmax(log_likelihood_k)` under uniform
  prior — **Bayes-optimal under macro-recall by construction**
- ZERO post-hoc log-bias retune (this IS the calibration mechanism)
- Why orthogonal to 16+ NN nulls: distance-based decision rule, not
  softmax-CE+argmax. Eliminates the bias-retune leak channel that
  bounds all prior NN attempts.
- Submission emitter ready (`scripts/L2_emit_ncm_submission.py`);
  embedding training not yet run.

**TabNet kernel scaffold** (commit `4c6abd8`):
- TabNet (Arik & Pfister 2020) — sequential attention + sparse feature
  selection. Confirmed NEVER tested on this comp.
- 17th NN family attempt (after MLP variants, FT-T, TabPFN, DAE,
  RealMLP, Trompt, Mambular, KAN, TabM, etc.). All 16 prior NN nulls
  closed on the magnitude trap.
- Bayesian prior of clearing magnitude rule (errs ≤ 1.05× anchor):
  ~10%. Even if it clears, blend lift bounded by Pareto-frontier on
  recipe FE.
- Kernel scaffolded; not yet pushed.

## Hard guard committed (commit `005d26e`)
After re-recommending an already-LB-tested regressor 8 hours later,
added top-of-CLAUDE.md rule: **always run
`python scripts/lb_status.py | grep <filename>` before recommending
any candidate as "unprobed"**. Cost asymmetry severe — duplicate
submissions waste daily quota AND erode trust.

## Pending work referenced for future sessions
- **Override candidates emitted but UNPROBED** (in `submissions/`):
  - `submission_lbbest_overridden_by_3of4.csv` (looser k=3 majority)
  - `submission_recursive_k4_override.csv` (recursive application)
  - `submission_consensus_override_all_helpers.csv`
  Per TC1 saturation finding, all three project marginal at best
  (B's pool already saturates further override mechanisms).
- **L2 SupCon-NCM** if base embedding completes
- **TabNet kernel** if user authorizes Kaggle GPU spend at
  ~10% LB-lift prior

## LB budget remaining for final day
2026-04-30: 2/10 used (B + TC1), **8 LB submissions remaining** before
deadline. With override mechanism saturated and every blend variant
projecting null, marginal LB-probe EV is below variance noise. Lock
the final pair.
