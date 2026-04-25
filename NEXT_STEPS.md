# Next steps

What surfaced from reducing the LB-best 4-stack (LB 0.98094) to a 2-script
recipe + pseudo + blend (LB 0.97998), and where to push from here.

## Lessons

Fold-std noise on this competition is σ≈0.00088. Re-reading the LB ladder
through that lens:

- **Most of the LB ladder above recipe is sub-σ.** Recipe → 2-way → 3-way →
  3-stack → 4-stack adds 0.00059 / 0.00007 / 0.00003 / 0.00086. Three of
  those four steps are below 0.1σ — indistinguishable from public-LB churn.
  Only the meta-stacker step (~1σ) was real signal.
- **Compute spent per σ collapses fast past recipe+pseudo.** The LB-best
  used ~30× the compute of the 2-way (5 h CPU + GPU vs ~100 min CPU) for
  ~1.1σ. Diminishing returns started immediately after the pseudo step.
- **OrderedTE is the irreplaceable atom.** Dropping OTE → −5.7σ collapse.
  Dropping combos / digits / orig_stats — each ≤0.18σ individually. The
  recipe's identity is "OTE on many keys"; FE blocks just feed keys.
- **Diversity didn't pay.** 14 NN-family probes were each null; RealMLP at
  α=0.20 added 0.03σ. The "blend orthogonal models" CV folklore broke
  here — the deterministic-rule core is fully captured by trees + OTE,
  leaving nothing for a second function class to fit.
- **Pseudo-label is the one cheap, real lever.** +0.00059 LB / ~0.7σ for
  one extra training run. Best LB-per-line-of-code in the whole stack.

## Ideas to progress, ranked by σ / effort

1. **Stage-2 pseudo with the 2-way blend as labeler.** Stage-1 used recipe
   (LB 0.97939) as labeler. The 2026-04-21 stage-2 attempt failed because
   its labeler was a regressor (hybrid_v3 LB 0.97352). The simplified
   2-way blend (LB 0.97998) is now the strongest cheap labeler — expected
   pseudo purity rises from ~99.5% to ~99.7% on rare-class boundary rows.
   Single new script (`recipe_pseudolabel_stage2.py`); compose existing
   `build_pseudo_subset` + `run_cv` against the 50/50-blended test probs.
2. **Replicate the meta-stacker with 5 inputs, not 62.** Per the original
   meta-stacker's perm-importance, most of its 62 OOF components contributed
   near-zero. A heavy-reg XGB stacker on `[recipe_oof, pseudo_oof,
   dgp_score, sm_dist, rf_dist]` is ~50 lines. If it captures a meaningful
   slice of the +0.00086 meta-stacker lift, we reproduce ~LB 0.98094 from
   a sub-1000-line repo.
3. **Sweep τ honestly on OOF.** τ=0.98 was inherited from the V10 kernel
   and never tuned on this dataset's error geometry. Sweep τ ∈ {0.94,
   0.96, 0.97, 0.98, 0.99} gating on OOF-tuned bal-acc, not LB. Cheap
   (5 retrains) and a +0.0001–0.0003 lift would still be within fold-std
   but reproducible.
4. **Re-estimate σ before chasing more.** Run the 2-way pipeline under
   three independent fold seeds (42, 7, 123), measure the spread of
   OOF-tuned bal-acc, pin σ_private from that. Single biggest lesson the
   ladder taught us: we kept treating ~0.0003 deltas as real. Calibrate
   the noise floor, then only chase >1.5σ moves.

## Idea #1 result: null at +0.10σ

Ran 2026-04-25. Full chain timings: recipe ≈41 min, stage-1 ≈51 min,
stage-2 ≈49 min, blend <5 s.

| Step                                  | OOF tuned | Δ vs prior | σ-equiv |
|---------------------------------------|-----------|-----------:|--------:|
| Stage 1: recipe_full_te               | 0.97967   | —          | —       |
| Stage 2: stage-1 pseudo (recipe lblr) | 0.97993   | +0.00026   | +0.30σ  |
| Stage 3: stage-2 pseudo (2-way lblr)  | 0.98002   | +0.00009   | +0.10σ  |

Mechanism worked exactly as predicted — 2-way labeler passed
**+3,145 more rows** through τ=0.98 (84.93% vs 83.76%) — but the OOF
moved by 0.10σ, indistinguishable from fold noise. Stage-2's
fold-std halved (0.00128 → 0.00063), so the model is *more stable* of
equivalent quality, not better.

Test predictions diverge on only **371 / 270,000 rows (0.137%)** vs
stage-1: net +133 High, −46 Low, −87 Medium. Whether those flips
land favourably on private LB is a coin-flip at this magnitude.

**Verdict: idea #1 ladder is exhausted.** Self-training on this
particular DGP saturates at the stage-1 labeler. The +0.00026 stage-1
gain stays the cheap+real lever; iterating further produces sub-σ
churn.

Verification side-effect: rebuilt `submission_recipe_pseudolabel_blend.csv`
is **byte-identical** to the LB-confirmed reference
`submission_recipe_greedy_recipe_pseudolabel.csv`. The simplified
pipeline reproduces LB 0.97998 to the exact bit.

## Next: idea #2 (5-input meta-stacker)

Heavy-reg XGB stacker on
`[recipe_oof, pseudo_oof, dgp_score, sm_dist, rf_dist]` — 5 features in,
3 probs out, 5-fold StratifiedKFold(seed=42). Target the +0.00086 LB
that the original 62-component meta-stacker captured. ~50 lines, single
new script, no new training data needed (we now have all required
OOF arrays on disk).
