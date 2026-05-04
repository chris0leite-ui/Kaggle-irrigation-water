# 04 — What failed

48 saturation events at LB 0.98150. The expensive ones below — by
"expensive" we mean either burned LB slots, multi-hour compute,
or both.

## 1. The NN expedition (18 architectures, all null)

Tested: TabPFN-10k, RealMLP n_ens={1,2,4}, FT-Transformer, KAN, Mamba,
Trompt, TabM, ExcelFormer, narrow sklearn MLPs at 12 capacity points,
specialised heads on score-{6,7,8} subsets, training-data-routed MLPs.

All passed standalone OOF on smoke configs. None passed the blend
gate. The ceiling was *not* capacity — a 1M-param MLP plateaued at
the same score as a 50k-param one.

**Root cause**: the DGP is rule-structured. Boosted trees with
axis-aligned splits align with the rule's thresholds; an MLP learns a
smoothed approximation that misses the same boundary rows the trees
miss. NN-as-structural-match-to-DGP is a hypothesis worth testing
*cheaply* (sklearn MLP, smoke config) but not worth scaling on
rule-structured tabular DGPs.

**Cost**: ~3 days of agent attention; one Kaggle GPU kernel killed
at t+3h34min in CPU preprocessing (the pytabkit `n_ens=8` × `cv=5`
multiplier blew up wall time silently).

## 2. The 7 leakage incidents (~0.0045 LB)

| Date | Incident | LB Δ |
|---|---|---|
| 2026-04-23 | stage-2 pseudo-label (labeler+target same folds) | −0.00009 |
| 2026-04-23 | stacking-inflation ceiling (3+ blends OOF 0.98030 → LB ~0.97995) | flat |
| 2026-04-24 | soft-distillation student memorizes teacher OOF | −0.00148 |
| 2026-04-25 | LR meta v1 + v4 ET+kNN + P3 perturbed | −0.00103 / −0.00102 / −0.00139 |
| 2026-04-26 | DROP_DETERMINISTIC removed boundary-anchor rows | regressed |
| 2026-04-27 | R2 hybrid grid-selected (24-point grid → OOF inflation) | −0.00046 |
| 2026-04-28 | stacking feature leak (80% gain from circular meta-of-metas) | regressed |

Each of these had a positive OOF Δ that did not transfer to LB. The
4-gate filter + minimal-input-meta-test were developed in response.

## 3. Hyperparameter / architecture search nulls

- **Multi-task XGB**: OOF +0.00036 but bank inflated to 149 components
  → flagged as stacking-inflation, never probed.
- **R3 NN-distance override family**: 18th saturation confirmation,
  signal not LB-extractable.
- **Macro-recall surrogate XGB**: first G4 PASS in 25+ saturations,
  but LB null. The G4 PASS was real; the magnitude was below the
  resolution of the public-LB split (80/20 puts a hard floor on probe
  resolution at ~0.00005).
- **L1-L5 loss-function-ensemble** (focal, conformal, etc.):
  44th–47th saturations, all regressions.
- **bagginglr_natural standalone**: 48th saturation, 0.98106 LB.

## 4. Submission-budget burns

- **04-26 retry-loop incident**: a `until ... | grep -q "successfully
  submitted"` loop with a case-mismatched success marker burned 4
  redundant slots on `submission_v6_full_a350.csv` (07:09:31, 07:10:04,
  07:14:44, 07:15:22 — all returning the same deterministic 0.98012).
  3 net slots wasted; the loop's terminator never matched Kaggle's
  capital-S "Successfully submitted" string.
- **04-30 already-submitted re-recommendation**: agent recommended
  `submission_rawashishsin_k4_overridden.csv` as "highest-EV unprobed
  candidate". It had been probed 8 hours earlier at LB 0.98112
  (−0.00022 regression). Agent hadn't checked
  `kaggle competitions submissions`.

These two incidents drove the **submission-rule** entries in
CLAUDE.md (ask-first, never-loop, check-before-recommend).

## 5. Premature ceiling declarations

The agent declared the structural ceiling reached at multiple plateaus:
0.97097, 0.97296, 0.97352, 0.97468, 0.97581, 0.97939, 0.97998, 0.98005,
0.98008, 0.98094. **Every one of these was broken by a mechanism the
agent had previously labelled "skip on principled grounds"** —
override decision rules, RF on natural-cal bank, triple-consensus
gating, hand-coded selective flips.

The CLAUDE.md ⚠️ NEVER-GIVE-UP rule was added in response. It treats
saturation evidence as bounded (we tested *known* levers, not all
levers) rather than as a structural ceiling proof.

## 6. Things we never properly tried

By comp end, the agent's hypothesis board listed several mechanisms
tagged "skip on principled grounds":

- External LLM judge (rate-limit blocked locally).
- DGP-archaeology via NN inversion (host's NN architecture unknown).
- Public-CSV blending — banned by ⚠️ rule, but was the pack's likely
  mechanism for hitting 0.98114.

Whether any of these would have closed the 0.00069 gap to the leader
is unknowable without trying. The honest reading: the LB ceiling at
0.98150 is the ceiling of *our* mechanism set, not the ceiling of the
problem.
