# 2026-04-30 — T1 LLM-judge: 41st saturation, mechanism-level closure

## Mechanism

External-supervision LLM judge as the 4th axis of an override on top of
4b (LB 0.98150). Haiku reads agronomic features, computes the DGP rule
score, applies a documented NN-flip heuristic, returns FINAL label +
CONF. We override 4b only when LLM, 14-bank-majority, AND H→M direction
all agree.

Fire rule (4-axis):
  (1) llm_final ≠ 4b argmax
  (2) llm_conf ≥ 0.7
  (3) bank_mean.argmax = llm_final
  (4) (4b = H) and (llm_final = M)  — direction restriction

## Harness-overhead finding (orthogonal to the mechanism result)

The Agent tool harness injects ~388k tokens of fixed context per call
for `general-purpose` and `claude-code-guide` subagent types — over
haiku's 200k cap, producing immediate "Prompt is too long" rejections.

Workaround: `subagent_type="statusline-setup"` injects only ~12k tokens
of harness overhead (32x smaller). Haiku honors override-the-persona
prefixes ("ignore your statusline-setup persona for this single
message") and produces the requested ROW-block format reliably.

Cost at sonnet (388k overhead): ~$1.20/call, $6+ for 500 rows.
Cost at haiku via statusline-setup: ~$0.05/100-row batch, ~$0.25 total
for 500 rows. **24x cost reduction** vs the sonnet path.

Useful pattern for future LLM-judge attempts: any task that doesn't
need the heavy default subagent context can be routed through
statusline-setup with an instruction-override prefix.

## TRAIN OOF validation (the kill)

We can't compute the LLM filter on TRAIN OOF (no labels there), but
the 3-axis floor (bank=M & 4b=H) is a lower bound on T1 precision —
the LLM-confirmed subset is at most this precise.

```
V1 floor (bank_argmax=M & fb_oof=H): 2979 TRAIN OOF rows
  P(true=M): 0.8174   ← below 0.92 break-even
  P(true=H): 0.1823
  P(true=L): 0.0003
  TRAIN OOF macro after override: 0.975596  (delta -0.005220)

V2 floor (V1 + bank_max<0.85): 2820 TRAIN OOF rows
  P(true=M): 0.8124
  TRAIN OOF macro after override: 0.975633  (delta -0.005183)
```

Apply the T6-documented 15-20pp TRAIN-OOF→test asymmetry haircut
(v1 is in the bank, inflating bank-precision on TRAIN OOF):
  V1 projected test P(M): ~0.642
  V2 projected test P(M): ~0.637

Both **way below 0.92 break-even**. Even if the LLM lifted precision
by +10pp over the bank floor, the test-side haircut would erode
another ~15pp, leaving the candidate at ~75% precision — clean LB
regression territory.

## Test-side fire counts (informational only)

500 borderline test rows classified by haiku (top-500 from
`T1_select_borderline.py` — disagreement + DGP score-band + bank
max-prob<0.85):

  axis (1) llm != 4b:           292
  axis (2) llm_conf >= 0.7:     495
  axis (3) bank_maj == llm:     249
  axis (4) 4b=H and llm=M:      56

  ALL 4 axes fire: 56 rows H->M

LLM FINAL distribution: L=266, M=186, H=48.
Mean LLM_CONF: 0.868 (median 0.880).

T6 directional compose for comparison: 45 H→M flips, TRAIN OOF P(M)
on the analogous filter was 95% (overestimate), test-side P(M) ≈ 77%
(LB regression -0.00029). T1 has 56 H→M flips at TRAIN OOF P(M) of
**81.7%** — strictly worse than T6's 95% on the analogous floor —
projected to test ~64%. Expected LB delta: -0.0006 to -0.0010.

## 4-gate verdict (per CLAUDE.md ⚠️ DEFEND AGAINST LEAKAGE)

  G1 (Δ standalone OOF ≥ +0.0001):     **FAIL**  (-0.00522 OOF)
  G2 (per-class recall ≥ 4b - 5e-4):   FAIL (H-recall lost on 18% of flips)
  G3 (H→M precision ≥ 92% break-even): **FAIL**  (81.7% TRAIN, ~64% projected test)
  G4 (net_H direction-positive):        N/A — directional filter

NO LB PROBE. T1 joins the saturation list as the 41st confirmation.

## Why this is a structural finding, not just T1-specific

The bank-majority H→M filter, with no orthogonal data source, has
**~82% precision floor on TRAIN OOF**. The 14 bank components
collectively misclassify ~18% of "high-confidence M, against a 4b
H-call" rows as if they were M — but the truth is they ARE H, and
the bank just got it wrong because the same DGP-saturated tree
ensembles share error modes.

Any 4-axis filter that uses bank-majority as its 3rd-axis
"confirmation" inherits this 82% ceiling. The LLM as a 4th axis
adds an independent vote, but the candidate set (4b=H rows where
bank says M) is already a noisy set, not a clean one. Adding a 4th
filter on a 4-row-deep noisy stack doesn't push past the floor.

Mechanisms that COULD push past:
  - A 4th axis that looks at data the bank doesn't see (e.g., raw
    features in a non-tree way) AND has independent error modes
    from trees. The LLM in principle does this, but its ~82% bank-
    parity (most haiku FINAL = bank-majority on these rows)
    suggests it inherits much of the same error structure on
    synthetic data.
  - External labels (DGP-faithful rule on a held-out auxiliary
    dataset, NN-inversion of the host's labeller, etc.) — not
    something we have access to here.

## Saturation roll-up

This is the 41st independent confirmation that LB 0.98150 is the
own-pipeline ceiling for this team on this problem. T6 was the 40th
(directional compose, LB 0.98121 -0.00029).

Mechanisms now closed (this session):
  T1 — LLM judge (closed, 4-gate fails on TRAIN OOF, no LB probe)
  T2 — conformal-certified override (TRAIN OOF blind spot, prior session)
  T3 — IP-argmax under prior (redundant w/ LP cap, prior session)
  T4 — perturbation stability (low-impact, prior session)
  T5 — 2-stage pseudo (consensus filter too tight, prior session)
  T6 — directional compose (LB regression, prior session)

CLAUDE.md NEVER-GIVE-UP rule still applies — the next mechanism must
be structurally distinct from "filter selection on top of saturated
14-bank components". External supervision via LLM with on-device
priors is now empirically closed; mechanisms that introduce TRULY new
data (e.g. NN-inversion of the host's labeller, or distillation from
larger LLM with seeded agronomic facts) are the remaining horizon,
both speculative and effort-heavy.

## Files

- `prompts/subagent_llm_judge.md` — prompt template
- `scripts/T1_select_borderline.py` — 7,631 borderline test rows ranked
- `scripts/T1_format_batch.py` — haiku-friendly prompt builder
- `scripts/T1_smoke_build.py` + `T1_emit_batches.py` — batch emit
- `scripts/T1_parse_responses.py` — regex parser for ROW blocks
- `scripts/T1_compose_override.py` — 4-axis fire and CSV emit
- `scripts/T1_validate_train_oof.py` — TRAIN OOF V1/V2 precision check
- `scripts/artifacts/T1_responses_batch_{0..4}.txt` — raw haiku output (500 rows)
- `scripts/artifacts/T1_borderline_top500.csv` — selected rows (regen)
- `scripts/artifacts/T1_*_results.json` — per-stage metrics
- `submissions/submission_T1_llm_judge_override.csv` — candidate (DO NOT
  LB-probe; structurally below break-even per TRAIN OOF)
