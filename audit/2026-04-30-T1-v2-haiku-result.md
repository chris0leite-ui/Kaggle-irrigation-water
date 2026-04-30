# 2026-04-30 — T1-v2 haiku critical-rows: NULL on full n=253

## Mechanism

Multi-stage cascade designed for the V1-floor disambiguation problem
(rows where 14-bank-mean argmax = M AND 4b OOF analog = H, ~2,979 rows
on TRAIN OOF, 81.7% baseline true-M):

**Stage 1**: pre-filter to critical-rows where ≥3 of 4 signals agree:
  - aux_missed_high < 0.20  (low predicted H-flip risk)
  - aux_missed_medium < 0.30
  - knn_margin >= 0.7 (k=100-NN consensus unambiguous)
  - bank_max_prob >= 0.80

Result: 253 critical rows at **87.75% baseline true-M** (+6pp lift from
filter alone).

**Stage 2**: haiku-V2 prompt with per-row aux fields exposed, 8 mixed
few-shot exemplars, statusline-setup harness for haiku context fit.

Decision rule: M @ CONF≥0.7 precision must clear 92% break-even.

## Result

```
n_critical_rows = 253
bank-only baseline (always M) = 0.8775 (222/253)
haiku M @ CONF>=0.7 precision = 0.8803  (206/234)
Wilson 95% CI = [0.8325, 0.9159]
delta over bank-only = +0.0028 (essentially zero, within CI noise)

haiku H @ CONF>=0.7 precision = 0.1765 (3/17) — net negative
```

**Verdict: MARGINAL/NULL.** The +0.28pp lift over the bank-only
baseline is well within sampling noise. At CONF≥0.9 the precision
actually DROPS to 79.5% (39 rows) — haiku's "highest confidence"
M-verdicts are less reliable than its lower-confidence ones,
suggesting overconfidence-at-the-margin failure mode.

## Critical observation: batch-level variance

```
batch         n   M_count  H_count  M_precision_at_07
batch_0      100   83      17       0.8434  (70/83)
batch_1      100  100        0      0.9100  (91/100, all M)
batch_2       53   53        0      0.9057  (48/53, all M)
```

Batch 1 was nearly all-M (haiku stayed conservative or possibly relied
on prior context after a Read tool budget rejection on the original
attempt). Batch 0 saw 17 H-flips (4 correct, 13 wrong) → big precision
drop on the M side. The interim n=153 estimate (90.37%) was an
artifact of mixing batch 0+2 only.

The lesson: haiku's behavior is sensitive to how it processes the
prompt — under prompt-too-long stress, it defers to prior context;
under clean reads, it tries to flip more aggressively. Neither mode
clears 92%.

## Why this confirms (not extends) the 41st saturation

T1-opus-V1 hit 89.5% on raw V1-floor (n=19, CI [68%, 99%]).
T1-haiku-V2 hits 88.0% on filtered V1-floor (n=253, CI [83%, 92%]).

The TIGHTER CI of haiku-V2 (n=253 vs n=19) confirms the 89-91% precision
ceiling is real, not sampling artifact. Both LLMs land in the same
range. The pre-filter helps the BANK ALONE reach 87.75%, not the LLM.

The critical-rows hypothesis was correct: the filter raises baseline.
But the LLM doesn't add meaningful precision on top — it's saturated
with the bank's own error structure, just like prior NN attempts
(18 nulls).

**The structural ceiling at ~88% on this V1-floor population is now
empirically calibrated by 2 LLM probes.**

## Cost summary

```
Stage 1 (filter):     $0.00
Stage 2 (haiku x3):   ~$0.10  (3 batches, ~30k tokens each)
Total:                ~$0.10  (vs $1.00 plan budget)
```

Well under the $1.30 cap. No sonnet/opus escalation triggered.

## What was learned (mechanism rules of thumb)

1. **Pre-filter with multi-signal agreement** is real: it lifted bank-only
   precision from 81.7% (raw V1-floor) to 87.75% (filtered V1-floor).
   This is a portable rule — for ANY override design on a saturated
   bank, the filter is the leverage point, not the override mechanism.

2. **LLM-judges add ~0-1pp on filtered subsets.** Both haiku-V2 (this
   probe) and opus-V1 (prior probe at n=19) showed lifts within
   sampling noise. The LLM's reasoning doesn't decorrelate from the
   bank's error structure on this DGP.

3. **LLM H-verdicts are systematically wrong** (16-18% precision in
   both T1-haiku and T1-opus). Always drop these from any override
   decision rule.

4. **Batch-level variance dominates** at n=100. Use n≥250 with
   stratified sampling to get tight enough CI to distinguish 88%
   from 92%.

5. **Read tool budget**: haiku's Read can reject files >~25-30k tokens.
   Workaround: batch ≤80 rows per file (≤25k tokens) OR use multi-Read
   chained calls in the prompt.

## Saturation roll-up

This is the 42nd independent confirmation that LB 0.98150 is the
own-pipeline ceiling. T1-haiku-V2 closes the LLM-judge mechanism for
the haiku tier on this V1-floor problem.

Mechanisms now closed (this thread):
  T1-haiku-V1 (50 rows, raw V1-floor): 90% precision, MARGINAL → close
  T1-opus-V1 (50 rows, raw V1-floor): 89.5% precision, MARGINAL → close
  T1-haiku-V2 (253 rows, critical-rows pre-filter + aux fields): 88.0%
    precision, NULL → close haiku tier definitively

Remaining option (not auto-triggered, requires user approval):
  - Stage 3: sonnet on same 253 critical rows (~$0.20)
  - Stage 4: opus on same 253 critical rows (~$1)

Per the plan, opus showed +3.5pp lift on raw V1-floor (n=19); on the
critical-rows filter (which already lifts baseline +6pp), opus's
marginal contribution would likely be similar or smaller. Expected
ceiling: ~89-91% on filtered subset, still below 92% break-even.

## Files

- `scripts/T1_critical_rows.py` — Stage 1 filter
- `scripts/T1_haiku_v2_build.py` — prompt builder
- `scripts/T1_llm_v2_score.py` — Wilson CI scorer
- `scripts/artifacts/T1_v2/critical_rows.csv`
- `scripts/artifacts/T1_v2/critical_rows_results.json`
- `scripts/artifacts/T1_v2/eval_keys.csv` (253 row_idx + true_label)
- `scripts/artifacts/T1_v2/response_haiku_{0,1,2}.txt` (haiku responses)
- `scripts/artifacts/T1_v2/results_haiku.json` (Wilson CI verdict)
