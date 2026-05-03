# 06 — Final results

Pulled 2026-05-03 via `kaggle competitions leaderboard
playground-series-s6e4 --download` (4 315 teams) and
`kaggle competitions submissions` (per-CSV public + private).

## Public leaderboard final

| | Score | Rank | Notes |
|---|---:|---:|---|
| Leader (Kevin E R MILLE) | **0.98236** | 1 | +0.00017 above Cdeotte's kickoff 0.98219 |
| Pack at rank 100 | 0.98151 | 100 | +0.00037 above kickoff 0.98114 — the pack moved |
| Top-5% cutoff | 0.98151 | 215 | 5% of 4315 ≈ 215 |
| **Our PRIMARY** | **0.98150** | **226** | **top 5.24% — just outside the 5% target** |
| Last team strictly above us | 0.98151 | (rank 215) | tied pack of ~12 teams sits exactly between |

We landed **11 ranks below the top-5% cutoff**, missing the framework's
target by roughly two-tenths of a percentage point. The pack moved
+0.00037 in 10 days; we held our absolute score but lost relative
position.

## Our submissions: public vs private

84 submissions made (per LB CSV `SubmissionCount=84`). Highlights:

| Filename | Public | Private | Pub→Pri |
|---|---:|---:|---:|
| **`idea4b_selective_override` (PRIMARY)** | **0.98150** | 0.98051 | −0.00099 |
| `sklearn_rf_meta_natural` (HEDGE) | 0.98129 | 0.98047 | −0.00082 |
| `idea5_anchor_switch` | 0.98148 | **0.98058** | −0.00090 |
| `W3_MHonly` | 0.98127 | 0.98057 | −0.00070 |
| `bagginglr_natural_standalone` | 0.98106 | 0.98055 | −0.00051 |
| `4b_plus_w5_strict90` | 0.98143 | 0.98052 | −0.00091 |
| `98150_drop_lm` | 0.98148 | 0.98047 | −0.00101 |

**5 of our submissions beat PRIMARY on private LB.** The best private
score we made was `idea5_anchor_switch` at 0.98058 — which had a
−0.00002 *regression* on public LB and was rejected for that reason
(saturation log: "Idea 5 anchor-switch: 0.98148 LB −0.00002 vs PRIMARY").

## Spread compression

Private LB compressed our submission spread by 3×:

| | Min | Max | Spread |
|---|---:|---:|---:|
| Public | 0.97955 | 0.98150 | 0.00195 |
| Private | 0.97987 | 0.98058 | 0.00071 |

What looked like a 195bp difference on public was a 71bp difference
on private. Most of the lift we extracted was specific to the
public-LB slice.

## Public→private gap was wider than OOF→public

The OOF→public gap on calibrated mechanisms was ~5–10bp. The
public→private gap was 50–100bp. Two separate calibration anchors
diverged: OOF tracked public well, and *both* were optimistic vs
private by ~1 part in 1 000.

## Override-mechanism overfitting was the dominant story

The override family (Idea 4b + variants) gave us +6bp of public lift
over the underlying meta-stacker B (0.98140 → 0.98150). On private,
that lift evaporated:

| Mechanism | Public Δ vs B | Private Δ vs B |
|---|---:|---:|
| Idea 4b (108-flip override) | +0.00010 | +0.00005 |

The override flips ~108 rows of 270k test. With an 80/20 public split,
~22 of those flips landed on the public 54k slice. Our gating logic
was tuned against OOF and validated against public. The "selection of
which 108 rows" was itself implicitly chosen to help on public.

## What this comp's NEVER-GIVE-UP rule got right (and wrong)

The rule said "every plateau is bounded-prior evidence, not a ceiling
proof". In retrospect:

- **Right**: the override mechanism family broke the 0.98094 4-stack
  ceiling on public LB (+0.00056). That mechanism would not have been
  found if we had locked at 0.98094.
- **Wrong**: the +6bp of public lift it added on top of B was largely
  selection bias on the public split. The private "ceiling" was real
  at ~0.98050 — every Day-17/18 mechanism we tried clustered there.

The agent's saturation thesis was approximately correct *for private*
from around 0.98140 public onward. We continued chasing public lift
that didn't transfer.

## What about the leader

Kevin E R MILLE's public LB jumped from Cdeotte's 0.98219 (kickoff)
to 0.98236 in the last days of the comp. The pack moved +0.00037 in
the same window. We don't know what mechanism either group found —
that's the immediate **research-loop debt** we owe ourselves before
the next comp.

## Headline

We hit **top 5.24% public**, missed the 5% target by 11 ranks, and
**picked the wrong primary**: 5 of our own submissions had a higher
private score. Next-comp recommendations follow in
[07-next-comp-recommendations.md](07-next-comp-recommendations.md).
