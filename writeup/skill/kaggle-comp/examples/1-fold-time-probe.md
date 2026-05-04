# Example — 1-fold time-probe before GPU kernels

The single most expensive coordination mistake of the irrigation-
water comp: a Kaggle GPU kernel that ate 3h34min of CPU
preprocessing before training started, was killed at that point,
produced zero output, and burned a queued GPU slot.

## What happened

Day 4. The agent identified pytabkit's RealMLP as a promising
tabular NN. Public-kernel author claimed "~45 min on P100".

The agent's pre-flight sanity check:

```
Estimated wall time: ~45 min (cited from author)
n_folds = 5
n_ens = 8 (default)
Estimated total: 45 × 5 × 8 / 60 = 30 hours

Wait, that's not right — n_ens runs in parallel inside the kernel.
Let me re-estimate: 45 × 5 = 225 min ≈ 3.75 hours.

That's still over 1h. But P100 should be fast enough...
[pushed kernel anyway]
```

What actually happened:

- pytabkit `n_ens=8` runs **8 internal preprocessing passes** before
  any training (`TargetEncoder(cv=5)`).
- Per-fold preprocessing compounds: 8 × 5 = 40 internal CV passes.
- Sklearn / Lightning setup added another 30 min.
- **3h 34min of CPU before any GPU compute** started.
- Agent killed the kernel at that point. Zero output.

## The 1-fold time-probe rule

```bash
# Smoke (5 min cap):
SMOKE=1 N_FOLDS=1 N_ENS=1 SUBSAMPLE=50000 python kernel.py
# Verifies: imports work, shapes match, no permission errors.

# 1-fold time-probe (full data, fold 0 only, full features):
N_FOLDS=1 N_ENS=1 python kernel.py
# Measures: actual wall time on real hardware.

# Multiply for full config:
projected_wall = (1_fold_time × n_folds × n_ens) × 1.2  # 20% safety

if projected_wall > 60_minutes:
    # shrink: fewer folds, smaller n_ens, fewer epochs, subsample
    pass
```

For the RealMLP kernel, a 1-fold-probe would have shown:

```
1-fold preprocessing wall: ~26 min (with n_ens=1)
Full config: 26 × 5 × 8 = 1040 min ≈ 17 hours

DECISION: shrink. Either drop to n_ens=2 (4 hours), or skip
this kernel entirely.
```

The shrink decision saves the GPU slot AND saves the agent from
"waiting on a long job" for 4 hours of context.

## Generalised version

Don't trust *any* third-party "wall time" claim:

| Source | Reliability | Why |
|---|---|---|
| Public Kaggle notebook author claim | Low | Often optimistic, sometimes from a smaller config |
| Library docs | Low-medium | Doesn't account for `n_ens × cv` multipliers |
| Same-hardware 1-fold probe of YOUR config | High | Measured, not predicted |
| Your prior runs of the same library | Medium-high | Library version drift, hardware drift |

Only the third row is trustable enough to launch a multi-hour run on.

## The portable rule

```python
# Before launching any pipeline projected > 1h:
1. SMOKE=1 — 5-min sanity check
2. 1-fold time-probe on full data, full features, n_ens=1
3. Multiply by (n_folds × n_ens × 1.2 safety margin)
4. If projected > 60 min, shrink config (folds, n_ens, epochs, subsample)
5. If still > 60 min after shrinking, skip this pipeline
6. Re-confirm smoke + time-probe on the SHRUNK config before push
```

For Kaggle GPU kernels specifically:

- Always include the `SMOKE=1` env-var path in your kernel script.
- Prefer CPU pipelines for tabular work — most own-pipeline levers
  on tabular comps run fine on 12-16 core CPU under 2h.
- If a kernel is still in preprocessing at t+30min with no fold
  output, kill it.

## Why this rule deserves a guardrail

The cost asymmetry is severe:

- Cost of running a 5-min smoke + 30-min time-probe: ~35 min CPU.
- Cost of skipping it on a 4h kernel that fails: ~4h GPU + agent
  attention while it runs + zero output.

In the irrigation-water comp, this single missed gate cost ~4 hours
of agent context (waiting on monitors) and one queued GPU slot.
Adding a smoke + 1-fold-probe is the cheapest insurance available.
