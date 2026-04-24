# kernel_realmlp — A1 RealMLP-TD via pytabkit

Kaggle GPU kernel scaffolded on `claude/review-leaderboard-strategy-IMYgZ`.
Ports the mahoganybuttstrings RealMLP kernel (CV 0.97802 / LB 0.97685) —
first NN architecture in our test log that's purpose-built for tabular data.

## Why RealMLP

All 11 prior NN nulls (v5–v9 MLP, FT-Transformer, TabPFN, pretrain-FT,
NN-on-original, soft-distill, DAE) used from-scratch MLPs or generic
transformers. RealMLP-TD is different:
- `n_ens=8` parallel BatchEnsemble heads sharing weights via einsum
- PBLD (Periodic Basis with Learned Decay) numeric embedding
- smooth-clip scaler `x / sqrt(1 + (x/3)^2)`
- label smoothing with cosine schedule
- `flat_cos` LR schedule; scale-layer LR = 10× base LR

TabArena 46-benchmark: first NN consistently matching GBDT ceiling.

## Feature set (mahoganybuttstrings verbatim)

~64 features total:
- 11 raw numerics (kept as float32 → PBLD embedding path)
- 11 factorized numerics (embedding path)
- 8 factorized categoricals (embedding path)
- 15 pair combos of 6 rule-relevant features
  `{Soil_Moisture, Crop_Growth_Stage, Temperature_C, Mulching_Used,
  Wind_Speed_kmh, Rainfall_mm}` — filtered to drop combos where
  `nunique > N/2` (uninformative near-unique keys)
- Per-fold `TargetEncoder(target_type="multiclass", cv=5)` on the 15 combos
  → ~45 numeric TE cols

**NOT used**: our 443-col recipe OTE set, digit extraction (66 cols), or
ORIG mean/std (38 cols). RealMLP is sensitive to feature dimensionality
and mahoganybuttstrings hit 0.97802 CV with the smaller set; adding wide
recipe features is a second experiment if this one lands a blend
component.

## Pushing + running

From the repo root:

```bash
cd kaggle_kernel/kernel_realmlp
kaggle kernels push
# (later, after queue finishes)
kaggle kernels output chrisleitescha/irrigation-realmlp-pytabkit -p output/
```

Or smoke-test locally (requires pytabkit + GPU or CPU-fallback):

```bash
SMOKE=1 python kaggle_kernel/kernel_realmlp/realmlp_pytabkit.py
```

## Blend gate (runs locally after artefacts land)

```bash
# fetch outputs to scripts/artifacts/oof_realmlp.npy, test_realmlp.npy
python scripts/blend_realmlp.py   # (not yet scaffolded; uses common.tune_log_bias)
```

Gate rules (from CLAUDE.md blend heuristic):
- Fold-1 error Jaccard vs recipe_full_te + LB-best 2-way
  - `≥ 0.90` — abort (redundant with existing blend)
  - `0.85–0.90` — warn (blend lift capped ~+0.00015)
  - `< 0.85` — run all 5 folds, then fixed-bias blend sweep
- Fixed-bias α sweep vs LB-best: only emit submission if peak Δ ≥ +0.0002
- Expected LB if blend passes: +0.0005 to +0.0015

## Expected wall time

- Kaggle P100: ~45 min
- Smoke (N_FOLDS=2, 20k rows, n_epochs=3): ~3 min

## Failure modes to watch

- **pytabkit install failure** — fallback: manually install torch + pytabkit
  via `pip install pytabkit torch --index-url https://download.pytorch.org/whl/cu121`
- **GPU memory** — P100 has 16 GB; RealMLP at default batch ~1024 fits
  comfortably. If OOM, set `batch_size=512` in the CONFIG.
- **class_error vs bal_acc** — pytabkit defaults to classification error
  not balanced accuracy, so fold val metric will be near-zero (imbalanced
  prior). Safe to ignore; the post-hoc log-bias tune on OOF finds the
  macro-recall operating point.
