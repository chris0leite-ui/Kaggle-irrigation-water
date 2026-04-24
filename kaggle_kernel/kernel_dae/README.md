# kernel_dae — SwapNoise Denoising Autoencoder (A2 / P1)

Kaggle GPU kernel for the Porto Seguro-style denoising autoencoder.
Mechanism: train an encoder-decoder MLP to reconstruct the ORIGINAL
feature row from a SwapNoise-corrupted version (p=0.15 per-cell swap
with a random value from the same column). The 128-d bottleneck layer
is extracted as a row embedding, fed as extra numerics to
`recipe_full_te.py`.

Label-unaware: trained on train + test + orig JOINTLY (910k rows).
This is architecturally decoupled from every prior label-supervised
NN we tried, so the embedding can encode DGP structure that
gradient-based supervised signals on the same features never reach.

## Files

- `dae_swapnoise.py` — kernel entry point
- `kernel-metadata.json` — Kaggle kernel config
- `.gitignore` — drops `__pycache__/`

## Configuration

- `SMOKE=0` (default) — full 910k rows, 30 epochs
- `SMOKE=1` — 40k-row subsample, 2 epochs (validation)

Architecture: 43 → 1024 → 512 → 256 → 128 (encoder, GELU + BN + dropout
0.1), mirrored decoder. 1.48M params. AdamW lr=1e-3 wd=1e-5, cosine
schedule with 10% warmup, batch 4096, grad clip 1.0, MSE reconstruction.

## Output artefacts (in `/kaggle/working/`)

- `oof_dae_embed.npy`  — (630000, 128) fp32
- `test_dae_embed.npy` — (270000, 128) fp32
- `dae_embed_results.json` — config + per-epoch reconstruction MSE

## Regenerating embeddings from scratch

```bash
cd kaggle_kernel/kernel_dae
kaggle kernels push
# wait ~3 min on P100
cd -
rm -rf /tmp/dae_out && mkdir /tmp/dae_out
kaggle kernels output chrisleitescha/irrigation-dae-swapnoise -p /tmp/dae_out

# cast to fp16 and move to scripts/artifacts/ (halves disk footprint;
# XGB's float32 cast upstream makes the cast stderr ~7e-5 invisible)
python3 -c "
import numpy as np
for name in ('oof', 'test'):
    a = np.load(f'/tmp/dae_out/{name}_dae_embed.npy')
    np.save(f'scripts/artifacts/{name}_dae_embed.npy', a.astype(np.float16))
"
cp /tmp/dae_out/dae_embed_results.json scripts/artifacts/
```

Then:

```bash
DAE_EMBED_PATH=scripts/artifacts/oof_dae_embed.npy \
  python3 scripts/recipe_full_te.py
```

Takes ~55 min on CPU; emits `oof_recipe_full_te_dae.npy`,
`test_recipe_full_te_dae.npy`, and `submission_recipe_full_te_dae.csv`.

## Why the fp32 embeddings are NOT committed

The fp16 arrays are 153 MB + 65 MB — the git remote's pre-receive hook
rejects pushes that large. The DAE kernel runs in ~3 min on Kaggle GPU,
so regeneration is cheaper than LFS-level storage.

Consumers on other branches: check whether `scripts/artifacts/oof_dae_embed.npy`
exists locally; if not, run the regen recipe above before attempting
blends.

## Production run results (seed=42)

- Wall: 3.2 min on P100
- 30 epochs, MSE 0.268 → 0.106 (plateau around epoch 22)
- Final loss: 0.10574
- Embedding distribution: mean ≈ 0, std ≈ 0.36, range [-2.58, 2.95]

See `scripts/artifacts/dae_embed_results.json` for per-epoch losses.
