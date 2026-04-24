"""SwapNoise Denoising Autoencoder (A2 / P1 from NEXT_STEPS).

Mechanism: the Porto Seguro 1st-place architecture, label-unaware. Train
an encoder-decoder MLP to reconstruct the ORIGINAL feature row from a
SwapNoise-corrupted version (replace each cell with prob 0.15 with a
random value from the same column, drawn from another row). Bottleneck
128-d layer is extracted as a row embedding.

Trained on train + test + orig JOINTLY (no labels used anywhere). The
embeddings are then fed as 128 extra numeric features to recipe_full_te's
XGB. Because the DAE is label-unaware and architecturally decoupled
from the XGB loss surface, the embeddings can encode DGP structure
(e.g. the host-NN's latent manifold) that every prior label-supervised
NN we tried either parroted the rule or plateaued at ~0.965.

Outputs (to /kaggle/working/):
  oof_dae_embed.npy        (630000, 128)  — train-set embeddings
  test_dae_embed.npy       (270000, 128)  — test-set embeddings
  dae_embed_results.json   — config + final reconstruction loss

Smoke: SMOKE=1 → 20k train + 10k test + 10k orig, 2 epochs. ~60 s.
Production: full data, 30 epochs, ~30-40 min on P100.
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path

# P100 sm_60 shim matching kernel_ftt pattern.
def _gpu_arch():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        return [x.strip() for x in out if x.strip()]
    except Exception as e:
        return [f"err:{e}"]

_arches = _gpu_arch()
print(f"[boot] gpu compute_cap = {_arches}", flush=True)
if any(a in ("6.0", "6.1") for a in _arches):
    print("[boot] sm_60/61 detected - installing torch 2.5.1 cu121", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall", "--no-deps",
        "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121",
    ])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
TARGET = "Irrigation_Need"
BOTTLENECK = 128
SWAP_P = 0.15
BATCH = 4096
LR = 1e-3
WD = 1e-5
GRAD_CLIP = 1.0

SMOKE = os.environ.get("SMOKE", "0") == "1"  # v2: production (full data, 30 epochs)
EPOCHS = 2 if SMOKE else 30

KAGGLE_INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
OUT.mkdir(exist_ok=True, parents=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _find_one(pattern: str) -> Path:
    for p in KAGGLE_INPUT.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {KAGGLE_INPUT}")


# ------------------------------------------------------- data loading + encode
def load_all():
    log("listing /kaggle/input/ CSVs")
    for p in sorted(KAGGLE_INPUT.rglob("*.csv")):
        log(f"  {p}")
    train = pd.read_csv(_find_one("train.csv"))
    test = pd.read_csv(_find_one("test.csv"))
    orig_path = None
    for pat in ("irrigation_prediction.csv", "Irrigation_Prediction.csv",
                "irrigation-prediction.csv"):
        try:
            orig_path = _find_one(pat); break
        except FileNotFoundError:
            continue
    if orig_path is None:
        for p in KAGGLE_INPUT.rglob("*.csv"):
            if p.name not in ("train.csv", "test.csv", "sample_submission.csv"):
                orig_path = p; break
    if orig_path is None:
        raise FileNotFoundError("no original-dataset CSV found")
    log(f"  orig: {orig_path}")
    orig = pd.read_csv(orig_path)

    # Drop id / target — DAE is label-unaware.
    for df in (train, test, orig):
        if "id" in df.columns: df.drop(columns=["id"], inplace=True)
        if TARGET in df.columns: df.drop(columns=[TARGET], inplace=True)

    if SMOKE:
        log("SMOKE=1 — subsampling")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        orig = orig.sample(min(10_000, len(orig)), random_state=SEED).reset_index(drop=True)

    # Align columns — orig may have extra columns (e.g. a label column we didn't
    # drop above because its name differs). Keep only the intersection with test.
    common = [c for c in test.columns if c in train.columns and c in orig.columns]
    train = train[common]; test = test[common]; orig = orig[common]
    log(f"  shapes: train={train.shape} test={test.shape} orig={orig.shape}  "
        f"common_cols={len(common)}")
    return train, test, orig


def encode_features(train, test, orig):
    """Return (X_all, n_train, n_test, enc_info). X is float32 (n_rows, n_dim).

    - Numerics: z-scored (stats on train+test+orig union).
    - Cats: one-hot encoded (vocab over union).
    """
    nums = [c for c in train.columns if train[c].dtype.kind in "fi"]
    cats = [c for c in train.columns if c not in nums]
    log(f"  nums={len(nums)} cats={len(cats)}")

    all_df = pd.concat([train, test, orig], ignore_index=True, sort=False)
    n_train, n_test = len(train), len(test)

    # Z-score numerics.
    num_mat = all_df[nums].astype(np.float32).to_numpy()
    mu = num_mat.mean(axis=0)
    sd = num_mat.std(axis=0) + 1e-6
    num_mat = (num_mat - mu) / sd
    log(f"  numerics z-scored: shape={num_mat.shape}")

    # One-hot cats.
    cat_blocks = []
    cat_widths = {}
    for c in cats:
        vals = all_df[c].astype(str)
        cats_uniq = sorted(vals.unique().tolist())
        idx = {v: i for i, v in enumerate(cats_uniq)}
        codes = vals.map(idx).to_numpy()
        oh = np.zeros((len(codes), len(cats_uniq)), dtype=np.float32)
        oh[np.arange(len(codes)), codes] = 1.0
        cat_blocks.append(oh)
        cat_widths[c] = len(cats_uniq)
    cat_mat = np.concatenate(cat_blocks, axis=1) if cat_blocks else np.zeros((len(all_df), 0), dtype=np.float32)
    log(f"  cats one-hot: shape={cat_mat.shape} widths={cat_widths}")

    X = np.concatenate([num_mat, cat_mat], axis=1).astype(np.float32)
    log(f"  combined input: shape={X.shape}  dtype={X.dtype}")
    return X, n_train, n_test, dict(nums=nums, cats=cats, cat_widths=cat_widths,
                                    n_num=len(nums), total_dim=X.shape[1])


# ------------------------------------------------------- DAE model
class DAE(nn.Module):
    """Encoder-decoder MLP. Encoder output at depth -1 is the embedding."""
    def __init__(self, in_dim, bottleneck=BOTTLENECK, widths=(1024, 512, 256)):
        super().__init__()
        enc_layers = []
        prev = in_dim
        for w in widths:
            enc_layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.GELU(),
                           nn.Dropout(0.1)]
            prev = w
        enc_layers += [nn.Linear(prev, bottleneck)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = bottleneck
        for w in reversed(widths):
            dec_layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.GELU(),
                           nn.Dropout(0.1)]
            prev = w
        dec_layers += [nn.Linear(prev, in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def swap_noise(x, p=SWAP_P):
    """Replace each cell with prob p with a random value drawn from the SAME
    column, picked uniformly from another row in the same batch.

    Pure in-batch swap; matches Porto Seguro 1st-place's implementation.
    """
    b, d = x.shape
    mask = (torch.rand(b, d, device=x.device) < p)
    rand_idx = torch.randint(0, b, (b, d), device=x.device)
    gathered = x[rand_idx, torch.arange(d, device=x.device).expand(b, d)]
    return torch.where(mask, gathered, x)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------- training
def train_dae(X, enc_info):
    X_t = torch.from_numpy(X)  # stays on CPU; move batches to GPU
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    num_workers=0, pin_memory=True, drop_last=True)

    model = DAE(in_dim=X.shape[1]).to(DEVICE)
    log(f"  model params = {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(1, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    losses = []
    global_step = 0
    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        ep_batches = 0
        t0 = time.time()
        for (xb,) in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            xc = swap_noise(xb, p=SWAP_P)
            recon, _ = model(xc)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sch.step()
            ep_loss += loss.item()
            ep_batches += 1
            global_step += 1
        avg = ep_loss / ep_batches
        losses.append(avg)
        log(f"  epoch {ep+1}/{EPOCHS}  mse={avg:.5f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  wall={time.time()-t0:.1f}s")
    return model, losses


@torch.no_grad()
def extract_embeddings(model, X):
    """Pass uncorrupted rows through encoder; return (n, BOTTLENECK)."""
    model.eval()
    X_t = torch.from_numpy(X)
    embs = []
    for i in range(0, len(X_t), BATCH):
        xb = X_t[i:i+BATCH].to(DEVICE, non_blocking=True)
        z = model.encoder(xb)
        embs.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)


# ------------------------------------------------------- main
def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    train, test, orig = load_all()
    X, n_train, n_test, enc_info = encode_features(train, test, orig)

    log(f"training DAE on {len(X):,} rows, input_dim={X.shape[1]}, "
        f"bottleneck={BOTTLENECK}, swap_p={SWAP_P}, epochs={EPOCHS}")
    model, losses = train_dae(X, enc_info)

    log("extracting embeddings for train + test")
    emb = extract_embeddings(model, X)
    train_emb = emb[:n_train]
    test_emb = emb[n_train:n_train+n_test]
    log(f"  train_emb={train_emb.shape}  test_emb={test_emb.shape}  "
        f"mean={emb.mean():.4f}  std={emb.std():.4f}")

    suffix = "_smoke" if SMOKE else ""
    np.save(OUT / f"oof_dae_embed{suffix}.npy", train_emb)
    np.save(OUT / f"test_dae_embed{suffix}.npy", test_emb)
    log(f"wrote oof_dae_embed{suffix}.npy + test_dae_embed{suffix}.npy")

    summary = dict(
        seed=SEED, smoke=SMOKE, epochs=EPOCHS, batch=BATCH,
        bottleneck=BOTTLENECK, swap_p=SWAP_P, lr=LR, weight_decay=WD,
        input_dim=int(X.shape[1]),
        n_train=int(n_train), n_test=int(n_test),
        n_total_unsup=int(len(X)),
        epoch_losses=[float(x) for x in losses],
        final_loss=float(losses[-1]) if losses else None,
        emb_mean=float(emb.mean()),
        emb_std=float(emb.std()),
        total_wall_seconds=time.time() - t_start,
        enc_info={k: (v if not isinstance(v, dict) else {kk: int(vv) for kk, vv in v.items()})
                  for k, v in enc_info.items() if k != "cat_widths"} | {
            "cat_widths": {k: int(v) for k, v in enc_info["cat_widths"].items()},
        },
    )
    with open(OUT / f"dae_embed_results{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"wrote dae_embed_results{suffix}.json  "
        f"total wall {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
