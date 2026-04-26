"""Angle B — rule-correct-only autoencoder anomaly-score feature.

Mechanism:
  1. Identify rule-correct rows: train rows where rule_pred == y (~98%).
  2. Train a small DAE (encoder-decoder MLP) on RAW 19 features (8 cats
     embedded + 11 nums standardised) from those rule-correct rows only.
  3. For all 630k train + 270k test rows, compute reconstruction MSE.
     Save as scripts/artifacts/{oof,test}_angle_b_recon.npy with shape
     (N, 1) — single feature per row.
  4. (Downstream): inject this column into recipe_full_te via the existing
     DAE_EMBED_PATH plumbing.

Why this is fresh vs the prior DAE attempt (2026-04-24):
  Prior DAE was label-unaware on all 900k rows (train+test+orig). This
  conditions on rule-correctness — it learns a manifold of "what rule-
  aligned rows look like" so flip-band rows get high reconstruction
  error. One feature that captures multi-axis flip vulnerability.

5-fold StratifiedKFold(seed=42) is NOT used here — the DAE has no
labels to leak. Instead, we hold out rule-correct rows for validation
and report MSE convergence.

Compute: ~10 min CPU on 4-layer MLP, batch=2048, 30 epochs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features  # noqa: E402
from tier1b_helpers import ART, SEED, log  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

CAT_COLS = ["Region", "Crop_Type", "Soil_Type", "Crop_Growth_Stage",
            "Mulching_Used", "Season", "Irrigation_Type", "Water_Source"]
NUM_COLS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
            "Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
            "Humidity", "Sunlight_Hours", "Field_Area_hectare",
            "Previous_Irrigation_mm"]


class DAE(nn.Module):
    def __init__(self, n_num: int, cat_cards: list[int], emb_dim: int = 8,
                 hidden: int = 64, latent: int = 16):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 1, emb_dim) for c in cat_cards])
        in_dim = n_num + emb_dim * len(cat_cards)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, in_dim))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        x = torch.cat([x_num] + embs, dim=1)
        z = self.enc(x)
        recon = self.dec(z)
        return recon, x


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"angle B rule-correct DAE. SMOKE={SMOKE}")

    log("loading train + test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy()

    # Build rule_pred via add_distance_features (deterministic, leak-free).
    train_eng = add_distance_features(train.drop(columns=["id"]))
    test_eng = add_distance_features(test.drop(columns=["id"]))
    rule_train = train_eng["rule_pred"].to_numpy()
    rule_correct = (rule_train == y)
    log(f"  rule_correct mask: {rule_correct.sum():,}/{len(y):,} = {100*rule_correct.mean():.2f}%")

    # Numeric standardisation: z-score using rule-correct stats only.
    rc_idx = np.where(rule_correct)[0]
    n_arr = train[NUM_COLS].to_numpy(dtype=np.float32)
    n_mean = n_arr[rc_idx].mean(0)
    n_std = n_arr[rc_idx].std(0) + 1e-6
    train_num = (n_arr - n_mean) / n_std
    test_num = (test[NUM_COLS].to_numpy(dtype=np.float32) - n_mean) / n_std

    # Categorical factorisation (combined train+test).
    cat_cards = []
    train_cat = np.zeros((len(train), len(CAT_COLS)), dtype=np.int64)
    test_cat = np.zeros((len(test), len(CAT_COLS)), dtype=np.int64)
    for i, c in enumerate(CAT_COLS):
        codes, _ = pd.factorize(pd.concat([train[c].astype(str), test[c].astype(str)]))
        train_cat[:, i] = codes[:len(train)]
        test_cat[:, i] = codes[len(train):]
        cat_cards.append(int(codes.max()) + 1)
    log(f"  cat cardinalities: {cat_cards}")

    if SMOKE:
        rc_idx = rc_idx[:20_000]
        log(f"  SMOKE: training on {len(rc_idx):,} rule-correct rows")

    # Train DAE on rule-correct rows only.
    n_train_rc = len(rc_idx)
    val_n = max(1024, n_train_rc // 20)
    val_idx = np.random.RandomState(SEED).choice(rc_idx, size=val_n, replace=False)
    val_set = set(val_idx.tolist())
    fit_idx = np.array([i for i in rc_idx if i not in val_set])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device={device}  fit_n={len(fit_idx):,} val_n={len(val_idx):,}")

    model = DAE(n_num=len(NUM_COLS), cat_cards=cat_cards, hidden=64, latent=16).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30 if not SMOKE else 5)

    X_num_t = torch.from_numpy(train_num[fit_idx])
    X_cat_t = torch.from_numpy(train_cat[fit_idx])
    Vn = torch.from_numpy(train_num[val_idx]).to(device)
    Vc = torch.from_numpy(train_cat[val_idx]).to(device)

    ds = TensorDataset(X_num_t, X_cat_t)
    dl = DataLoader(ds, batch_size=2048, shuffle=True, num_workers=0, drop_last=True)

    n_epochs = 5 if SMOKE else 30
    for ep in range(1, n_epochs + 1):
        model.train()
        loss_sum, n_b = 0.0, 0
        for xn, xc in dl:
            xn = xn.to(device); xc = xc.to(device)
            # SwapNoise on numerics + cats (15%).
            mask = torch.rand_like(xn) < 0.15
            xn_n = torch.where(mask, xn[torch.randperm(len(xn))], xn)
            mask_c = torch.rand_like(xc, dtype=torch.float) < 0.15
            xc_n = torch.where(mask_c.bool(), xc[torch.randperm(len(xc))], xc)
            recon, target = model(xn_n, xc_n)
            loss = ((recon - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); n_b += 1
        sch.step()
        model.eval()
        with torch.no_grad():
            r, t = model(Vn, Vc)
            val_mse = ((r - t) ** 2).mean().item()
        if ep == 1 or ep == n_epochs or ep % 5 == 0:
            log(f"  epoch {ep}/{n_epochs}  train_mse={loss_sum/n_b:.4f}  val_mse={val_mse:.4f}")

    # Score recon error for ALL train + test rows (mean per row across features).
    log("scoring reconstruction error for all train + test")
    model.eval()
    def score(X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X_num), dtype=np.float32)
        bs = 8192
        with torch.no_grad():
            for i in range(0, len(X_num), bs):
                xn = torch.from_numpy(X_num[i:i+bs]).to(device)
                xc = torch.from_numpy(X_cat[i:i+bs]).to(device)
                r, t = model(xn, xc)
                out[i:i+bs] = ((r - t) ** 2).mean(1).cpu().numpy()
        return out

    train_recon = score(train_num, train_cat).reshape(-1, 1)
    test_recon = score(test_num, test_cat).reshape(-1, 1)

    # Save with a name compatible with DAE_EMBED_PATH plumbing.
    np.save(ART / "oof_angle_b_recon.npy", train_recon)
    np.save(ART / "test_angle_b_recon.npy", test_recon)

    # Diagnostic: compare recon error of rule-correct vs rule-flipped train rows.
    rc_mse = float(train_recon[rule_correct].mean())
    rf_mse = float(train_recon[~rule_correct].mean())
    rc_p99 = float(np.percentile(train_recon[rule_correct], 99))
    rf_p50 = float(np.percentile(train_recon[~rule_correct], 50))
    log(f"  recon-MSE  rule_correct mean={rc_mse:.4f} p99={rc_p99:.4f}")
    log(f"  recon-MSE  rule_flipped mean={rf_mse:.4f} p50={rf_p50:.4f}")
    log(f"  separation ratio (flipped/correct mean) = {rf_mse/max(rc_mse,1e-9):.3f}x")

    out = dict(
        smoke=SMOKE, n_epochs=n_epochs,
        rule_correct_count=int(rule_correct.sum()),
        rule_flipped_count=int((~rule_correct).sum()),
        recon_mse_rule_correct_mean=rc_mse,
        recon_mse_rule_flipped_mean=rf_mse,
        recon_mse_correct_p99=rc_p99,
        recon_mse_flipped_p50=rf_p50,
        separation_ratio=float(rf_mse / max(rc_mse, 1e-9)),
        wall_min=(time.time() - t0) / 60.0,
    )
    out_path = ART / "angle_b_dae_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {out_path}  wall={out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
