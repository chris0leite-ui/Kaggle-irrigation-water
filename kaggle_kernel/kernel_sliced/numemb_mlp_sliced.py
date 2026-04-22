"""NumEmb MLPs on feature-sliced inputs: v6 (non-rule only) + v7 (top-3 flip).

Tests whether restricting the MLP to low-importance / non-rule features
forces it to learn the NN-generator's flip perturbation orthogonally
to the tree ensemble. Direct NN analog of the XGB-nonrule experiment
that won +0.00056 LB on main.

Two variants trained in the same kernel for quota efficiency:

v6 — 13 non-rule features (6 cat + 7 num):
    Soil_Type, Crop_Type, Season, Irrigation_Type, Water_Source, Region,
    Soil_pH, Organic_Carbon, Electrical_Conductivity, Humidity,
    Sunlight_Hours, Field_Area_hectare, Previous_Irrigation_mm
    Arch: [256, 192, 128, 96]  (~150k params)

v7 — top-3 flip-significant numerics only:
    Humidity, Previous_Irrigation_mm, Electrical_Conductivity
    (the only features with Cohen's d > 0.05 on flipped rows per the
    2026-04-21 DGP residuals EDA)
    Arch: [128, 96, 64]  (~15k params)

Both use the same 5-fold StratifiedKFold(shuffle=True, random_state=42)
split as every other OOF on disk. Same Balanced-Softmax loss, same
AdamW + cosine schedule as numemb_mlp.py.

Fold-1 error-Jaccard gate vs BOTH greedy and XGB-nonrule — the key
question is whether v6/v7 add signal orthogonal to XGB-nonrule (which
is already in the current LB-best stack at α=0.15), not just to greedy.
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path

# P100 compatibility shim (see v5 kernel notes)
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
    print("[boot] sm_60/61 detected — installing torch 2.5.1 cu121", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall", "--no-deps",
        "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    print("[boot] torch reinstall done", flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

EMB_NUM = 8
EMB_CAT = 16
BATCH = 4096
EPOCHS = 30
LR = 1e-3
WD = 1e-4
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0
DROPOUT = 0.25

NONRULE_CATS = ["Soil_Type", "Crop_Type", "Season", "Irrigation_Type", "Water_Source", "Region"]
NONRULE_NUMS = ["Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
                "Humidity", "Sunlight_Hours", "Field_Area_hectare", "Previous_Irrigation_mm"]
TOP3_NUMS = ["Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity"]

OUT = Path("/kaggle/working"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class NumEmb(nn.Module):
    def __init__(self, n_feat, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, emb_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, emb_dim))
    def forward(self, x):
        return x.unsqueeze(-1) * self.w + self.b


class TabMLP(nn.Module):
    def __init__(self, n_num, cat_cards, hidden, dropout=DROPOUT,
                 emb_num=EMB_NUM, emb_cat=EMB_CAT, n_classes=3):
        super().__init__()
        self.num_emb = NumEmb(n_num, emb_num) if n_num > 0 else None
        self.cat_emb = nn.ModuleList([nn.Embedding(c, emb_cat) for c in cat_cards])
        in_dim = n_num * emb_num + len(cat_cards) * emb_cat
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x_num, x_cat):
        parts = []
        if self.num_emb is not None and x_num is not None:
            parts.append(self.num_emb(x_num).flatten(1))
        if len(self.cat_emb) > 0 and x_cat is not None:
            parts.append(torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1))
        h = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        h = self.backbone(h)
        return self.head(h)


def cosine_lr(step, total, warmup, base):
    if step < warmup: return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * p))


def tune_bias(oof, y, prior):
    log_o = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_o + bias).argmax(1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = bias.copy(); scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_o + base).argmax(1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]; best = scores[j]; imp = True
        if not imp: break
    return bias, best


def train_variant(variant_name, hidden, num_cols, cat_cols,
                  X_num_tr, X_cat_tr, X_num_te, X_cat_te, y, prior,
                  cat_cards, gate_oof, gate_xgb_nonrule):
    log(f"=== training variant {variant_name} | hidden={hidden} "
        f"| n_num={len(num_cols)} n_cat={len(cat_cols)} ===")
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test_probs = np.zeros((len(X_num_te), 3) if len(num_cols) > 0 else (len(X_cat_te), 3),
                          dtype=np.float64)
    fold_logs = []
    log_prior_t = torch.from_numpy(np.log(prior).astype(np.float32)).to(DEVICE)

    X_num_te_t = torch.from_numpy(X_num_te).float() if len(num_cols) > 0 else None
    X_cat_te_t = torch.from_numpy(X_cat_te).long() if len(cat_cols) > 0 else None

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        X_num_ftr = torch.from_numpy(X_num_tr[tr_idx]).float() if len(num_cols) > 0 else None
        X_cat_ftr = torch.from_numpy(X_cat_tr[tr_idx]).long() if len(cat_cols) > 0 else None
        y_ftr = torch.from_numpy(y[tr_idx]).long()
        X_num_fva = torch.from_numpy(X_num_tr[va_idx]).float() if len(num_cols) > 0 else None
        X_cat_fva = torch.from_numpy(X_cat_tr[va_idx]).long() if len(cat_cols) > 0 else None
        y_fva = y[va_idx]

        tensors = [t for t in (X_num_ftr, X_cat_ftr, y_ftr) if t is not None]
        ds = TensorDataset(*tensors)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

        model = TabMLP(n_num=len(num_cols), cat_cards=cat_cards, hidden=hidden).to(DEVICE)
        if fold == 0:
            log(f"  {variant_name} params: {sum(p.numel() for p in model.parameters()):,}")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        total = EPOCHS * len(loader); warm = int(WARMUP_FRAC * total); step = 0

        best_bal = -1.0; best_probs = None
        for epoch in range(EPOCHS):
            model.train(); running = 0.0
            for batch in loader:
                # unpack based on presence of num/cat
                if len(num_cols) > 0 and len(cat_cols) > 0:
                    xn, xc, yb = batch
                elif len(num_cols) > 0:
                    xn, yb = batch; xc = None
                else:
                    xc, yb = batch; xn = None
                lr_now = cosine_lr(step, total, warm, LR)
                for g in opt.param_groups: g["lr"] = lr_now
                if xn is not None: xn = xn.to(DEVICE, non_blocking=True)
                if xc is not None: xc = xc.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                logits = model(xn, xc)
                loss = F.cross_entropy(logits + log_prior_t, yb)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                running += loss.item() * yb.size(0); step += 1

            model.eval()
            with torch.no_grad():
                vl = []
                bs = 16384
                n = len(va_idx)
                for i in range(0, n, bs):
                    xn = X_num_fva[i:i+bs].to(DEVICE) if X_num_fva is not None else None
                    xc = X_cat_fva[i:i+bs].to(DEVICE) if X_cat_fva is not None else None
                    vl.append(model(xn, xc).cpu())
                val_logits = torch.cat(vl, 0).numpy()
            val_probs = torch.softmax(torch.from_numpy(val_logits), 1).numpy()
            bal = balanced_accuracy_score(y_fva, val_probs.argmax(1))
            if bal > best_bal:
                best_bal = bal; best_probs = val_probs
        log(f"  fold {fold+1}/{N_FOLDS} best val bal_acc {best_bal:.5f}  ({time.time()-t0:.0f}s)")
        oof[va_idx] = best_probs

        # test pass (using final-epoch weights)
        model.eval()
        with torch.no_grad():
            tl = []
            bs = 16384
            n = X_num_te_t.shape[0] if X_num_te_t is not None else X_cat_te_t.shape[0]
            for i in range(0, n, bs):
                xn = X_num_te_t[i:i+bs].to(DEVICE) if X_num_te_t is not None else None
                xc = X_cat_te_t[i:i+bs].to(DEVICE) if X_cat_te_t is not None else None
                tl.append(model(xn, xc).cpu())
            tp = torch.softmax(torch.cat(tl, 0), 1).numpy()
        test_probs += tp / N_FOLDS
        fold_logs.append({"fold": fold+1, "val_bal_acc_best": float(best_bal),
                          "seconds": round(time.time()-t0, 1)})

        if fold == 0:
            mlp_pred = oof[va_idx].argmax(1)
            err_m = set(va_idx[mlp_pred != y_fva])
            err_g = set(va_idx[gate_oof[va_idx].argmax(1) != y_fva])
            err_x = set(va_idx[gate_xgb_nonrule[va_idx].argmax(1) != y_fva])
            j_g = len(err_m & err_g) / max(1, len(err_m | err_g))
            j_x = len(err_m & err_x) / max(1, len(err_m | err_x))
            log(f"  fold-1 jaccard vs greedy={j_g:.4f} | vs xgb_nonrule={j_x:.4f} "
                f"(|E_mlp|={len(err_m)}, |E_greedy|={len(err_g)}, |E_nonrule|={len(err_x)})")
            fold_logs[-1]["jaccard_vs_greedy"] = j_g
            fold_logs[-1]["jaccard_vs_xgb_nonrule"] = j_x

    argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
    bias, tuned_bal = tune_bias(oof, y, prior)
    log(f"{variant_name} OOF argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}  bias={bias.round(4).tolist()}")

    np.save(OUT / f"oof_mlp_{variant_name}.npy", oof)
    np.save(OUT / f"test_mlp_{variant_name}.npy", test_probs)
    return {
        "argmax_bal_acc": float(argmax_bal),
        "tuned_bal_acc": float(tuned_bal),
        "log_bias": bias.tolist(),
        "fold_logs": fold_logs,
        "hidden": hidden,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def main():
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} device={DEVICE}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    log("listing /kaggle/input/")
    for p in sorted(Path("/kaggle/input").rglob("*")):
        if p.is_file(): log(f"  {p}")

    def find_one(pat):
        for p in Path("/kaggle/input").rglob(pat): return p
        return None
    tr = pd.read_csv(find_one("train.csv"))
    te = pd.read_csv(find_one("test.csv"))
    gate_oof = np.load(find_one("oof_greedy_blend.npy"))
    gate_xgb_nonrule_path = find_one("oof_xgb_nonrule.npy")
    assert gate_xgb_nonrule_path, "oof_xgb_nonrule.npy not in input — attach dataset containing it"
    gate_xgb_nonrule = np.load(gate_xgb_nonrule_path)
    log(f"train {tr.shape} test {te.shape}  greedy OOF {gate_oof.shape}  xgb_nonrule OOF {gate_xgb_nonrule.shape}")

    # cat encoding
    cat_cards_full = {}
    for c in NONRULE_CATS:
        vocab = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
        mp = {v: i for i, v in enumerate(vocab)}
        tr[c] = tr[c].astype(str).map(mp).astype("int64")
        te[c] = te[c].astype(str).map(mp).astype("int64")
        cat_cards_full[c] = len(vocab)
    log(f"cat cardinalities: {cat_cards_full}")

    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    prior = np.bincount(y, minlength=3) / len(y)

    # --- v6: 13 non-rule features ---
    num_v6 = NONRULE_NUMS
    cat_v6 = NONRULE_CATS
    X_num_tr6 = tr[num_v6].to_numpy(dtype=np.float32)
    X_num_te6 = te[num_v6].to_numpy(dtype=np.float32)
    mu6 = X_num_tr6.mean(0); sd6 = X_num_tr6.std(0) + 1e-6
    X_num_tr6 = (X_num_tr6 - mu6) / sd6
    X_num_te6 = (X_num_te6 - mu6) / sd6
    X_cat_tr6 = tr[cat_v6].to_numpy(dtype=np.int64)
    X_cat_te6 = te[cat_v6].to_numpy(dtype=np.int64)
    r_v6 = train_variant(
        "nonrule", hidden=[256, 192, 128, 96],
        num_cols=num_v6, cat_cols=cat_v6,
        X_num_tr=X_num_tr6, X_cat_tr=X_cat_tr6,
        X_num_te=X_num_te6, X_cat_te=X_cat_te6,
        y=y, prior=prior,
        cat_cards=[cat_cards_full[c] for c in cat_v6],
        gate_oof=gate_oof, gate_xgb_nonrule=gate_xgb_nonrule,
    )

    # --- v7: top-3 flip-significant numerics only ---
    num_v7 = TOP3_NUMS
    cat_v7 = []
    X_num_tr7 = tr[num_v7].to_numpy(dtype=np.float32)
    X_num_te7 = te[num_v7].to_numpy(dtype=np.float32)
    mu7 = X_num_tr7.mean(0); sd7 = X_num_tr7.std(0) + 1e-6
    X_num_tr7 = (X_num_tr7 - mu7) / sd7
    X_num_te7 = (X_num_te7 - mu7) / sd7
    r_v7 = train_variant(
        "top3", hidden=[128, 96, 64],
        num_cols=num_v7, cat_cols=cat_v7,
        X_num_tr=X_num_tr7, X_cat_tr=np.zeros((len(tr), 0), dtype=np.int64),
        X_num_te=X_num_te7, X_cat_te=np.zeros((len(te), 0), dtype=np.int64),
        y=y, prior=prior,
        cat_cards=[],
        gate_oof=gate_oof, gate_xgb_nonrule=gate_xgb_nonrule,
    )

    with open(OUT / "mlp_sliced_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "class_priors": prior.tolist(),
            "v6_nonrule": r_v6,
            "v7_top3": r_v7,
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
