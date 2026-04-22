"""FT-Transformer on the 43-feature dist set.

Rationale: 10 plain-MLP nulls (v5-v9, idea-1, idea-2) all plateaued at
~0.965 with the per-feature embedding + wide MLP backbone. Attention
over feature tokens is the one tabular-NN family we have NOT tried —
it learns cross-feature interactions through dot-product attention
rather than element-wise products in hidden layers, which is a
structurally different inductive bias and may land on a different
attractor than MLPs.

Reference: Gorishniy et al. 2021 "Revisiting Deep Learning Models for
Tabular Data". We implement the canonical FT-Transformer from scratch
(no rtdl dependency): per-feature token embedding + learnable [CLS]
token + standard transformer encoder + [CLS] classification head.

Same 5-fold StratifiedKFold(shuffle=True, seed=42) split as every
other OOF, Balanced Softmax loss matching the priors, fold-1
error-Jaccard gate vs oof_greedy_blend.npy.
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path

# sm_60 (P100) shim — keep compatibility if Kaggle allocates P100
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
ACTIVE_STAGES = ("Flowering", "Vegetative")

# FT-Transformer hyperparameters (size roughly matched to v5's 1M)
D_TOKEN = 192          # per-feature token dim
N_BLOCKS = 4           # transformer encoder layers
N_HEADS = 8            # attention heads
ATTN_DROPOUT = 0.15
FFN_DROPOUT = 0.15
RESIDUAL_DROPOUT = 0.0
FFN_FACTOR = 4.0 / 3   # standard FT-Transformer ratio
# Total params ~1-1.2M for 43 features + 8 cats + [CLS]
BATCH = 2048
EPOCHS = 20
LR = 3e-4              # transformers often want lower LR than MLPs
WD = 1e-5
WARMUP_FRAC = 0.1
GRAD_CLIP = 1.0

JACCARD_KILL = 0.90
JACCARD_WARN = 0.85

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_distance_features(df):
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values

    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)

    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    out["dry"] = dry; out["norain"] = norain; out["hot"] = hot
    out["windy"] = windy; out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)

    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)
    return out


# --- FT-Transformer implementation -----------------------------------------
class NumericalFeatureTokenizer(nn.Module):
    """x_num shape (B, N) -> (B, N, d_token). Each feature gets its own
    learnable weight and bias projecting the scalar to d_token."""
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x):
        # (B, N) * (N, d) -> (B, N, d)
        return x.unsqueeze(-1) * self.weight + self.bias


class CategoricalFeatureTokenizer(nn.Module):
    """Per-column embedding tables. x_cat (B, C) long -> (B, C, d_token)."""
    def __init__(self, cardinalities, d_token):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cardinalities
        ])
        for emb in self.embeddings:
            nn.init.kaiming_uniform_(emb.weight, a=math.sqrt(5))

    def forward(self, x_cat):
        # x_cat: (B, C) long
        return torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )  # (B, C, d_token)


class CLSToken(nn.Module):
    def __init__(self, d_token):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.token, a=math.sqrt(5))

    def forward(self, x):  # x: (B, N, d) -> (B, N+1, d)
        cls = self.token.expand(x.size(0), 1, -1)
        return torch.cat([cls, x], dim=1)


class FTBlock(nn.Module):
    """FT-Transformer style pre-LN encoder block with MHA + FFN."""
    def __init__(self, d_token, n_heads, attn_dropout, ffn_dropout,
                 residual_dropout, ffn_factor):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_token)
        d_hidden = int(d_token * ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_hidden, d_token),
        )
        self.res_drop = nn.Dropout(residual_dropout)

    def forward(self, x):
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.res_drop(a)
        h = self.norm_ffn(x)
        x = x + self.res_drop(self.ffn(h))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cards, n_classes=3,
                 d_token=D_TOKEN, n_blocks=N_BLOCKS, n_heads=N_HEADS,
                 attn_dropout=ATTN_DROPOUT, ffn_dropout=FFN_DROPOUT,
                 residual_dropout=RESIDUAL_DROPOUT, ffn_factor=FFN_FACTOR):
        super().__init__()
        self.num_tok = NumericalFeatureTokenizer(n_num, d_token)
        self.cat_tok = CategoricalFeatureTokenizer(cat_cards, d_token) if cat_cards else None
        self.cls = CLSToken(d_token)
        self.blocks = nn.ModuleList([
            FTBlock(d_token, n_heads, attn_dropout, ffn_dropout,
                    residual_dropout, ffn_factor) for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x_num, x_cat):
        num_tok = self.num_tok(x_num)  # (B, N, d)
        if self.cat_tok is not None and x_cat is not None and x_cat.size(1) > 0:
            cat_tok = self.cat_tok(x_cat)  # (B, C, d)
            tokens = torch.cat([num_tok, cat_tok], dim=1)
        else:
            tokens = num_tok
        tokens = self.cls(tokens)  # (B, N+C+1, d)
        for blk in self.blocks:
            tokens = blk(tokens)
        cls_out = self.head_norm(tokens[:, 0])  # (B, d)
        return self.head(cls_out)


# --- training helpers ------------------------------------------------------
def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * prog))


def tune_log_bias(oof, y, prior):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main():
    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={DEVICE}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    log("listing /kaggle/input/")
    for p in sorted(Path("/kaggle/input").rglob("*")):
        if p.is_file():
            log(f"  {p}")

    def find_one(pattern):
        for p in Path("/kaggle/input").rglob(pattern):
            return p
        return None
    train_csv = find_one("train.csv")
    test_csv = find_one("test.csv")
    gate_oof_path = find_one("oof_greedy_blend.npy")
    assert train_csv and test_csv, "missing comp csvs"
    assert gate_oof_path, "missing oof_greedy_blend.npy"
    log(f"train:{train_csv}  test:{test_csv}  gate:{gate_oof_path}")

    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    gate_oof = np.load(gate_oof_path)
    log(f"train {tr.shape}  test {te.shape}  gate {gate_oof.shape}")

    tr = add_distance_features(tr)
    te = add_distance_features(te)

    cat_cols = [c for c in tr.columns
                if c not in (TARGET, ID) and tr[c].dtype == object]
    num_cols = [c for c in tr.columns
                if c not in (TARGET, ID) and c not in cat_cols]
    log(f"features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    cat_cards = []
    for c in cat_cols:
        vocab = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
        mp = {v: i for i, v in enumerate(vocab)}
        tr[c] = tr[c].astype(str).map(mp).astype("int64")
        te[c] = te[c].astype(str).map(mp).astype("int64")
        cat_cards.append(len(vocab))
    log(f"cat cards: {dict(zip(cat_cols, cat_cards))}")

    X_num_tr = tr[num_cols].to_numpy(dtype=np.float32)
    X_num_te = te[num_cols].to_numpy(dtype=np.float32)
    mu = X_num_tr.mean(axis=0, keepdims=True)
    sd = X_num_tr.std(axis=0, keepdims=True) + 1e-6
    X_num_tr = (X_num_tr - mu) / sd
    X_num_te = (X_num_te - mu) / sd

    X_cat_tr = tr[cat_cols].to_numpy(dtype=np.int64)
    X_cat_te = te[cat_cols].to_numpy(dtype=np.int64)

    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    prior = np.bincount(y, minlength=3) / len(y)
    log_prior = torch.from_numpy(np.log(prior).astype(np.float32)).to(DEVICE)
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    X_num_te_t = torch.from_numpy(X_num_te).float()
    X_cat_te_t = torch.from_numpy(X_cat_te).long()

    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    fold_logs = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        log(f"=== fold {fold+1}/{N_FOLDS}  train {len(tr_idx)}  val {len(va_idx)} ===")
        torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)

        model = FTTransformer(
            n_num=X_num_tr.shape[1], cat_cards=cat_cards,
        ).to(DEVICE)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            log(f"  model params: {n_params:,}")

        X_num_ftr = torch.from_numpy(X_num_tr[tr_idx]).float()
        X_cat_ftr = torch.from_numpy(X_cat_tr[tr_idx]).long()
        y_ftr = torch.from_numpy(y[tr_idx]).long()
        X_num_fva = torch.from_numpy(X_num_tr[va_idx]).float()
        X_cat_fva = torch.from_numpy(X_cat_tr[va_idx]).long()

        ds_tr = TensorDataset(X_num_ftr, X_cat_ftr, y_ftr)
        loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        total_steps = EPOCHS * len(loader_tr)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        step = 0

        best_val_bal = -1.0
        best_val_probs = None
        for epoch in range(EPOCHS):
            model.train()
            running = 0.0
            for xn, xc, yb in loader_tr:
                lr_now = cosine_lr(step, total_steps, warmup_steps, LR)
                for g in opt.param_groups: g["lr"] = lr_now
                xn = xn.to(DEVICE, non_blocking=True)
                xc = xc.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                logits = model(xn, xc)
                logits_bs = logits + log_prior.unsqueeze(0)  # Balanced Softmax
                loss = F.cross_entropy(logits_bs, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                running += loss.item() * yb.size(0)
                step += 1

            model.eval()
            with torch.no_grad():
                val_logits = []
                bs = 8192
                for i in range(0, len(X_num_fva), bs):
                    xn = X_num_fva[i:i+bs].to(DEVICE, non_blocking=True)
                    xc = X_cat_fva[i:i+bs].to(DEVICE, non_blocking=True)
                    val_logits.append(model(xn, xc).cpu())
                val_logits = torch.cat(val_logits, 0).numpy()
            val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
            bal = balanced_accuracy_score(y[va_idx], val_probs.argmax(axis=1))
            log(f"  ep {epoch+1:2d}/{EPOCHS}  loss {running/len(tr_idx):.4f}  "
                f"val bal {bal:.5f}  lr {lr_now:.2e}")
            if bal > best_val_bal:
                best_val_bal = bal
                best_val_probs = val_probs

        oof[va_idx] = best_val_probs

        model.eval()
        with torch.no_grad():
            test_logits = []
            bs = 8192
            for i in range(0, len(X_num_te_t), bs):
                xn = X_num_te_t[i:i+bs].to(DEVICE, non_blocking=True)
                xc = X_cat_te_t[i:i+bs].to(DEVICE, non_blocking=True)
                test_logits.append(model(xn, xc).cpu())
            test_logits = torch.cat(test_logits, 0).numpy()
        test_fold = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        test_probs += test_fold / N_FOLDS

        fold_entry = {
            "fold": fold + 1,
            "val_bal_acc_best": float(best_val_bal),
            "seconds": round(time.time() - t0, 1),
        }
        fold_logs.append(fold_entry)

        if fold == 0:
            ftt_pred = oof[va_idx].argmax(axis=1)
            greedy_pred_fold1 = gate_oof[va_idx].argmax(axis=1)
            e_ftt = set(va_idx[ftt_pred != y[va_idx]])
            e_grd = set(va_idx[greedy_pred_fold1 != y[va_idx]])
            inter = len(e_ftt & e_grd)
            union = len(e_ftt | e_grd) or 1
            jac = inter / union
            log(f"  fold-1 Jaccard (ftt vs greedy) = {jac:.4f}  "
                f"ftt errs={len(e_ftt)}  greedy errs={len(e_grd)}")
            fold_entry["jaccard_vs_greedy"] = jac
            if jac >= JACCARD_KILL:
                log(f"  *** KILL: Jaccard {jac:.4f} >= {JACCARD_KILL} — aborting ***")
                with open(OUT / "ft_transformer_results.json", "w") as f:
                    json.dump({"killed_at_fold": 1,
                               "jaccard_vs_greedy": jac,
                               "fold_logs": fold_logs}, f, indent=2)
                return

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"OOF argmax={argmax_bal:.5f}  reweight={reweight_bal:.5f}  "
        f"tuned={tuned_bal:.5f}  bias={bias.round(4).tolist()}")

    np.save(OUT / "oof_ft_transformer.npy", oof)
    np.save(OUT / "test_ft_transformer.npy", test_probs)

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF tuned confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    tuned_test_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT / "submission_ft_transformer_tuned.csv", index=False
    )

    with open(OUT / "ft_transformer_results.json", "w") as f:
        json.dump({
            "n_num": len(num_cols),
            "n_cat": len(cat_cols),
            "cat_cards": cat_cards,
            "d_token": D_TOKEN, "n_blocks": N_BLOCKS, "n_heads": N_HEADS,
            "batch": BATCH, "epochs": EPOCHS, "lr": LR, "wd": WD,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
            "fold_logs": fold_logs,
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
