"""Idea 2: pretrain on 10k original, fine-tune on 630k synthetic.

Hypothesis (2026-04-22 reframing): the synthetic 630k labels are
the host NN applied to synthetic features, trained on the 10k
original. Our v5 MLP (1M params) plateaued at 0.965 when trained
from scratch on synthetic — the network couldn't settle into the
rule basin cleanly with rule+flips as a noisy joint target.

Pretraining on the 10k original first gives the net a rule-aligned
initialization (original is rule-perfect). Then fine-tuning on
synthetic with a low LR pushes it toward the host's specific NN
instance without having to rediscover the rule. Structurally
different from v5 (never saw the original) and from weighted
external-data augmentation (loss-level bias, not init-level).

Protocol per fold:
  Phase 1: 30 epochs on 10k original, CE + original Balanced Softmax
           prior, lr=1e-3, batch=512.
  Phase 2: 15 epochs on 630k synthetic-train fold, Balanced Softmax
           with synthetic prior, lr=1e-4 (10x lower), batch=4096.
  Eval: synthetic-val fold throughout phase 2, keep best-val probs.

Fold-1 gates:
  - error Jaccard vs greedy >= 0.90 -> kill (mimicking trees)
  - Jaccard <  0.85                 -> full 5-fold run
  - 0.85 <= J < 0.90                -> full run but downgrade blend
                                       expectations to ~+0.0002.

Outputs (/kaggle/working):
  oof_mlp_pretrain_ft.npy, test_mlp_pretrain_ft.npy,
  mlp_pretrain_ft_results.json,
  submission_mlp_pretrain_ft_tuned.csv.
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path

# sm_60 (P100) shim — keep from v5
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

EMB_NUM = 8
EMB_CAT = 16
HIDDEN = [768, 512, 384, 256]
DROPOUT = 0.25

# Pretrain on original
PRE_EPOCHS = 30
PRE_BATCH = 512
PRE_LR = 1e-3
PRE_WD = 1e-4

# Fine-tune on synthetic
FT_EPOCHS = 15
FT_BATCH = 4096
FT_LR = 1e-4
FT_WD = 1e-4
FT_WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0

JACCARD_KILL = 0.90
JACCARD_WARN = 0.85

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_distance_features(df):
    """Full 43-feature dist set — same as benchmark_dist.py / v5 MLP."""
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


class NumEmb(nn.Module):
    def __init__(self, n_feat, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, emb_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, emb_dim))

    def forward(self, x):
        return x.unsqueeze(-1) * self.w + self.b


class TabMLP(nn.Module):
    def __init__(self, n_num, cat_cards, n_classes=3,
                 emb_num=EMB_NUM, emb_cat=EMB_CAT,
                 hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.num_emb = NumEmb(n_num, emb_num)
        self.cat_emb = nn.ModuleList([nn.Embedding(c, emb_cat) for c in cat_cards])
        in_dim = n_num * emb_num + len(cat_cards) * emb_cat
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(),
                       nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x_num, x_cat):
        e_num = self.num_emb(x_num).flatten(1)
        e_cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1)
        return self.head(self.backbone(torch.cat([e_num, e_cat], dim=1)))


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * prog))


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def pretrain_on_original(model, X_num_o, X_cat_o, y_o, log_prior_o_t):
    """Phase 1: 30 epochs on 10k original, Balanced Softmax w/ original prior."""
    opt = torch.optim.AdamW(model.parameters(), lr=PRE_LR, weight_decay=PRE_WD)
    Xn = torch.from_numpy(X_num_o).float().to(DEVICE)
    Xc = torch.from_numpy(X_cat_o).long().to(DEVICE)
    yt = torch.from_numpy(y_o).long().to(DEVICE)
    n = len(y_o)
    for epoch in range(PRE_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        running = 0.0
        for i in range(0, n, PRE_BATCH):
            idx = perm[i:i + PRE_BATCH]
            logits = model(Xn[idx], Xc[idx])
            logits_bs = logits + log_prior_o_t.unsqueeze(0)
            loss = F.cross_entropy(logits_bs, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            running += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == PRE_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                p = F.softmax(model(Xn, Xc), dim=1).cpu().numpy()
            bal_o = balanced_accuracy_score(y_o, p.argmax(axis=1))
            log(f"    [pretrain] ep {epoch+1:2d}/{PRE_EPOCHS}  "
                f"loss {running/n:.4f}  orig bal {bal_o:.4f}")


def finetune_on_synthetic(model, X_num_tr, X_cat_tr, y_tr, tr_idx, va_idx,
                          X_num_full_val, X_cat_full_val, y_val,
                          log_prior_s_t):
    """Phase 2: fine-tune on synthetic fold. Return best-val probs."""
    Xn_tr = torch.from_numpy(X_num_tr[tr_idx]).float()
    Xc_tr = torch.from_numpy(X_cat_tr[tr_idx]).long()
    yt = torch.from_numpy(y_tr[tr_idx]).long()
    ds = TensorDataset(Xn_tr, Xc_tr, yt)
    loader = DataLoader(ds, batch_size=FT_BATCH, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WD)
    total_steps = FT_EPOCHS * len(loader)
    warmup_steps = int(FT_WARMUP_FRAC * total_steps)
    step = 0

    best_val_bal = -1.0
    best_val_probs = None
    for epoch in range(FT_EPOCHS):
        model.train()
        running = 0.0
        for xn, xc, yb in loader:
            lr_now = cosine_lr(step, total_steps, warmup_steps, FT_LR)
            for g in opt.param_groups: g["lr"] = lr_now
            xn = xn.to(DEVICE, non_blocking=True)
            xc = xc.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model(xn, xc)
            logits_bs = logits + log_prior_s_t.unsqueeze(0)
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
            bs = 16384
            for i in range(0, len(X_num_full_val), bs):
                xn = X_num_full_val[i:i+bs].to(DEVICE, non_blocking=True)
                xc = X_cat_full_val[i:i+bs].to(DEVICE, non_blocking=True)
                val_logits.append(model(xn, xc).cpu())
            val_logits = torch.cat(val_logits, 0).numpy()
        val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        bal = balanced_accuracy_score(y_val, val_probs.argmax(axis=1))
        log(f"    [finetune] ep {epoch+1:2d}/{FT_EPOCHS}  loss {running/len(tr_idx):.4f}  "
            f"val bal {bal:.5f}  lr {lr_now:.1e}")
        if bal > best_val_bal:
            best_val_bal = bal
            best_val_probs = val_probs
    return best_val_probs, best_val_bal


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
    orig_csv = find_one("irrigation_prediction.csv")
    gate_oof_path = find_one("oof_greedy_blend.npy")
    assert train_csv and test_csv, f"missing comp csvs"
    assert orig_csv, "missing irrigation_prediction.csv (l3llff dataset not attached?)"
    assert gate_oof_path, "missing oof_greedy_blend.npy"
    log(f"train:{train_csv}  test:{test_csv}  orig:{orig_csv}  gate:{gate_oof_path}")

    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    orig = pd.read_csv(orig_csv)
    gate_oof = np.load(gate_oof_path)
    log(f"train {tr.shape}  test {te.shape}  orig {orig.shape}  gate {gate_oof.shape}")

    log("feature engineering")
    tr = add_distance_features(tr)
    te = add_distance_features(te)
    orig = add_distance_features(orig)

    cat_cols = [c for c in tr.columns
                if c not in (TARGET, ID) and tr[c].dtype == object]
    num_cols = [c for c in tr.columns
                if c not in (TARGET, ID) and c not in cat_cols]
    log(f"features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    # Unified cat vocab across train U test U orig
    cat_cards = []
    for c in cat_cols:
        vocab = sorted(set(tr[c].astype(str))
                       | set(te[c].astype(str))
                       | set(orig[c].astype(str)))
        mp = {v: i for i, v in enumerate(vocab)}
        tr[c] = tr[c].astype(str).map(mp).astype("int64")
        te[c] = te[c].astype(str).map(mp).astype("int64")
        orig[c] = orig[c].astype(str).map(mp).astype("int64")
        cat_cards.append(len(vocab))
    log(f"cat cards: {dict(zip(cat_cols, cat_cards))}")

    # Standardize numerics with synthetic-train stats (downstream-target
    # alignment; original falls into those stats via shared DGP).
    X_num_tr = tr[num_cols].to_numpy(dtype=np.float32)
    X_num_te = te[num_cols].to_numpy(dtype=np.float32)
    X_num_orig = orig[num_cols].to_numpy(dtype=np.float32)
    mu = X_num_tr.mean(axis=0, keepdims=True)
    sd = X_num_tr.std(axis=0, keepdims=True) + 1e-6
    X_num_tr = (X_num_tr - mu) / sd
    X_num_te = (X_num_te - mu) / sd
    X_num_orig = (X_num_orig - mu) / sd

    X_cat_tr = tr[cat_cols].to_numpy(dtype=np.int64)
    X_cat_te = te[cat_cols].to_numpy(dtype=np.int64)
    X_cat_orig = orig[cat_cols].to_numpy(dtype=np.int64)

    y_tr = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    y_orig = orig[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    prior_tr = np.bincount(y_tr, minlength=3) / len(y_tr)
    prior_orig = np.bincount(y_orig, minlength=3) / len(y_orig)
    log_prior_tr = torch.from_numpy(np.log(prior_tr).astype(np.float32)).to(DEVICE)
    log_prior_orig = torch.from_numpy(np.log(prior_orig).astype(np.float32)).to(DEVICE)
    log(f"priors synth {prior_tr.round(4).tolist()}  orig {prior_orig.round(4).tolist()}")

    X_num_te_t = torch.from_numpy(X_num_te).float()
    X_cat_te_t = torch.from_numpy(X_cat_te).long()

    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    fold_logs = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_tr)), y_tr)):
        t0 = time.time()
        log(f"=== fold {fold+1}/{N_FOLDS}  train {len(tr_idx)}  val {len(va_idx)} ===")

        # Fresh model per fold — pretrain each fold so the init is always
        # rule-aligned. (Alternative: pretrain once, reuse; but this makes
        # the experiment more deterministic per fold.)
        torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
        model = TabMLP(n_num=X_num_tr.shape[1], cat_cards=cat_cards).to(DEVICE)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            log(f"  model params: {n_params:,}")

        log(f"  [phase 1] pretrain on 10k original ({PRE_EPOCHS} epochs)")
        t_pre = time.time()
        pretrain_on_original(model, X_num_orig, X_cat_orig, y_orig, log_prior_orig)
        log(f"  pretrain done ({time.time()-t_pre:.0f}s)")

        log(f"  [phase 2] fine-tune on synthetic fold ({FT_EPOCHS} epochs, "
            f"lr={FT_LR}, batch={FT_BATCH})")
        X_num_fva_t = torch.from_numpy(X_num_tr[va_idx]).float()
        X_cat_fva_t = torch.from_numpy(X_cat_tr[va_idx]).long()
        t_ft = time.time()
        best_val_probs, best_val_bal = finetune_on_synthetic(
            model, X_num_tr, X_cat_tr, y_tr, tr_idx, va_idx,
            X_num_fva_t, X_cat_fva_t, y_tr[va_idx], log_prior_tr,
        )
        log(f"  finetune done ({time.time()-t_ft:.0f}s)  best val {best_val_bal:.5f}")
        oof[va_idx] = best_val_probs

        model.eval()
        with torch.no_grad():
            test_logits = []
            bs = 16384
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
            mlp_pred = oof[va_idx].argmax(axis=1)
            greedy_pred_fold1 = gate_oof[va_idx].argmax(axis=1)
            e_mlp = set(va_idx[mlp_pred != y_tr[va_idx]])
            e_grd = set(va_idx[greedy_pred_fold1 != y_tr[va_idx]])
            inter = len(e_mlp & e_grd)
            union = len(e_mlp | e_grd) or 1
            jac = inter / union
            log(f"  fold-1 error-Jaccard (MLP vs greedy) = {jac:.4f}  "
                f"inter={inter}  union={union}  MLP errs={len(e_mlp)}  greedy errs={len(e_grd)}")
            fold_entry["jaccard_vs_greedy"] = jac
            if jac >= JACCARD_KILL:
                log(f"  *** KILL: Jaccard {jac:.4f} >= {JACCARD_KILL} — aborting ***")
                with open(OUT / "mlp_pretrain_ft_results.json", "w") as f:
                    json.dump({"killed_at_fold": 1,
                               "jaccard_vs_greedy": jac,
                               "fold_logs": fold_logs}, f, indent=2)
                return
            elif jac >= JACCARD_WARN:
                log(f"  warn: Jaccard {jac:.4f} in [{JACCARD_WARN}, {JACCARD_KILL}) — "
                    f"blend lift ceiling ~+0.0002")

    argmax_bal = balanced_accuracy_score(y_tr, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y_tr, (oof / prior_tr).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y_tr, prior_tr)
    log(f"OOF argmax={argmax_bal:.5f}  reweight={reweight_bal:.5f}  "
        f"tuned={tuned_bal:.5f}  bias={bias.round(4).tolist()}")

    np.save(OUT / "oof_mlp_pretrain_ft.npy", oof)
    np.save(OUT / "test_mlp_pretrain_ft.npy", test_probs)

    cm = confusion_matrix(y_tr, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF tuned confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    tuned_test_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT / "submission_mlp_pretrain_ft_tuned.csv", index=False
    )

    with open(OUT / "mlp_pretrain_ft_results.json", "w") as f:
        json.dump({
            "n_num": len(num_cols),
            "n_cat": len(cat_cols),
            "cat_cards": cat_cards,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "pretrain": {"epochs": PRE_EPOCHS, "batch": PRE_BATCH, "lr": PRE_LR, "wd": PRE_WD},
            "finetune": {"epochs": FT_EPOCHS, "batch": FT_BATCH, "lr": FT_LR, "wd": FT_WD},
            "seed": SEED,
            "synth_prior": prior_tr.tolist(),
            "orig_prior": prior_orig.tolist(),
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
            "fold_logs": fold_logs,
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
