"""NumEmb + wide MLP on the 43-feature dist set.

Goal: test whether a large-capacity tabular NN (the only untried
model family) breaks our tree-ensemble OOF plateau at ~0.9738.

Design
------
- Same 5-fold StratifiedKFold(shuffle=True, random_state=42) split
  used by every other OOF on disk — so the output `.npy` plugs
  straight into the existing greedy log-blend.
- 43-feature set matching benchmark_dist.py: 11 raw numerics + 24
  DGP-distance / score / pairwise features + 8 categoricals.
- Architecture: per-feature numeric embedding (Linear(1, 8)) + per-cat
  embedding table → concat (~688 dim) → 4x[Linear/GELU/Dropout/LayerNorm]
  → 3-class logits.
- Loss: Balanced Softmax (Menon 2021) — add log(prior) to logits at
  train time, subtract at inference. Matches the class-imbalance
  prior we tune post-hoc with log-bias on trees.
- Optimizer: AdamW + cosine schedule w/ warmup, lr 1e-3, batch 4096,
  30 epochs, grad clip 1.0.
- Fold-1 error-Jaccard gate vs `oof_greedy_blend.npy`. If Jaccard
  >= 0.90 on fold 1, we exit early — the NN is mimicking the
  tree ensemble and won't add blend diversity.

Inputs (Kaggle kernel):
  /kaggle/input/playground-series-s6e4/{train,test,sample_submission}.csv
  /kaggle/input/irrigation-greedy-blend-oof/oof_greedy_blend.npy

Outputs (/kaggle/working):
  oof_mlp_numemb.npy        (630000, 3) softmax probs
  test_mlp_numemb.npy       (270000, 3) softmax probs
  mlp_numemb_results.json   fold metrics + Jaccard + tuned bias
  submission_mlp_numemb_tuned.csv
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path

# Kaggle's pre-installed torch 2.10 drops sm_60 (P100) support. If we
# happen to land on a P100 kernel (random allocation), detect that and
# install torch 2.8 with cu121 which still ships sm_60 kernels. This
# only runs when needed — on T4/L4/etc we keep the pre-installed torch.
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
    print("[boot] sm_60/61 detected — installing torch 2.8.0 cu121 (has P100 kernels)", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu121",
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

# ----- config -------------------------------------------------------------
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

EMB_NUM = 8          # per-numeric-feature embedding dim
EMB_CAT = 16         # per-categorical-feature embedding dim
HIDDEN = [768, 512, 384, 256]
DROPOUT = 0.25
BATCH = 4096
EPOCHS = 30
LR = 1e-3
WD = 1e-4
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0
JACCARD_KILL = 0.90
JACCARD_WARN = 0.85

# Kaggle paths (resolved at runtime in main())
OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----- DGP / distance feature engineering (matches benchmark_dist.py) ----
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


# ----- model --------------------------------------------------------------
class NumEmb(nn.Module):
    """Per-feature learnable embedding: Linear(1, d) with bias, applied
    independently to each numeric feature, then concatenated."""
    def __init__(self, n_feat, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, emb_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, emb_dim))

    def forward(self, x):  # x: (B, n_feat)
        # broadcast to (B, n_feat, emb_dim)
        return x.unsqueeze(-1) * self.w + self.b


class TabMLP(nn.Module):
    def __init__(self, n_num, cat_cards, n_classes=3,
                 emb_num=EMB_NUM, emb_cat=EMB_CAT,
                 hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.num_emb = NumEmb(n_num, emb_num)
        self.cat_emb = nn.ModuleList([
            nn.Embedding(card, emb_cat) for card in cat_cards
        ])
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
        # x_num: (B, n_num) float; x_cat: (B, n_cat) long
        e_num = self.num_emb(x_num).flatten(1)  # (B, n_num * emb_num)
        e_cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1)
        h = torch.cat([e_num, e_cat], dim=1)
        h = self.backbone(h)
        return self.head(h)


# ----- training loop helpers ---------------------------------------------
def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * prog))


def tune_log_bias(oof, y, prior, classes=CLASSES):
    """Coord-ascent over per-class additive log-bias, maximize bal_acc."""
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 4.5, 71)  # extended upward since best High is typically ~+3.4
    for _ in range(25):
        improved = False
        for k in range(len(classes)):
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


def main():
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} device={DEVICE}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    log("listing /kaggle/input/")
    for p in sorted(Path("/kaggle/input").rglob("*")):
        if p.is_file():
            log(f"  {p}")

    log("loading data")
    # Kaggle mounts competitions at /kaggle/input/competitions/<slug>/
    # and datasets at /kaggle/input/datasets/<user>/<slug>/. Resolve via
    # rglob so we're robust to any future mount-path change.
    def find_one(pattern):
        for p in Path("/kaggle/input").rglob(pattern):
            return p
        return None
    train_csv = find_one("train.csv")
    test_csv = find_one("test.csv")
    gate_oof_path = find_one("oof_greedy_blend.npy")
    assert train_csv and test_csv, f"missing comp csvs; train={train_csv} test={test_csv}"
    assert gate_oof_path, "missing oof_greedy_blend.npy (gate dataset not attached?)"
    log(f"train csv: {train_csv}")
    log(f"test csv:  {test_csv}")
    log(f"gate oof:  {gate_oof_path}")
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    gate_oof = np.load(gate_oof_path)
    assert gate_oof.shape == (len(tr), 3), f"gate shape {gate_oof.shape}"
    log(f"train {tr.shape} test {te.shape}  gate OOF {gate_oof.shape}")

    log("feature engineering")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    # split num vs cat
    cat_cols = [c for c in tr.columns
                if c not in (TARGET, ID) and tr[c].dtype == object]
    num_cols = [c for c in tr.columns
                if c not in (TARGET, ID) + tuple(cat_cols)]
    log(f"features: {len(num_cols)} numeric + {len(cat_cols)} categorical")
    log(f"  cat cols: {cat_cols}")

    # cat → int codes (fit on train ∪ test to ensure test embeddings are valid)
    cat_cards = []
    for c in cat_cols:
        vocab = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
        mp = {v: i for i, v in enumerate(vocab)}
        tr[c] = tr[c].astype(str).map(mp).astype("int64")
        te[c] = te[c].astype(str).map(mp).astype("int64")
        cat_cards.append(len(vocab))
    log(f"  cat cardinalities: {dict(zip(cat_cols, cat_cards))}")

    # numeric standardization (fit on train, apply to test)
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
    log_prior = np.log(prior).astype(np.float32)
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    X_num_te_t = torch.from_numpy(X_num_te).float()
    X_cat_te_t = torch.from_numpy(X_cat_te).long()

    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    fold_logs = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        log(f"--- fold {fold+1}/{N_FOLDS} | train {len(tr_idx)} val {len(va_idx)} ---")

        X_num_ftr = torch.from_numpy(X_num_tr[tr_idx]).float()
        X_cat_ftr = torch.from_numpy(X_cat_tr[tr_idx]).long()
        y_ftr = torch.from_numpy(y[tr_idx]).long()
        X_num_fva = torch.from_numpy(X_num_tr[va_idx]).float()
        X_cat_fva = torch.from_numpy(X_cat_tr[va_idx]).long()
        y_fva = torch.from_numpy(y[va_idx]).long()

        ds_tr = TensorDataset(X_num_ftr, X_cat_ftr, y_ftr)
        loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)

        model = TabMLP(n_num=X_num_tr.shape[1], cat_cards=cat_cards).to(DEVICE)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            log(f"  model params: {n_params:,}")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        total_steps = EPOCHS * len(loader_tr)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        step = 0

        log_prior_t = torch.from_numpy(log_prior).to(DEVICE)
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
                # Balanced Softmax: shift logits by log_prior at train time
                # (equivalent to minimizing post-hoc-calibrated CE)
                logits_bs = logits + log_prior_t.unsqueeze(0)
                loss = F.cross_entropy(logits_bs, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                running += loss.item() * yb.size(0)
                step += 1
            # val pass
            model.eval()
            with torch.no_grad():
                val_logits = []
                bs = 16384
                for i in range(0, len(X_num_fva), bs):
                    xn = X_num_fva[i:i+bs].to(DEVICE, non_blocking=True)
                    xc = X_cat_fva[i:i+bs].to(DEVICE, non_blocking=True)
                    val_logits.append(model(xn, xc).cpu())
                val_logits = torch.cat(val_logits, 0).numpy()
            # Balanced softmax inference: logits stay as-is (prior baked into training)
            val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
            bal = balanced_accuracy_score(y[va_idx], val_probs.argmax(axis=1))
            log(f"  ep {epoch+1:2d}/{EPOCHS} | loss {running/len(tr_idx):.4f} "
                f"| val bal_acc(argmax) {bal:.5f} | lr {lr_now:.2e}")
            if bal > best_val_bal:
                best_val_bal = bal
                best_val_probs = val_probs

        oof[va_idx] = best_val_probs

        # test predictions with current model (using the final epoch's weights)
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

        fold_logs.append({
            "fold": fold + 1,
            "val_bal_acc_best": float(best_val_bal),
            "seconds": round(time.time() - t0, 1),
        })
        log(f"  fold {fold+1} done | best val bal_acc {best_val_bal:.5f}  ({time.time()-t0:.0f}s)")

        # ------ fold-1 error-Jaccard kill gate vs greedy blend ------
        if fold == 0:
            mlp_pred = oof[va_idx].argmax(axis=1)
            greedy_pred_fold1 = gate_oof[va_idx].argmax(axis=1)
            e_mlp = set(va_idx[mlp_pred != y[va_idx]])
            e_grd = set(va_idx[greedy_pred_fold1 != y[va_idx]])
            inter = len(e_mlp & e_grd)
            union = len(e_mlp | e_grd) or 1
            jac = inter / union
            log(f"  fold-1 error-Jaccard (MLP vs greedy) = {jac:.4f} "
                f"({inter} shared / {union} union, MLP errs {len(e_mlp)}, greedy errs {len(e_grd)})")
            fold_logs[-1]["jaccard_vs_greedy"] = jac
            if jac >= JACCARD_KILL:
                log(f"  *** KILL GATE TRIPPED: Jaccard {jac:.4f} >= {JACCARD_KILL}. "
                    f"MLP mimicking greedy — aborting remaining folds. ***")
                with open(OUT / "mlp_numemb_results.json", "w") as f:
                    json.dump({
                        "killed_at_fold": 1,
                        "jaccard_vs_greedy": jac,
                        "fold_logs": fold_logs,
                    }, f, indent=2)
                return
            elif jac >= JACCARD_WARN:
                log(f"  warn: Jaccard {jac:.4f} in [{JACCARD_WARN}, {JACCARD_KILL}). "
                    f"Blend lift ceiling likely ~+0.00015; running remaining folds "
                    f"but downgrading expectations.")

    # ------ OOF metrics + log-bias tuning ------
    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"OOF argmax bal_acc = {argmax_bal:.5f}")
    log(f"OOF prior-reweight bal_acc = {reweight_bal:.5f}")
    log(f"OOF tuned log-bias bal_acc = {tuned_bal:.5f}  bias={bias.round(4).tolist()}")

    # ------ save artifacts ------
    np.save(OUT / "oof_mlp_numemb.npy", oof)
    np.save(OUT / "test_mlp_numemb.npy", test_probs)

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF tuned confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # submission at tuned bias
    tuned_test_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    sub = pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]})
    sub.to_csv(OUT / "submission_mlp_numemb_tuned.csv", index=False)

    with open(OUT / "mlp_numemb_results.json", "w") as f:
        json.dump({
            "n_num": len(num_cols),
            "n_cat": len(cat_cols),
            "cat_cols": cat_cols,
            "cat_cards": cat_cards,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "batch": BATCH,
            "epochs": EPOCHS,
            "lr": LR,
            "wd": WD,
            "seed": SEED,
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
