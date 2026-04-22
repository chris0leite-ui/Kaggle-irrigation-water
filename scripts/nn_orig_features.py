"""NN-on-original ensemble -> predictions as features (idea 1).

Reframing (2026-04-22):
  The host's label-generating NN was trained on the 10k original.
  Applied to synthetic features, its smooth decision boundary
  differs from the axis-aligned rule near thresholds -- those
  differences ARE what we've been calling "flips". Training OUR
  OWN ensemble of NNs on the 10k original and applying to synthetic
  features should partially reproduce that flip pattern because
  we're fitting the same underlying function.

Protocol:
  1. Load 10k original (rule-perfect) + 630k synthetic train + 270k test.
  2. Build the same 43-feature `dist` set used by benchmark_dist.py.
  3. Standardize numerics with SYNTHETIC-train stats; unify cat vocab
     across original U train U test.
  4. Train 5 diverse small MLPs on 10k original (different seed, depth,
     width, activation). Each sees the full 10k -- no inner CV, just
     fixed epochs + weight decay.
  5. Inference: predict 3-class softmax on 630k train + 270k test.
     Average the 5 architectures' predictions.
  6. Save as oof_nn_orig_ens.npy (630k,3) and test_nn_orig_ens.npy (270k,3)
     -- these are NOT CV-OOF but single-fold (NN trained on disjoint
     source), so no leakage. They plug into the existing fixed-bias
     blend protocol the same way xgb-nonrule did.

Expected lift: unknown. If our 5-arch envelope partially covers the
host's smooth-boundary hypothesis, fixed-bias sweep vs greedy should
show a unimodal curve with peak >= +0.0003 OOF. Null if the host used
a structurally different model.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

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

DATA_DIR = Path("data")
OOF_DIR = Path("scripts/artifacts")
OOF_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----- feature engineering: RAW ONLY (no rule/score/discrete-threshold cols) --
# Key design: the first run with the full 43-feature `dist` set produced an
# ensemble that collapsed exactly to the rule ceiling (0.96097 synth_bal on
# every arch) because `rule_pred` and `dgp_score` are algebraic proxies for
# the target and NNs trivially parrot them. For idea 1 to produce the
# smoothed-rule predictions we want, the NN has to re-discover the rule
# boundary from CONTINUOUS raw features — so we drop rule_pred, dgp_score,
# all discrete threshold indicators (dry/norain/hot/windy/nomulch/kc_active),
# and the score-distance columns. The remaining signed/abs distances keep
# the BOUNDARY LOCATIONS visible but require smooth interpolation near them.
def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values

    # Signed distances to each continuous threshold — tells the NN WHERE
    # the boundary is, but as a continuous signal, not a discrete hint.
    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    # Absolute distances — how-close-to-boundary, same smooth nature.
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    # Continuous minimum-axis signal.
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)
    # A handful of pairwise smooth interactions.
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    return out


# ----- model: small MLP with per-numeric embedding + cat embeddings ------
class NumEmb(nn.Module):
    def __init__(self, n_feat, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, emb_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, emb_dim))

    def forward(self, x):
        return x.unsqueeze(-1) * self.w + self.b


class TabMLP(nn.Module):
    def __init__(self, n_num, cat_cards, n_classes=3,
                 emb_num=4, emb_cat=8,
                 hidden=(128, 64), activation="relu", dropout=0.2):
        super().__init__()
        self.num_emb = NumEmb(n_num, emb_num)
        self.cat_emb = nn.ModuleList([nn.Embedding(c, emb_cat) for c in cat_cards])
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
               "leaky": nn.LeakyReLU}[activation]
        in_dim = n_num * emb_num + len(cat_cards) * emb_cat
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x_num, x_cat):
        e_num = self.num_emb(x_num).flatten(1)
        e_cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1)
        return self.head(self.backbone(torch.cat([e_num, e_cat], dim=1)))


# ----- training per arch -------------------------------------------------
ARCHS = [
    dict(name="arch_a", hidden=(128, 64),           activation="relu",  dropout=0.20, seed=42,   epochs=150, lr=1e-3, wd=1e-4),
    dict(name="arch_b", hidden=(96, 96, 96),        activation="gelu",  dropout=0.15, seed=7,    epochs=150, lr=1e-3, wd=1e-4),
    dict(name="arch_c", hidden=(256, 128),          activation="relu",  dropout=0.30, seed=123,  epochs=150, lr=8e-4, wd=2e-4),
    dict(name="arch_d", hidden=(192, 96, 48),       activation="tanh",  dropout=0.10, seed=2024, epochs=150, lr=1e-3, wd=1e-4),
    dict(name="arch_e", hidden=(128, 128, 64, 32),  activation="leaky", dropout=0.20, seed=9999, epochs=150, lr=1e-3, wd=1e-4),
]


def train_one(arch, X_num_o, X_cat_o, y_o, cat_cards, log_prior):
    torch.manual_seed(arch["seed"])
    np.random.seed(arch["seed"])
    model = TabMLP(
        n_num=X_num_o.shape[1], cat_cards=cat_cards,
        hidden=arch["hidden"], activation=arch["activation"],
        dropout=arch["dropout"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=arch["lr"], weight_decay=arch["wd"])
    log_prior_t = torch.from_numpy(log_prior).to(DEVICE)

    Xn = torch.from_numpy(X_num_o).float().to(DEVICE)
    Xc = torch.from_numpy(X_cat_o).long().to(DEVICE)
    yt = torch.from_numpy(y_o).long().to(DEVICE)

    batch = 512
    n = len(y_o)
    for epoch in range(arch["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        running = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(Xn[idx], Xc[idx])
            logits_bs = logits + log_prior_t.unsqueeze(0)
            loss = F.cross_entropy(logits_bs, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(idx)
        if (epoch + 1) % 30 == 0 or epoch == arch["epochs"] - 1:
            # quick train bal_acc (small dataset, cheap)
            model.eval()
            with torch.no_grad():
                p = F.softmax(model(Xn, Xc), dim=1).cpu().numpy()
            bal_o = balanced_accuracy_score(y_o, p.argmax(axis=1))
            log(f"    [{arch['name']}] ep {epoch+1:3d}/{arch['epochs']} "
                f"loss {running/n:.4f}  train bal {bal_o:.4f}")
    return model, n_params


@torch.no_grad()
def predict_probs(model, X_num, X_cat, chunk=32768):
    model.eval()
    out = np.zeros((len(X_num), 3), dtype=np.float32)
    for i in range(0, len(X_num), chunk):
        xn = torch.from_numpy(X_num[i:i+chunk]).float().to(DEVICE)
        xc = torch.from_numpy(X_cat[i:i+chunk]).long().to(DEVICE)
        out[i:i+chunk] = F.softmax(model(xn, xc), dim=1).cpu().numpy()
    return out


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


def main():
    log(f"torch {torch.__version__}  device={DEVICE}")

    # --- load all three sources ---
    log("loading 10k original")
    with zipfile.ZipFile(DATA_DIR / "archive.zip") as zf:
        with zf.open("irrigation_prediction.csv") as fh:
            orig = pd.read_csv(fh)
    log(f"  original: {orig.shape}")

    log("loading synthetic train + test")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    log(f"  train: {tr.shape}  test: {te.shape}")

    # --- feature engineering ---
    log("building distance features on all three")
    orig = add_distance_features(orig)
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    # --- column schema ---
    drop = {TARGET, ID}
    num_cols = [c for c in tr.columns
                if c not in drop and pd.api.types.is_numeric_dtype(tr[c])]
    cat_cols = [c for c in tr.columns if c not in drop and c not in num_cols]
    log(f"  {len(num_cols)} numeric + {len(cat_cols)} categorical")
    log(f"  cat cols: {cat_cols}")

    # --- unified cat vocab over train U test U original ---
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
    log(f"  cat cardinalities: {dict(zip(cat_cols, cat_cards))}")

    # --- numeric standardization with SYNTHETIC-train stats ---
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
    log(f"  synthetic priors: {dict(zip(CLASSES, prior_tr.round(4)))}")
    log(f"  original  priors: {dict(zip(CLASSES, prior_orig.round(4)))}")

    # NN trains on ORIGINAL -> use original's log-prior for Balanced Softmax
    log_prior_orig = np.log(prior_orig).astype(np.float32)

    # --- train each arch on 10k original, predict on 630k + 270k ---
    oof_each = np.zeros((len(ARCHS), len(tr), 3), dtype=np.float32)
    test_each = np.zeros((len(ARCHS), len(te), 3), dtype=np.float32)
    orig_each = np.zeros((len(ARCHS), len(orig), 3), dtype=np.float32)
    per_arch_log = []

    for i, arch in enumerate(ARCHS):
        t0 = time.time()
        log(f"--- [{arch['name']}] training on 10k original ---")
        model, n_params = train_one(
            arch, X_num_orig, X_cat_orig, y_orig, cat_cards, log_prior_orig,
        )
        log(f"  params: {n_params:,}")
        orig_each[i] = predict_probs(model, X_num_orig, X_cat_orig)
        oof_each[i]  = predict_probs(model, X_num_tr, X_cat_tr)
        test_each[i] = predict_probs(model, X_num_te, X_cat_te)
        orig_bal = balanced_accuracy_score(y_orig, orig_each[i].argmax(axis=1))
        synth_bal = balanced_accuracy_score(y_tr, oof_each[i].argmax(axis=1))
        log(f"  [{arch['name']}] orig train_acc(bal)={orig_bal:.5f}  "
            f"synth_bal={synth_bal:.5f}  ({time.time()-t0:.0f}s)")
        per_arch_log.append({
            "name": arch["name"],
            "params": n_params,
            "hidden": list(arch["hidden"]),
            "activation": arch["activation"],
            "dropout": arch["dropout"],
            "seed": arch["seed"],
            "epochs": arch["epochs"],
            "orig_bal_argmax": float(orig_bal),
            "synth_bal_argmax": float(synth_bal),
            "seconds": round(time.time() - t0, 1),
        })

    # --- ensemble = mean over archs (in prob space) ---
    oof_ens = oof_each.mean(axis=0)
    test_ens = test_each.mean(axis=0)

    # diagnostics on synthetic train
    argmax_bal = balanced_accuracy_score(y_tr, oof_ens.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y_tr, (oof_ens / prior_tr).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof_ens, y_tr, prior_tr)
    log(f"ensemble synth_bal argmax={argmax_bal:.5f}  "
        f"reweight={reweight_bal:.5f}  tuned={tuned_bal:.5f}")

    # confusion matrix at tuned operating point
    log_oof = np.log(np.clip(oof_ens, 1e-9, 1.0))
    cm = confusion_matrix(y_tr, (log_oof + bias).argmax(axis=1))
    log(f"tuned OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # agreement diagnostic vs greedy (if available)
    greedy_path = OOF_DIR / "oof_greedy_blend.npy"
    jaccard = None
    if greedy_path.exists():
        gr = np.load(greedy_path)
        ens_pred = oof_ens.argmax(axis=1)
        gr_pred = gr.argmax(axis=1)
        e_ens = set(np.where(ens_pred != y_tr)[0])
        e_gr = set(np.where(gr_pred != y_tr)[0])
        inter = len(e_ens & e_gr)
        union = len(e_ens | e_gr) or 1
        jaccard = inter / union
        log(f"error Jaccard vs greedy = {jaccard:.4f}  "
            f"(ens errs={len(e_ens)}, greedy errs={len(e_gr)}, inter={inter})")

    # --- save artifacts ---
    np.save(OOF_DIR / "oof_nn_orig_ens.npy", oof_ens)
    np.save(OOF_DIR / "test_nn_orig_ens.npy", test_ens)
    np.save(OOF_DIR / "oof_nn_orig_each.npy", oof_each)   # (5, 630k, 3)
    np.save(OOF_DIR / "test_nn_orig_each.npy", test_each)
    log(f"saved oof_nn_orig_ens + test_nn_orig_ens to {OOF_DIR}/")

    with open(OOF_DIR / "nn_orig_features_results.json", "w") as f:
        json.dump({
            "n_archs": len(ARCHS),
            "archs": per_arch_log,
            "n_num": len(num_cols),
            "n_cat": len(cat_cols),
            "cat_cards": cat_cards,
            "synth_prior": prior_tr.tolist(),
            "orig_prior": prior_orig.tolist(),
            "log_bias": bias.tolist(),
            "synth_argmax_bal": float(argmax_bal),
            "synth_reweight_bal": float(reweight_bal),
            "synth_tuned_bal": float(tuned_bal),
            "jaccard_vs_greedy": jaccard,
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
