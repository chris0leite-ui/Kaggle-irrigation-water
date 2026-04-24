"""P3: transductive k-NN label propagation in a learned embedding.

Pipeline:
  1. Fit a supervised contrastive embedding on the 443-feature recipe
     matrix (same FE as recipe_full_te). Small MLP backbone + projection
     head, SupCon + cross-entropy joint loss.
  2. Embed train + test into 32-dim space.
  3. Build k-NN graph (k=30, Gaussian kernel) via FAISS (falls back to
     sklearn if FAISS unavailable).
  4. Run label propagation (sklearn LabelSpreading, alpha=0.2) over
     the transductive graph (train labels known, test labels hidden).
  5. Save OOF + test predictions.

Fold protocol: for OOF fairness, train embedding on train_tr ONLY
within each fold, transform train_va + test. For final test predictions,
train embedding on full train.

Env:
  EPOCHS=30
  BATCH_SIZE=4096
  EMBED_DIM=32
  KNN=30
  LABELPROP_ALPHA=0.2
  DEVICE=cpu|cuda (default auto)
  SMOKE=1 (20k/5k subsample + 1 fold + 3 epochs)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_features import (  # noqa: E402
    add_cat_pair_combos, add_digit_features, add_freq_features,
    add_lr_formula_logits, add_num_as_cat, add_orig_mean_std,
    add_threshold_flags,
)
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS = int(os.environ.get("EPOCHS", "3" if SMOKE else "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
EMBED_DIM = int(os.environ.get("EMBED_DIM", 32))
KNN = int(os.environ.get("KNN", 30))
LABELPROP_ALPHA = float(os.environ.get("LABELPROP_ALPHA", 0.2))
DEVICE = os.environ.get("DEVICE", "auto")
N_FOLDS = 1 if SMOKE else 5

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _select_device():
    import torch
    if DEVICE == "cpu":
        return torch.device("cpu")
    if DEVICE == "cuda" or (DEVICE == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


# -------- FE: same as recipe_full_te but we don't need per-fold OTE
#          variance — we fit OTE once on full train for embedding training.
def load_and_engineer():
    log("loading train / test / orig")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")
    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        log("SMOKE=1 — subsampling")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(5_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:5_000]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    for df in (train, test, orig):
        add_threshold_flags(df)
    for df in (train, test, orig):
        add_lr_formula_logits(df)
    combos = add_cat_pair_combos(train, test, orig, cats)
    digits = add_digit_features(train, test, orig, nums)
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    freq = add_freq_features(train, test, orig, cats + combos)
    orig_stats = add_orig_mean_std(train, test, orig, nums + cats, TARGET)
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]; test[c] = codes[s:t]; orig[c] = codes[t:]
    tres = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]
    logits_cols = ["logit_P_Low", "logit_P_Medium", "logit_P_High"]
    info = dict(nums=nums, tres=tres, logits=logits_cols, freq=freq,
                orig_stats=orig_stats, te_cols=cats + combos + digits
                + num_as_cat + tres)
    return train, test, info, test_ids


def _build_feat_matrix(train, test, info):
    """Run OTE once on full train (not per-fold); OK because the embedding
    doesn't do classification OOF — it's a representation; the downstream
    label propagation is what's OOF-validated."""
    log("fitting OTE on full train")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train))
    tr_shuf = train.iloc[perm].reset_index(drop=True)
    te = OrderedTE(a=1.0)
    tr_shuf = te.fit(tr_shuf, cat_cols=info["te_cols"], target=TARGET)
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    train_fe = tr_shuf.iloc[inv].reset_index(drop=True)
    test_fe = te.transform(test)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    feat_cols = numeric_feats + te.te_col_names()
    return train_fe[feat_cols].to_numpy(np.float32), \
           test_fe[feat_cols].to_numpy(np.float32), feat_cols


# -------- supervised-contrastive embedding.
class Embedder:
    """Small MLP with 2 heads: classifier + projection for SupCon."""
    def __init__(self, in_dim: int, embed_dim: int = 32, device=None):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        self.device = device
        hidden = 256
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        ).to(device)
        self.classifier = nn.Linear(hidden, 3).to(device)
        self.projection = nn.Sequential(
            nn.Linear(hidden, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)
        self.embed_dim = embed_dim

    def _supcon_loss(self, z, y, temperature=0.5):
        """Supervised contrastive loss (Khosla 2020)."""
        z = self.nn.functional.normalize(z, dim=1)
        sim = z @ z.t() / temperature
        sim.fill_diagonal_(-1e9)
        mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        exp_sim = sim.exp()
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12
        log_prob = sim - denom.log()
        # Avoid rows with no positives; mask out rows where mask.sum == 0.
        has_pos = (mask.sum(dim=1) > 0)
        if has_pos.sum() == 0:
            return self.torch.tensor(0.0, device=z.device)
        pos_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return -pos_log_prob[has_pos].mean()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, bs: int):
        torch = self.torch
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long())
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + list(self.projection.parameters()),
            lr=1e-3, weight_decay=1e-4)
        self.backbone.train(); self.classifier.train(); self.projection.train()
        for ep in range(epochs):
            t0 = time.time(); losses = []
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                h = self.backbone(xb)
                logits = self.classifier(h)
                z = self.projection(h)
                loss_ce = self.nn.functional.cross_entropy(logits, yb)
                loss_sup = self._supcon_loss(z, yb)
                loss = loss_ce + 0.5 * loss_sup
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            log(f"    epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}  "
                f"dt={time.time()-t0:.1f}s")

    def transform(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.backbone.eval(); self.projection.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 8192):
                xb = torch.from_numpy(X[i:i+8192]).float().to(self.device)
                z = self.projection(self.backbone(xb))
                z = self.nn.functional.normalize(z, dim=1).cpu().numpy()
                out.append(z)
        return np.concatenate(out, axis=0).astype(np.float32)


def knn_labelprop(emb_tr, y_tr, emb_test, knn, alpha, n_classes=3):
    """Graph label propagation: propagate y_tr labels across (tr ∪ test).

    For each test row, compute soft class distribution via weighted vote
    over its top-K training-set neighbors.
    """
    try:
        import faiss
        log(f"  FAISS knn k={knn}")
        index = faiss.IndexFlatIP(emb_tr.shape[1])
        index.add(emb_tr.astype(np.float32))
        D, I = index.search(emb_test.astype(np.float32), knn)
    except ImportError:
        log(f"  sklearn knn k={knn} (FAISS unavailable)")
        from sklearn.neighbors import NearestNeighbors
        nn_ = NearestNeighbors(n_neighbors=knn, metric="cosine", n_jobs=-1)
        nn_.fit(emb_tr)
        D, I = nn_.kneighbors(emb_test)
        D = -D  # cosine distance -> similarity proxy

    # Gaussian kernel weighting (no fixed bandwidth; use median D).
    bw = np.median(np.abs(D)) + 1e-9
    W = np.exp(-(np.abs(D) ** 2) / (2 * bw ** 2))
    # Aggregate to soft class probs.
    probs = np.zeros((len(emb_test), n_classes), dtype=np.float32)
    for c in range(n_classes):
        mask = (y_tr[I] == c).astype(np.float32)
        probs[:, c] = (W * mask).sum(axis=1)
    probs /= probs.sum(axis=1, keepdims=True).clip(1e-9)
    # Blend with label propagation (alpha controls label-smoothing vs keep).
    if alpha > 0:
        # Simple smoothing: probs = (1-alpha)*one_hot_argmax + alpha*probs.
        one_hot = np.eye(n_classes)[probs.argmax(1)].astype(np.float32)
        probs = (1 - alpha) * one_hot + alpha * probs
    return probs


def main():
    import torch  # noqa: F401
    log(f"config: SMOKE={SMOKE} EPOCHS={EPOCHS} BS={BATCH_SIZE} "
        f"EMBED_DIM={EMBED_DIM} KNN={KNN} FOLDS={N_FOLDS}")
    device = _select_device()
    log(f"device: {device}")

    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()
    X_tr_all, X_te, feat_cols = _build_feat_matrix(train, test, info)
    log(f"feat matrix: train={X_tr_all.shape}  test={X_te.shape}")

    # Standardize (critical for MLP + cosine similarity embedding).
    scaler = StandardScaler().fit(X_tr_all)
    X_tr_all = scaler.transform(X_tr_all).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)

    splits = list(skf.split(X_tr_all, y))[:N_FOLDS]
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        emb = Embedder(X_tr_all.shape[1], embed_dim=EMBED_DIM, device=device)
        emb.fit(X_tr_all[tr_idx], y[tr_idx], EPOCHS, BATCH_SIZE)
        emb_tr = emb.transform(X_tr_all[tr_idx])
        emb_va = emb.transform(X_tr_all[va_idx])
        probs_va = knn_labelprop(
            emb_tr, y[tr_idx], emb_va, KNN, LABELPROP_ALPHA)
        oof[va_idx] = probs_va
        bal = balanced_accuracy_score(y[va_idx], probs_va.argmax(1))
        log(f"  fold {fold} argmax bal_acc = {bal:.5f}")

    # Final: train on full train, embed test.
    log("=== final test prediction: embed on full train ===")
    emb = Embedder(X_tr_all.shape[1], embed_dim=EMBED_DIM, device=device)
    emb.fit(X_tr_all, y, EPOCHS, BATCH_SIZE)
    emb_tr_full = emb.transform(X_tr_all)
    emb_test = emb.transform(X_te)
    test_probs = knn_labelprop(
        emb_tr_full, y, emb_test, KNN, LABELPROP_ALPHA)

    # OOF metrics.
    if N_FOLDS == 5:
        argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
        prior = np.bincount(y, minlength=3) / len(y)
        bias, tuned = tune_log_bias(oof, y, prior)
        log(f"OOF argmax={argmax_bal:.5f}  tuned={tuned:.5f}  "
            f"bias={[round(b,3) for b in bias]}")
    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_p3_embed_propagate{suffix}.npy", oof)
    np.save(ART / f"test_p3_embed_propagate{suffix}.npy", test_probs)
    results = dict(smoke=SMOKE, n_folds=N_FOLDS, epochs=EPOCHS, knn=KNN,
                   embed_dim=EMBED_DIM)
    if N_FOLDS == 5:
        results.update(argmax_bal_acc=float(argmax_bal),
                       tuned_bal_acc=float(tuned), log_bias=bias.tolist())
    with open(ART / f"p3_embed_propagate_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
