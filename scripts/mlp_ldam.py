"""MLP+DGP with LDAM loss + Deferred Re-Weighting (Cao et al. 2019).

Why: v3 Balanced Softmax closed the raw-vs-tuned gap for the MLP but
left a persistent ~0.007 shortfall vs LGBM+DGP. BalSoft rebalances
with log(pi), but it doesn't enforce class-dependent margins — every
class has the same decision surface curvature. LDAM adds margins
m_c ∝ n_c^{-1/4} (~2.3× larger for High than Low at our prior
distribution), which specifically targets the Medium↔High confusion
band where our errors concentrate.

Schedule (DRW, Cao 2019):
- Phase 1 (first ~½ of epochs): plain CE, no class weights → backbone
  learns representation freely without tail-class overfitting.
- Phase 2 (remaining epochs): LDAM loss + effective-number class
  weights (β=0.9999) → margins + re-weighting on a well-trained
  backbone.

Uses fixed epoch budget (no early stopping) so Phase 2 always fires.
Selection on raw-argmax val bal_acc across all epochs.
"""
from __future__ import annotations

import json
import math
import time
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

BATCH_SIZE = 4096
EVAL_BATCH = 16384
TOTAL_EPOCHS = 30
DRW_START_EPOCH = 15  # switch from CE to LDAM + CB weights here
LR = 2e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.15
HIDDEN = [256, 128, 64]
MAX_M = 0.5        # LDAM max-margin
LDAM_SCALE = 15.0  # logits scale (s); smaller than face-recog default 30

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_dgp_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float)
    rm = out["Rainfall_mm"].astype(float)
    tc = out["Temperature_C"].astype(float)
    ws = out["Wind_Speed_kmh"].astype(float)
    out["dgp_dry"] = (sm < 25).astype(np.int8)
    out["dgp_norain"] = (rm < 300).astype(np.int8)
    out["dgp_hot"] = (tc > 30).astype(np.int8)
    out["dgp_windy"] = (ws > 10).astype(np.int8)
    out["dgp_nomulch"] = (out["Mulching_Used"].astype(str) == "No").astype(np.int8)
    out["dgp_kc"] = np.where(
        out["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]),
        2, 0,
    ).astype(np.int8)
    out["dgp_score"] = (
        2 * (out["dgp_dry"] + out["dgp_norain"])
        + (out["dgp_hot"] + out["dgp_windy"] + out["dgp_nomulch"])
        + out["dgp_kc"]
    ).astype(np.int8)
    out["dgp_dist_moist"] = sm - 25.0
    out["dgp_dist_rain"] = rm - 300.0
    out["dgp_dist_temp"] = tc - 30.0
    out["dgp_dist_wind"] = ws - 10.0
    out["dgp_abs_moist"] = out["dgp_dist_moist"].abs()
    out["dgp_abs_rain"] = out["dgp_dist_rain"].abs()
    out["dgp_abs_temp"] = out["dgp_dist_temp"].abs()
    out["dgp_abs_wind"] = out["dgp_dist_wind"].abs()
    return out


log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
tr = add_dgp_features(tr)
te = add_dgp_features(te)

raw_cats = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
num_cols = [c for c in tr.columns
            if c not in raw_cats + [TARGET, ID]
            and pd.api.types.is_numeric_dtype(tr[c])]

cat_vocabs: dict[str, dict[str, int]] = {}
for c in raw_cats:
    vals = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
    cat_vocabs[c] = {v: i for i, v in enumerate(vals)}
    tr[c] = tr[c].astype(str).map(cat_vocabs[c]).astype(np.int64)
    te[c] = te[c].astype(str).map(cat_vocabs[c]).astype(np.int64)
cat_cards = {c: len(cat_vocabs[c]) for c in raw_cats}

X_num = tr[num_cols].to_numpy(dtype=np.float32)
X_cat = tr[raw_cats].to_numpy(dtype=np.int64)
X_num_te = te[num_cols].to_numpy(dtype=np.float32)
X_cat_te = te[raw_cats].to_numpy(dtype=np.int64)
y = tr[TARGET].map(CLS2IDX).to_numpy().astype(np.int64)
prior = np.bincount(y) / len(y)
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")


def emb_dim(card: int) -> int:
    return max(2, min(8, int(math.ceil(card / 2))))


class TabMLP(nn.Module):
    def __init__(self, n_num: int, cat_cards: list[int], hidden: list[int],
                 n_classes: int, dropout: float):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(card, emb_dim(card))
                                   for card in cat_cards])
        emb_total = sum(emb_dim(card) for card in cat_cards)
        in_dim = n_num + emb_total
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb = [self.embs[i](x_cat[:, i]) for i in range(x_cat.size(1))]
        x = torch.cat([x_num] + emb, dim=1)
        return self.net(x)


class LDAMLoss(nn.Module):
    """LDAM loss (Cao et al. 2019): margin m_c = max_m * (n_c^{-1/4})
    normalised so max margin == max_m. Applies subtractive margin to
    the true-class logit before softmax-CE."""

    def __init__(self, cls_num_list: np.ndarray, max_m: float = 0.5,
                 weight: torch.Tensor | None = None, s: float = 30.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list.astype(np.float64)))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        batch_m = self.m_list[target].view(-1, 1)
        x_m = logits - batch_m
        output = torch.where(index, x_m, logits)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


def predict_logits(model: nn.Module, xn: np.ndarray, xc: np.ndarray) -> np.ndarray:
    model.eval()
    out = np.empty((len(xn), len(CLASSES)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(xn), EVAL_BATCH):
            xnb = torch.from_numpy(xn[i:i + EVAL_BATCH])
            xcb = torch.from_numpy(xc[i:i + EVAL_BATCH])
            out[i:i + EVAL_BATCH] = model(xnb, xcb).cpu().numpy()
    return out


def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

log(f"running {N_FOLDS}-fold stratified MLP + LDAM-DRW "
    f"(CE ep 0-{DRW_START_EPOCH-1}, LDAM+CB ep {DRW_START_EPOCH}-{TOTAL_EPOCHS-1})")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num, y)):
    t0 = time.time()

    mu = X_num[tr_idx].mean(axis=0)
    sd = X_num[tr_idx].std(axis=0) + 1e-6
    xn_tr = ((X_num[tr_idx] - mu) / sd).astype(np.float32)
    xn_va = ((X_num[va_idx] - mu) / sd).astype(np.float32)
    xn_te = ((X_num_te - mu) / sd).astype(np.float32)
    xc_tr = X_cat[tr_idx]
    xc_va = X_cat[va_idx]
    xc_te = X_cat_te

    cls_num_list = np.bincount(y[tr_idx])
    # Effective-number class weights (Cui et al. 2019), beta=0.9999
    beta = 0.9999
    eff_num = 1.0 - np.power(beta, cls_num_list)
    cb_weights = (1.0 - beta) / eff_num
    cb_weights = cb_weights / cb_weights.sum() * len(CLASSES)
    cb_weights_t = torch.tensor(cb_weights, dtype=torch.float32)

    ldam_loss = LDAMLoss(cls_num_list, max_m=MAX_M, s=LDAM_SCALE)
    ldam_loss_cb = LDAMLoss(cls_num_list, max_m=MAX_M, s=LDAM_SCALE,
                            weight=cb_weights_t)
    ce_loss = nn.CrossEntropyLoss()

    ds_tr = TensorDataset(torch.from_numpy(xn_tr),
                          torch.from_numpy(xc_tr),
                          torch.from_numpy(y[tr_idx]))
    loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=False)

    model = TabMLP(
        n_num=xn_tr.shape[1],
        cat_cards=[cat_cards[c] for c in raw_cats],
        hidden=HIDDEN,
        n_classes=len(CLASSES),
        dropout=DROPOUT,
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS)

    best_bal = -1.0
    best_logits_va = None
    best_logits_te = None

    for epoch in range(TOTAL_EPOCHS):
        phase = "CE" if epoch < DRW_START_EPOCH else "LDAM+CB"
        loss_fn = ce_loss if epoch < DRW_START_EPOCH else ldam_loss_cb

        model.train()
        loss_sum = 0.0
        n_seen = 0
        for xnb, xcb, yb in loader:
            opt.zero_grad()
            logits = model(xnb, xcb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xnb.size(0)
            n_seen += xnb.size(0)
        sched.step()

        val_logits = predict_logits(model, xn_va, xc_va)
        val_raw = balanced_accuracy_score(y[va_idx], val_logits.argmax(axis=1))
        if val_raw > best_bal + 1e-6:
            best_bal = val_raw
            best_logits_va = val_logits
            best_logits_te = predict_logits(model, xn_te, xc_te)
        log(
            f"  fold {fold+1}/{N_FOLDS} ep {epoch+1:02d} [{phase}]  "
            f"loss={loss_sum/n_seen:.4f}  val_raw={val_raw:.5f}  "
            f"best={best_bal:.5f}"
        )

    assert best_logits_va is not None and best_logits_te is not None
    oof[va_idx] = softmax_np(best_logits_va)
    test_pred += softmax_np(best_logits_te) / N_FOLDS
    fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
    log(
        f"  fold {fold+1}/{N_FOLDS} done  bal_acc(argmax)={fold_bal:.5f}  "
        f"({time.time()-t0:.1f}s)"
    )


def bench(name: str, pred_idx: np.ndarray) -> dict:
    return {
        "name": name,
        "bal_acc": balanced_accuracy_score(y, pred_idx),
        "cm": confusion_matrix(y, pred_idx).tolist(),
    }


results = [
    bench("MLP+LDAM-DRW argmax", oof.argmax(axis=1)),
    bench("MLP+LDAM-DRW prior-reweight argmax", (oof / prior).argmax(axis=1)),
]

log("coord-ascent over per-class log-bias (insurance)")
log_oof = np.log(np.clip(oof, 1e-9, 1.0))


def score_bias(bias: np.ndarray) -> float:
    return balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))


bias = np.zeros(len(CLASSES))
best = score_bias(bias)
grid = np.linspace(-2.5, 2.5, 51)
for _ in range(20):
    improved = False
    for k in range(len(CLASSES)):
        base = bias.copy()
        scores = []
        for g in grid:
            base[k] = bias[k] + g
            scores.append(score_bias(base))
        j = int(np.argmax(scores))
        if scores[j] > best + 1e-6:
            bias[k] = bias[k] + grid[j]
            best = scores[j]
            improved = True
    if not improved:
        break
log(f"  best bias (from zero) = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={best:.5f}")
results.append(bench("MLP+LDAM-DRW tuned log-bias", (log_oof + bias).argmax(axis=1)))

print("\n=== MLP+LDAM-DRW summary (OOF balanced accuracy) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")

print("\nconfusion matrix (rows=true, cols=pred) for best rule:")
best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"best: {best_rule['name']}")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))

np.save(ART_DIR / "oof_mlp_ldam.npy", oof)
np.save(ART_DIR / "test_mlp_ldam.npy", test_pred)
with open(ART_DIR / "mlp_ldam_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "batch_size": BATCH_SIZE,
            "total_epochs": TOTAL_EPOCHS,
            "drw_start_epoch": DRW_START_EPOCH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "hidden": HIDDEN,
            "max_m": MAX_M,
            "ldam_scale": LDAM_SCALE,
            "loss": "LDAM (Cao 2019) + deferred effective-number reweighting",
            "class_priors": prior.tolist(),
            "log_bias_from_zero": bias.tolist(),
            "num_cols": num_cols,
            "cat_cols": raw_cats,
            "cat_cardinalities": cat_cards,
            "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
            "best_rule": best_rule["name"],
        },
        f,
        indent=2,
    )
log(f"OOF + test probs saved to {ART_DIR}/")

argmax_test_idx = test_pred.argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
    OUT_DIR / "submission_mlp_ldam_argmax.csv", index=False
)
tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_mlp_ldam_tuned.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")
