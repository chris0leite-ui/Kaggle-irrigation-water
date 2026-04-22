"""NumEmb MLPs: v8 specialist {6,7,8} + v9 training-data-routed {0,1,2}.

Two NN-architectural experiments paralleling the LB-winning tree-side
tricks:

v8 — specialist on dgp_score in {6,7,8}
    Train MLP ONLY on rows with score in {6,7,8} (56k rows, 69% Medium,
    31% High — the only sub-domain with minority-class mass in the
    20-80% range that made xgb_specialist_678 add +0.00019 OOF). MLP
    should match the smooth NN-generator decision surface at the
    Medium↔High flip band better than axis-aligned XGB.
    Predict on FULL val fold so caller decides override policy.
    Arch: [384, 256, 192, 128] (~200k params), heavier dropout 0.40 on
    the tiny training set.

v9 — training-data routed, exclude score in {0,1,2}
    Train MLP on rows with score NOT in {0,1,2} (359k rows = 57% of
    train, but ~100% of the flip-band rows). At inference, route
    score-{0,1,2} rows to the rule (one-hot Low) exactly like
    xgb_dist_routed_v3. Direct NN analog of the +0.00047-LB-winning
    training-data-engineering trick.
    Arch: [768, 512, 384, 256] (same as v5, same capacity), ~1M params.
    Prior rebalanced on non-routed rows for Balanced Softmax.

Both use the 43-feature dist set, same 5-fold StratifiedKFold(seed=42)
split as every other OOF. Fold-1 Jaccard gate vs BOTH greedy and
xgb_nonrule (since xgb_nonrule is already in our LB-best stack at α=0.15).
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path

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
ACTIVE_STAGES = ("Flowering", "Vegetative")

EMB_NUM = 8
EMB_CAT = 16
BATCH = 4096
LR = 1e-3
WD = 1e-4
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0

SPEC_SCORES = (6, 7, 8)
ROUTED_SCORES = (0, 1, 2)

OUT = Path("/kaggle/working"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


class NumEmb(nn.Module):
    def __init__(self, n_feat, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, emb_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, emb_dim))
    def forward(self, x):
        return x.unsqueeze(-1) * self.w + self.b


class TabMLP(nn.Module):
    def __init__(self, n_num, cat_cards, hidden, dropout,
                 emb_num=EMB_NUM, emb_cat=EMB_CAT, n_classes=3):
        super().__init__()
        self.num_emb = NumEmb(n_num, emb_num)
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
        e_num = self.num_emb(x_num).flatten(1)
        e_cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1)
        h = torch.cat([e_num, e_cat], dim=1)
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


def train_one(variant, hidden, dropout, epochs, train_mask_fn,
              X_num, X_cat, X_num_te, X_cat_te, y, score, score_te,
              cat_cards, gate_oof, gate_xgb_nonrule):
    """Train MLP with fold-specific train_mask. OOF stores MLP probs on
    FULL val fold (callers decide override/routing at eval time)."""
    log(f"=== variant {variant} | hidden={hidden} | dropout={dropout} | epochs={epochs} ===")
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test_probs = np.zeros((len(X_num_te), 3), dtype=np.float64)
    fold_logs = []

    X_num_t = torch.from_numpy(X_num).float()
    X_cat_t = torch.from_numpy(X_cat).long()
    X_num_te_t = torch.from_numpy(X_num_te).float()
    X_cat_te_t = torch.from_numpy(X_cat_te).long()
    y_t = torch.from_numpy(y).long()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        tr_mask = train_mask_fn(tr_idx, score)
        tr_sel = tr_idx[tr_mask]
        log(f"  fold {fold+1}/{N_FOLDS} | train rows (post-filter): {len(tr_sel)} / {len(tr_idx)} "
            f"| val: {len(va_idx)}")

        # recompute prior on actually-used training rows for Balanced Softmax
        prior_fold = np.bincount(y[tr_sel], minlength=3) / max(1, len(tr_sel))
        # guard against zero-prior classes (v8 spec has no Low)
        safe_prior = np.where(prior_fold > 0, prior_fold, 1e-6)
        log_prior_t = torch.from_numpy(np.log(safe_prior).astype(np.float32)).to(DEVICE)

        ds = TensorDataset(X_num_t[tr_sel], X_cat_t[tr_sel], y_t[tr_sel])
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

        model = TabMLP(n_num=X_num.shape[1], cat_cards=cat_cards,
                       hidden=hidden, dropout=dropout).to(DEVICE)
        if fold == 0:
            log(f"  {variant} params: {sum(p.numel() for p in model.parameters()):,}")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        total = epochs * max(1, len(loader)); warm = int(WARMUP_FRAC * total); step = 0
        best_bal = -1.0; best_probs = None
        for epoch in range(epochs):
            model.train(); running = 0.0
            for xn, xc, yb in loader:
                lr_now = cosine_lr(step, total, warm, LR)
                for g in opt.param_groups: g["lr"] = lr_now
                xn = xn.to(DEVICE); xc = xc.to(DEVICE); yb = yb.to(DEVICE)
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
                for i in range(0, len(va_idx), bs):
                    idx_chunk = va_idx[i:i+bs]
                    xn = X_num_t[idx_chunk].to(DEVICE)
                    xc = X_cat_t[idx_chunk].to(DEVICE)
                    vl.append(model(xn, xc).cpu())
                val_logits = torch.cat(vl, 0).numpy()
            val_probs = torch.softmax(torch.from_numpy(val_logits), 1).numpy()
            bal = balanced_accuracy_score(y[va_idx], val_probs.argmax(1))
            if bal > best_bal:
                best_bal = bal; best_probs = val_probs
        log(f"  fold {fold+1} best val bal_acc (full fold) {best_bal:.5f}  ({time.time()-t0:.0f}s)")
        oof[va_idx] = best_probs

        # test pass
        model.eval()
        with torch.no_grad():
            tl = []; bs = 16384
            for i in range(0, len(X_num_te_t), bs):
                xn = X_num_te_t[i:i+bs].to(DEVICE)
                xc = X_cat_te_t[i:i+bs].to(DEVICE)
                tl.append(model(xn, xc).cpu())
            tp = torch.softmax(torch.cat(tl, 0), 1).numpy()
        test_probs += tp / N_FOLDS

        fold_logs.append({"fold": fold+1, "val_bal_acc_best": float(best_bal),
                          "seconds": round(time.time()-t0, 1),
                          "train_rows_post_filter": int(len(tr_sel))})
        if fold == 0:
            mlp_pred = oof[va_idx].argmax(1)
            err_m = set(va_idx[mlp_pred != y[va_idx]])
            err_g = set(va_idx[gate_oof[va_idx].argmax(1) != y[va_idx]])
            err_x = set(va_idx[gate_xgb_nonrule[va_idx].argmax(1) != y[va_idx]])
            j_g = len(err_m & err_g) / max(1, len(err_m | err_g))
            j_x = len(err_m & err_x) / max(1, len(err_m | err_x))
            log(f"  fold-1 jaccard vs greedy={j_g:.4f} | vs xgb_nonrule={j_x:.4f}")
            fold_logs[-1]["jaccard_vs_greedy"] = j_g
            fold_logs[-1]["jaccard_vs_xgb_nonrule"] = j_x

    argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned_bal = tune_bias(oof, y, prior)
    log(f"{variant} OOF (full 630k) argmax={argmax_bal:.5f} tuned={tuned_bal:.5f} "
        f"bias={bias.round(4).tolist()}")

    np.save(OUT / f"oof_mlp_{variant}.npy", oof)
    np.save(OUT / f"test_mlp_{variant}.npy", test_probs)
    return {
        "variant": variant,
        "hidden": hidden,
        "dropout": dropout,
        "epochs": epochs,
        "argmax_bal_acc_full": float(argmax_bal),
        "tuned_bal_acc_full": float(tuned_bal),
        "log_bias_full": bias.tolist(),
        "fold_logs": fold_logs,
    }


def build_v9_routed_oof(oof_mlp, score_vec):
    """Replace score-{0,1,2} rows with rule prediction (one-hot Low=[1,0,0])."""
    out = oof_mlp.copy()
    mask = np.isin(score_vec, ROUTED_SCORES)
    out[mask] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return out, int(mask.sum())


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
    gate_xgb_nonrule = np.load(find_one("oof_xgb_nonrule.npy"))
    log(f"train {tr.shape} test {te.shape}")

    log("feature engineering")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    cat_cols = [c for c in tr.columns if c not in (TARGET, ID) and tr[c].dtype == object]
    num_cols = [c for c in tr.columns if c not in (TARGET, ID) + tuple(cat_cols)]
    log(f"features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    cat_cards = []
    for c in cat_cols:
        vocab = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
        mp = {v: i for i, v in enumerate(vocab)}
        tr[c] = tr[c].astype(str).map(mp).astype("int64")
        te[c] = te[c].astype(str).map(mp).astype("int64")
        cat_cards.append(len(vocab))

    X_num_tr = tr[num_cols].to_numpy(dtype=np.float32)
    X_num_te = te[num_cols].to_numpy(dtype=np.float32)
    mu = X_num_tr.mean(0, keepdims=True); sd = X_num_tr.std(0, keepdims=True) + 1e-6
    X_num_tr = (X_num_tr - mu) / sd; X_num_te = (X_num_te - mu) / sd
    X_cat_tr = tr[cat_cols].to_numpy(dtype=np.int64)
    X_cat_te = te[cat_cols].to_numpy(dtype=np.int64)
    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)

    score_tr = tr["dgp_score"].to_numpy(dtype=np.int8)
    score_te = te["dgp_score"].to_numpy(dtype=np.int8)
    log(f"score==(6,7,8) train rows: {np.isin(score_tr, SPEC_SCORES).sum()}")
    log(f"score==(0,1,2) train rows: {np.isin(score_tr, ROUTED_SCORES).sum()}")

    # --- v8: specialist on score in {6,7,8} ---
    r_v8 = train_one(
        "v8_spec_678",
        hidden=[384, 256, 192, 128],
        dropout=0.40,
        epochs=40,
        train_mask_fn=lambda tr_idx, score: np.isin(score[tr_idx], SPEC_SCORES),
        X_num=X_num_tr, X_cat=X_cat_tr,
        X_num_te=X_num_te, X_cat_te=X_cat_te,
        y=y, score=score_tr, score_te=score_te,
        cat_cards=cat_cards,
        gate_oof=gate_oof, gate_xgb_nonrule=gate_xgb_nonrule,
    )

    # --- v9: routed training, exclude score in {0,1,2} from training ---
    r_v9_raw = train_one(
        "v9_routed",
        hidden=[768, 512, 384, 256],
        dropout=0.25,
        epochs=30,
        train_mask_fn=lambda tr_idx, score: ~np.isin(score[tr_idx], ROUTED_SCORES),
        X_num=X_num_tr, X_cat=X_cat_tr,
        X_num_te=X_num_te, X_cat_te=X_cat_te,
        y=y, score=score_tr, score_te=score_te,
        cat_cards=cat_cards,
        gate_oof=gate_oof, gate_xgb_nonrule=gate_xgb_nonrule,
    )

    # v9 with inference-time routing: score-{0,1,2} rows replaced with one-hot Low
    log("v9 with inference-time rule routing on score-{0,1,2} rows")
    oof_v9 = np.load(OUT / "oof_mlp_v9_routed.npy")
    test_v9 = np.load(OUT / "test_mlp_v9_routed.npy")
    oof_v9_r, tr_routed = build_v9_routed_oof(oof_v9, score_tr)
    test_v9_r, te_routed = build_v9_routed_oof(test_v9, score_te)

    prior = np.bincount(y, minlength=3) / len(y)
    argmax_r = balanced_accuracy_score(y, oof_v9_r.argmax(1))
    bias_r, tuned_r = tune_bias(oof_v9_r, y, prior)
    log(f"v9_routed_inference OOF: argmax={argmax_r:.5f} tuned={tuned_r:.5f}  "
        f"train routed rows={tr_routed}  test routed rows={te_routed}")

    np.save(OUT / "oof_mlp_v9_routed_inference.npy", oof_v9_r)
    np.save(OUT / "test_mlp_v9_routed_inference.npy", test_v9_r)

    # fold-1-style jaccard vs greedy on the routed-inference OOF (full set)
    err_m_all = np.where(oof_v9_r.argmax(1) != y)[0]
    err_g_all = np.where(gate_oof.argmax(1) != y)[0]
    err_x_all = np.where(gate_xgb_nonrule.argmax(1) != y)[0]
    def jac(a, b):
        A = set(a); B = set(b)
        return len(A & B) / max(1, len(A | B))
    log(f"v9 routed-inference full-OOF jaccard vs greedy={jac(err_m_all, err_g_all):.4f}  "
        f"vs xgb_nonrule={jac(err_m_all, err_x_all):.4f}")

    with open(OUT / "mlp_route_spec_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "v8_spec_678": r_v8,
            "v9_routed_raw": r_v9_raw,
            "v9_routed_inference": {
                "argmax_bal_acc_full": float(argmax_r),
                "tuned_bal_acc_full": float(tuned_r),
                "log_bias_full": bias_r.tolist(),
                "train_routed_rows": tr_routed,
                "test_routed_rows": te_routed,
                "jaccard_vs_greedy_full": jac(err_m_all, err_g_all),
                "jaccard_vs_xgb_nonrule_full": jac(err_m_all, err_x_all),
            },
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
