"""Path A — PyTorch MLP on the V10 recipe feature set (~440 cols), Kaggle GPU.

Brainstormed mechanism: leader (Chris Deotte, 0.98219) likely cloned the host
NN architecture and trained on rule-perfect 10k → reproduced flips. We try a
weaker imitation: train a deep PyTorch MLP on the SAME 443-feature recipe
matrix that produces our LB-best 0.98094 XGB primary. Distinct from all 15
prior NN nulls because every prior NN attempt used 19-66 raw-only features;
none saw recipe FE (OTE + digits + FREQ + ORIG_stats + LR-formula logits).

Hypothesis: NN inductive bias (smooth boundaries) + recipe FE (precomputed
soft signal via OTE) may finally clear the magnitude floor that closed all
15 prior NNs.

Wall budget: 1h Kaggle GPU cap (CLAUDE.md hard rule). PROBE config (default,
PROBE=1) runs fold 1 only with full data (504k tr, 126k va, 270k te) to
generate a Jaccard-vs-LB-best signal in ~30 min. Production 5-fold needs a
separate kernel push if PROBE passes.

Outputs (in /kaggle/working/):
  oof_path_a_recipe_mlp[_smoke].npy         (zeros except completed folds)
  test_path_a_recipe_mlp[_smoke].npy
  path_a_recipe_mlp[_smoke]_results.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from functools import reduce
from itertools import combinations
from pathlib import Path

# ========================= environment setup =========================
import numpy as np
import pandas as pd

try:
    import torch
    print(f"[boot] torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch
    print(f"[boot] torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"],
        text=True, timeout=10,
    ).strip()
    print(f"[boot] GPU info: {out}", flush=True)
except Exception as e:
    print(f"[boot] nvidia-smi error: {e}", flush=True)

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

KAGGLE_INPUT = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_one(pattern: str) -> Path:
    for p in KAGGLE_INPUT.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {KAGGLE_INPUT}")

SMOKE = os.environ.get("SMOKE", "1") == "1"  # FIRST PUSH: SMOKE on by default
PROBE = os.environ.get("PROBE", "1") == "1"  # default: 1-fold PROBE
if SMOKE:
    N_FOLDS = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================= inlined recipe features =========================
_LOGIT_COEFS = {
    "Low":    dict(bias=16.3173,
                   soil_lt_25=-11.0237, temp_gt_30=-5.8559,
                   rain_lt_300=-10.8500, wind_gt_10=-5.8284,
                   stage=dict(Flowering=-5.4155, Harvest=5.5073,
                              Sowing=5.2299, Vegetative=-5.4617),
                   mulch=dict(No=-3.0014, Yes=2.8613)),
    "Medium": dict(bias=4.6524,
                   soil_lt_25=0.3290, temp_gt_30=-0.0204,
                   rain_lt_300=0.1542, wind_gt_10=0.0841,
                   stage=dict(Flowering=0.3586, Harvest=-0.1348,
                              Sowing=-0.3547, Vegetative=0.3334),
                   mulch=dict(No=0.1883, Yes=0.0142)),
    "High":   dict(bias=-20.9697,
                   soil_lt_25=10.6947, temp_gt_30=5.8763,
                   rain_lt_300=10.6958, wind_gt_10=5.7444,
                   stage=dict(Flowering=5.0569, Harvest=-5.3725,
                              Sowing=-4.8752, Vegetative=5.1283),
                   mulch=dict(No=2.8131, Yes=-2.8755)),
}


def add_threshold_flags(df):
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(np.int8)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(np.int8)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(np.int8)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(np.int8)
    return ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]


def add_lr_formula_logits(df):
    stage = df["Crop_Growth_Stage"].astype(str).values
    mulch = df["Mulching_Used"].astype(str).values
    soil = df["soil_lt_25"].values
    temp = df["temp_gt_30"].values
    rain = df["rain_lt_300"].values
    wind = df["wind_gt_10"].values
    cols = []
    for cls, coefs in _LOGIT_COEFS.items():
        logit = (coefs["bias"]
                 + coefs["soil_lt_25"] * soil
                 + coefs["temp_gt_30"] * temp
                 + coefs["rain_lt_300"] * rain
                 + coefs["wind_gt_10"] * wind)
        stage_vals = np.array([coefs["stage"].get(s, 0.0) for s in stage])
        mulch_vals = np.array([coefs["mulch"].get(m, 0.0) for m in mulch])
        name = f"logit_P_{cls}"
        df[name] = (logit + stage_vals + mulch_vals).astype(np.float32)
        cols.append(name)
    return cols


def add_cat_pair_combos(train, test, orig, cats):
    new_cols = []
    for c1, c2 in combinations(cats, 2):
        col = f"COMBO_{c1}_{c2}"
        for df in (train, test, orig):
            df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
        combined = pd.concat([train[col], test[col], orig[col]])
        codes, _ = pd.factorize(combined)
        split_tr = len(train)
        split_te = split_tr + len(test)
        train[col] = codes[:split_tr]
        test[col] = codes[split_tr:split_te]
        orig[col] = codes[split_te:]
        new_cols.append(col)
    return new_cols


def add_digit_features(train, test, orig, nums, digit_range=range(-4, 4)):
    cols = []
    for c in nums:
        for k in digit_range:
            name = f"{c}_digit{k}"
            for df in (train, test, orig):
                df[name] = (df[c] // (10.0 ** k) % 10).astype("int8")
            cols.append(name)
    drop = [c for c in cols if test[c].nunique() == 1]
    for c in drop:
        for df in (train, test, orig):
            df.drop(columns=[c], inplace=True)
    return [c for c in cols if c not in drop]


def add_freq_features(train, test, orig, cats):
    new_cols = []
    for c in cats:
        freq = pd.concat([train[c], test[c], orig[c]]).value_counts(normalize=True)
        name = f"FREQ_{c}"
        for df in (train, test, orig):
            df[name] = df[c].map(freq).fillna(0).astype(np.float32)
        new_cols.append(name)
    return new_cols


def add_orig_mean_std(train, test, orig, cols_to_aggregate, target):
    new_cols = []
    for c in cols_to_aggregate:
        stats = orig.groupby(c)[target].agg(["mean", "std"]).reset_index()
        stats.columns = [c, f"ORIG_{c}_mean", f"ORIG_{c}_std"]
        for df_name in ("train", "test"):
            df = {"train": train, "test": test}[df_name]
            merged = df.merge(stats, on=c, how="left")
            df[f"ORIG_{c}_mean"] = merged[f"ORIG_{c}_mean"].fillna(0.5).astype(np.float32).values
            df[f"ORIG_{c}_std"]  = merged[f"ORIG_{c}_std"].fillna(0).astype(np.float32).values
        new_cols += [f"ORIG_{c}_mean", f"ORIG_{c}_std"]
    return new_cols


def add_num_as_cat(train, test, orig, nums):
    new_cols = []
    for c in nums:
        name = f"CAT_{c}"
        for df in (train, test, orig):
            df[name] = df[c].astype(str)
        combined = pd.concat([train[name], test[name], orig[name]])
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[name] = codes[:s]
        test[name] = codes[s:t]
        orig[name] = codes[t:]
        new_cols.append(name)
    return new_cols


# ========================= inlined OrderedTE =========================
class OrderedTE:
    def __init__(self, a=1.0):
        self.a = float(a)
        self.classes_ = None
        self.prior_ = None
        self.stats_ = {}
        self.cols_ = []

    def fit(self, df, cat_cols, target):
        y = df[target].to_numpy()
        self.classes_ = np.array(sorted(pd.unique(y)))
        counts = np.array([(y == k).sum() for k in self.classes_], dtype=np.float64)
        self.prior_ = counts / counts.sum()
        self.cols_ = list(cat_cols)
        te_cols_out = {}
        for c in self.cols_:
            stats_list = []
            key = df[c].to_numpy()
            for k, cls in enumerate(self.classes_):
                y_bin = (df[target] == cls).astype(np.int32).to_numpy()
                grp = pd.DataFrame({c: key, "y": y_bin})
                grouped = grp.groupby(c, observed=True, sort=False)["y"]
                cum_cnt = grouped.cumcount().to_numpy()
                cum_sum_incl = grouped.cumsum().to_numpy()
                cum_sum_excl = cum_sum_incl - y_bin
                prior = self.prior_[k]
                te = (cum_sum_excl + self.a * prior) / (cum_cnt + self.a)
                te_cols_out[f"{c}_TE_cls{cls}"] = te.astype(np.float32)
                agg = grouped.agg(["count", "sum"]).reset_index()
                agg.columns = [c, f"{c}_n_{cls}", f"{c}_s_{cls}"]
                stats_list.append(agg)
            self.stats_[c] = reduce(
                lambda a_df, b_df: a_df.merge(b_df, on=c, how="outer"),
                stats_list,
            )
        te_df = pd.DataFrame(te_cols_out, index=df.index)
        return pd.concat([df, te_df], axis=1)

    def transform(self, df):
        te_cols_out = {}
        for c in self.cols_:
            stats = self.stats_[c]
            merged = df[[c]].merge(stats, on=c, how="left")
            for k, cls in enumerate(self.classes_):
                n_col = f"{c}_n_{cls}"
                s_col = f"{c}_s_{cls}"
                prior = self.prior_[k]
                n = merged[n_col].fillna(0).to_numpy()
                s = merged[s_col].fillna(0).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    te = np.where(
                        n > 0,
                        (s + self.a * prior) / (n + self.a),
                        prior,
                    )
                te_cols_out[f"{c}_TE_cls{cls}"] = te.astype(np.float32)
        te_df = pd.DataFrame(te_cols_out, index=df.index)
        return pd.concat([df, te_df], axis=1)

    def te_col_names(self):
        return [f"{c}_TE_cls{cls}" for c in self.cols_ for cls in self.classes_]


# ========================= inlined log-bias tuner =========================
def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(
                    balanced_accuracy_score(y, (log_oof + base).argmax(axis=1))
                )
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


# ========================= data loading =========================
def load_and_engineer():
    log("listing /kaggle/input/")
    for p in sorted(KAGGLE_INPUT.rglob("*.csv")):
        log(f"  {p}")
    log("loading train / test / orig via rglob")
    train_path = _find_one("train.csv")
    test_path = _find_one("test.csv")
    # Orig dataset CSV — try common names.
    orig_path = None
    for pattern in ("irrigation_prediction.csv", "Irrigation_Prediction.csv",
                    "irrigation-prediction.csv"):
        try:
            orig_path = _find_one(pattern)
            break
        except FileNotFoundError:
            continue
    if orig_path is None:
        # Fall back to any non-train/test csv.
        for p in KAGGLE_INPUT.rglob("*.csv"):
            if p.name not in ("train.csv", "test.csv", "sample_submission.csv"):
                orig_path = p
                break
    if orig_path is None:
        raise FileNotFoundError("no original-dataset CSV found")
    log(f"  train: {train_path}")
    log(f"  test:  {test_path}")
    log(f"  orig:  {orig_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_path)

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        log("SMOKE=1 — subsampling 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:10_000]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)}  cats={len(cats)}  "
        f"train={len(train)}  test={len(test)}  orig={len(orig)}")

    log("adding threshold flags + LR-formula logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)

    log("adding cat x cat pair combos")
    combos = add_cat_pair_combos(train, test, orig, cats)
    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)
    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)
    log("adding ORIG mean/std per col")
    orig_stats = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: "
        f"cats={len(cats)} combos={len(combos)} digits={len(digits)} "
        f"num_as_cat={len(num_as_cat)} tres={len(tres)} logits={len(logits)} "
        f"freq={len(freq)} orig_stats={len(orig_stats)} "
        f"te_cols={len(info['te_cols'])}")
    return train, test, info, test_ids


# ========================= MLP =========================
class RecipeMLP(nn.Module):
    """4-layer MLP with BatchNorm + GELU + dropout. ~2M params for 443 inputs."""
    def __init__(self, n_in: int, n_classes: int = 3, dropout: float = 0.15):
        super().__init__()
        h = [n_in, 1024, 512, 256, 128]
        layers = []
        for a, b in zip(h[:-1], h[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(h[-1], n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, sw_tr, X_va, y_va, X_te, n_epochs, batch_size, smoke):
    """Single-fold MLP fit with class-balanced sample weights + AdamW + cosine."""
    n_in = X_tr.shape[1]
    model = RecipeMLP(n_in).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    tx = torch.from_numpy(X_tr).float()
    ty = torch.from_numpy(y_tr).long()
    tw = torch.from_numpy(sw_tr).float()
    ds = TensorDataset(tx, ty, tw)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=DEVICE.type == "cuda")

    Xva_t = torch.from_numpy(X_va).float().to(DEVICE)
    yva_t = torch.from_numpy(y_va).long().to(DEVICE)
    Xte_t = torch.from_numpy(X_te).float().to(DEVICE)

    best_va = 0.0
    best_state = None
    for ep in range(1, n_epochs + 1):
        model.train()
        for xb, yb, wb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = (F.cross_entropy(logits, yb, reduction="none") * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        # Validation pass
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xva_t), 8192):
                chunks.append(F.softmax(model(Xva_t[i:i+8192]), dim=-1).cpu().numpy())
            p_va = np.concatenate(chunks, axis=0)
        ba = balanced_accuracy_score(yva_t.cpu().numpy(), p_va.argmax(1))
        log(f"    ep {ep:02d}/{n_epochs}  loss={loss.item():.4f}  va_bal={ba:.5f}")
        if ba > best_va:
            best_va = ba
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, len(Xva_t), 8192):
            chunks.append(F.softmax(model(Xva_t[i:i+8192]), dim=-1).cpu().numpy())
        p_va = np.concatenate(chunks, axis=0)
        chunks = []
        for i in range(0, len(Xte_t), 8192):
            chunks.append(F.softmax(model(Xte_t[i:i+8192]), dim=-1).cpu().numpy())
        p_te = np.concatenate(chunks, axis=0)
    return p_va.astype(np.float32), p_te.astype(np.float32), float(best_va)


# ========================= CV loop =========================
def run_cv(train, test, info, a_ote=1.0):
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    n_epochs = 3 if SMOKE else 25
    batch = 4096 if not SMOKE else 1024
    log(f"MLP config: n_epochs={n_epochs}  batch={batch}  PROBE={PROBE}")

    t_start = time.time()
    TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 12 * 60
    folds_completed = 0
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if PROBE and fold > 1:
            log(f"PROBE mode: stopping after fold 1 (fold {fold} skipped)")
            break
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a_ote)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()

        # Sanitize: NaN → 0, ±inf → 0, all to float32.
        Xtr = X_tr[feat_cols].to_numpy().astype(np.float32)
        Xva = X_va[feat_cols].to_numpy().astype(np.float32)
        Xte = X_te[feat_cols].to_numpy().astype(np.float32)
        for arr in (Xtr, Xva, Xte):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # StandardScaler fit on train only.
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr).astype(np.float32)
        Xva = sc.transform(Xva).astype(np.float32)
        Xte = sc.transform(Xte).astype(np.float32)
        sw_tr = compute_sample_weight("balanced", y[tr_idx]).astype(np.float32)

        log(f"  training MLP on {Xtr.shape[1]} features, {len(Xtr):,} rows, device={DEVICE}")
        t_fit = time.time()
        p_va, p_te, best_va = train_mlp(Xtr, y[tr_idx], sw_tr, Xva, y[va_idx], Xte,
                                         n_epochs=n_epochs, batch_size=batch, smoke=SMOKE)
        oof[va_idx] = p_va
        test_pred += p_te / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(bal)
        elapsed = time.time() - t_start
        log(f"  fold {fold}  argmax={bal:.5f}  best_va={best_va:.5f}  "
            f"fit_wall={time.time()-t_fit:.1f}s  total={elapsed/60:.1f}m")

        # Save partial outputs after each fold for rehydrate resilience.
        suffix = "_smoke" if SMOKE else ("_probe" if PROBE else "")
        np.save(OUT_DIR / f"oof_path_a_recipe_mlp{suffix}.npy", oof)
        np.save(OUT_DIR / f"test_path_a_recipe_mlp{suffix}.npy", test_pred)
        folds_completed = fold
        if elapsed > TOTAL_KILL_SEC:
            log(f"!! TOTAL WALL-TIME KILL {elapsed/60:.1f}m > {TOTAL_KILL_SEC/60:.0f}m")
            break

    if folds_completed and folds_completed < N_FOLDS and not PROBE:
        test_pred *= N_FOLDS / folds_completed
    overall = (balanced_accuracy_score(y[oof.sum(1) > 0], oof[oof.sum(1) > 0].argmax(1))
               if (oof.sum(1) > 0).any() else 0.0)
    log(f"=== overall OOF argmax (filled rows) = {overall:.5f}  folds={folds_completed}")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols,
                folds_completed=folds_completed)


def main():
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    suffix = "_smoke" if SMOKE else ("_probe" if PROBE else "")
    np.save(OUT_DIR / f"oof_path_a_recipe_mlp{suffix}.npy", result["oof"])
    np.save(OUT_DIR / f"test_path_a_recipe_mlp{suffix}.npy", result["test"])

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, probe=PROBE, smoke=SMOKE,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        n_features=len(result["feat_cols"]),
        folds_completed=result["folds_completed"],
        n_epochs=(3 if SMOKE else 25),
        device=str(DEVICE),
    )
    # Tune log-bias only when at least one full fold is done.
    filled = result["oof"].sum(1) > 0
    if filled.sum() > 0:
        prior = np.bincount(y, minlength=3) / len(y)
        bias, tuned = tune_log_bias(result["oof"][filled], y[filled], prior)
        summary["tuned_log_bias_bal_acc"] = float(tuned)
        summary["log_bias"] = bias.tolist()
        log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    with open(OUT_DIR / f"path_a_recipe_mlp{suffix}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote results json")


if __name__ == "__main__":
    main()
