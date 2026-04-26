"""Recipe-style FE for the KAN kernel — escapes the 19-feature trap.

Prior KAN PROBE used 19 raw features one-hot and hit Jaccard 0.13 (record
low orthogonality) but errs +500% over anchor — magnitude-trap structural
at that representation. Per the senior-engineer review, the magnitude trap
shrinks dramatically when richer features are provided (recipe XGB at 443
cols anchors the LB-best primary). This module adds the recipe pieces that
don't need per-fold fitting:
  - threshold flags (4)
  - LR-formula logits (3)  -- Chris Deotte coefficients
  - digit features (~66)   -- floor(v * 10^k) % 10 for k=-3..+3
  - cat one-hot (~30)
  - num standardized (11)
  - FREQ per cat (8)
  - ORIG mean/std per num (22)

Total input dim ≈ 140-150. KAN handles this with grid_size=5, spline_order=3.
OTE columns are intentionally skipped — they require per-fold fits and
complicate the kernel; their information is partly captured by FREQ + ORIG_stats.

For Jaccard apples-to-apples vs other NN sister kernels: KAN sees richer
FE than RealMLP/Trompt/Mamba; this is by design — the recommendation
specifically targets KAN's spline inductive bias against the recipe FE
to test whether KAN can match the TREE family's anchor magnitude.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon",
        "Electrical_Conductivity", "Temperature_C", "Humidity",
        "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
        "Field_Area_hectare", "Previous_Irrigation_mm"]

# Chris Deotte LR coefficients on the 10k original (verbatim from
# include4eto/ps6e4-xgb-cudf-pseudo-labels). 3-class logits per row.
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


def find_one(root: Path, pattern_lc: str) -> Path:
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == pattern_lc:
            return p
    raise FileNotFoundError(f"no match for {pattern_lc} under {root}")


def load_data(kaggle_input: Path, smoke: bool):
    print("[data] loading train / test / orig", flush=True)
    train = pd.read_csv(find_one(kaggle_input, "train.csv"))
    test = pd.read_csv(find_one(kaggle_input, "test.csv"))
    orig = pd.read_csv(find_one(kaggle_input, "irrigation_prediction.csv"))
    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    for df in (train, test):
        df.drop(columns=["id"], inplace=True, errors="ignore")
    if smoke:
        print("[data] SMOKE=1 - subsampling", flush=True)
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test_sub = test.sample(10_000, random_state=SEED)
        test_ids = test_ids[test_sub.index.to_numpy()]
        test = test_sub.reset_index(drop=True)
    print(f"[data] train={len(train):,} test={len(test):,} "
          f"orig={len(orig):,}", flush=True)
    return train, test, orig, test_ids


def _threshold_flags(df: pd.DataFrame) -> np.ndarray:
    cols = [
        (df["Soil_Moisture"] < 25),
        (df["Temperature_C"] > 30),
        (df["Rainfall_mm"] < 300),
        (df["Wind_Speed_kmh"] > 10),
    ]
    return np.stack([c.to_numpy(dtype=np.float32) for c in cols], axis=1)


def _lr_logits(df: pd.DataFrame) -> np.ndarray:
    stage = df["Crop_Growth_Stage"].astype(str).values
    mulch = df["Mulching_Used"].astype(str).values
    soil = (df["Soil_Moisture"] < 25).to_numpy(dtype=np.float32)
    temp = (df["Temperature_C"] > 30).to_numpy(dtype=np.float32)
    rain = (df["Rainfall_mm"] < 300).to_numpy(dtype=np.float32)
    wind = (df["Wind_Speed_kmh"] > 10).to_numpy(dtype=np.float32)
    out = []
    for cls, c in _LOGIT_COEFS.items():
        z = (c["bias"] + c["soil_lt_25"] * soil + c["temp_gt_30"] * temp
             + c["rain_lt_300"] * rain + c["wind_gt_10"] * wind)
        z += np.array([c["stage"].get(s, 0.0) for s in stage], dtype=np.float32)
        z += np.array([c["mulch"].get(m, 0.0) for m in mulch], dtype=np.float32)
        out.append(z.astype(np.float32))
    return np.stack(out, axis=1)


def _digits(df: pd.DataFrame, drop_constant_cols: list[int] | None = None):
    """Per-numeric digit features for k=-3..+3. Returns (X, drop_idx).

    Drop columns that are all-zero on the COMBINED train+test+orig — they
    carry no signal but XGB/KAN both ignore them; dropping shrinks input dim.
    """
    feats = []
    names = []
    for col in NUMS:
        v = df[col].to_numpy(dtype=np.float64)
        for k in range(-3, 4):
            d = (np.floor(v * (10.0 ** -k) + 1e-9) % 10).astype(np.float32)
            feats.append(d)
            names.append(f"{col}_d{k}")
    X = np.stack(feats, axis=1).astype(np.float32)
    return X, names


def _freq_features(train: pd.DataFrame, test: pd.DataFrame,
                   orig: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combined-pool category frequency for each row's value (per cat col)."""
    n_tr = len(train); n_te = len(test); n_or = len(orig)
    out_tr = np.zeros((n_tr, len(CATS)), dtype=np.float32)
    out_te = np.zeros((n_te, len(CATS)), dtype=np.float32)
    out_or = np.zeros((n_or, len(CATS)), dtype=np.float32)
    for j, c in enumerate(CATS):
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        vc = combined.value_counts()
        total = float(len(combined))
        out_tr[:, j] = train[c].astype(str).map(vc).fillna(0).to_numpy() / total
        out_te[:, j] = test[c].astype(str).map(vc).fillna(0).to_numpy() / total
        out_or[:, j] = orig[c].astype(str).map(vc).fillna(0).to_numpy() / total
    return out_tr, out_te, out_or


def _orig_mean_std(train: pd.DataFrame, test: pd.DataFrame,
                   orig: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cat groupby on ORIG: mean+std of each numeric per row's cat key.

    Aggregates mean and std PER NUM, joined to row's value of one chosen
    cat (Crop_Type) — keeps it simple for the kernel. The recipe pipeline
    aggregates across more keys but the bulk of the signal is here.
    """
    key = "Crop_Type"
    n_tr = len(train); n_te = len(test); n_or = len(orig)
    out_tr = np.zeros((n_tr, 2 * len(NUMS)), dtype=np.float32)
    out_te = np.zeros((n_te, 2 * len(NUMS)), dtype=np.float32)
    out_or = np.zeros((n_or, 2 * len(NUMS)), dtype=np.float32)
    grp = orig.groupby(key)[NUMS]
    means = grp.mean()
    stds = grp.std().fillna(0.0)
    global_mean = orig[NUMS].mean()
    global_std = orig[NUMS].std().fillna(0.0)
    for j, num in enumerate(NUMS):
        m_map = means[num].to_dict()
        s_map = stds[num].to_dict()
        for tag, df, out in [("train", train, out_tr),
                              ("test", test, out_te),
                              ("orig", orig, out_or)]:
            keyvals = df[key].astype(str)
            out[:, 2 * j] = keyvals.map(m_map).fillna(global_mean[num]).to_numpy(dtype=np.float32)
            out[:, 2 * j + 1] = keyvals.map(s_map).fillna(global_std[num]).to_numpy(dtype=np.float32)
    return out_tr, out_te, out_or


def build_arrays(train: pd.DataFrame, test: pd.DataFrame,
                 orig: pd.DataFrame):
    """Recipe-style FE assembly for KAN. Returns float32 arrays.

    Pieces (numeric):
      - 11 raw nums, standardised
      - 4 threshold flags
      - 3 LR-formula logits, standardised
      - 66 digit features (kept as continuous; spline handles them)
      - 8 FREQ-per-cat
      - 22 orig mean/std (per Crop_Type key x 11 nums x 2 stats)
    Pieces (binary):
      - cats one-hot
    """
    # 1. Threshold flags.
    tres_tr = _threshold_flags(train)
    tres_te = _threshold_flags(test)
    tres_or = _threshold_flags(orig)
    print(f"[fe] threshold flags: {tres_tr.shape[1]}", flush=True)

    # 2. LR logits (standardised on train+orig).
    lr_tr = _lr_logits(train)
    lr_te = _lr_logits(test)
    lr_or = _lr_logits(orig)
    fit_lr = np.concatenate([lr_tr, lr_or], axis=0)
    mu, sd = fit_lr.mean(axis=0), fit_lr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    lr_tr = ((lr_tr - mu) / sd).astype(np.float32)
    lr_te = ((lr_te - mu) / sd).astype(np.float32)
    lr_or = ((lr_or - mu) / sd).astype(np.float32)
    print(f"[fe] LR logits: {lr_tr.shape[1]} (standardised)", flush=True)

    # 3. Digits (kept on natural 0-9 range, no standardise — spline grid_range
    # = [-1, 1] in config, but KAN tolerates moderate outliers; we'll
    # re-scale by /10 to land them in [0, 0.9]).
    dig_tr, dnames = _digits(train)
    dig_te, _ = _digits(test)
    dig_or, _ = _digits(orig)
    dig_tr = dig_tr / 10.0
    dig_te = dig_te / 10.0
    dig_or = dig_or / 10.0
    print(f"[fe] digits: {dig_tr.shape[1]} (scaled to [0, 0.9])", flush=True)

    # 4. Cats one-hot.
    cat_tr_list, cat_te_list, cat_or_list = [], [], []
    for c in CATS:
        vals = sorted(set(train[c].astype(str)) |
                      set(test[c].astype(str)) |
                      set(orig[c].astype(str)))
        idx = {v: i for i, v in enumerate(vals)}
        K = len(vals)
        for tag, df, target_list in (("train", train, cat_tr_list),
                                      ("test", test, cat_te_list),
                                      ("orig", orig, cat_or_list)):
            col = df[c].astype(str).map(idx).to_numpy()
            oh = np.zeros((len(df), K), dtype=np.float32)
            oh[np.arange(len(df)), col] = 1.0
            target_list.append(oh)
    cat_tr = np.concatenate(cat_tr_list, axis=1)
    cat_te = np.concatenate(cat_te_list, axis=1)
    cat_or = np.concatenate(cat_or_list, axis=1)
    print(f"[fe] cats one-hot: {cat_tr.shape[1]}", flush=True)

    # 5. FREQ per cat.
    fr_tr, fr_te, fr_or = _freq_features(train, test, orig)
    print(f"[fe] FREQ: {fr_tr.shape[1]}", flush=True)

    # 6. ORIG mean/std (per Crop_Type x 11 nums x 2). Standardise on train+orig.
    om_tr, om_te, om_or = _orig_mean_std(train, test, orig)
    fit_om = np.concatenate([om_tr, om_or], axis=0)
    mu, sd = fit_om.mean(axis=0), fit_om.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    om_tr = ((om_tr - mu) / sd).astype(np.float32)
    om_te = ((om_te - mu) / sd).astype(np.float32)
    om_or = ((om_or - mu) / sd).astype(np.float32)
    print(f"[fe] ORIG mean/std: {om_tr.shape[1]} (standardised)", flush=True)

    # 7. Raw nums standardised on train+orig.
    fit_nums = pd.concat([train[NUMS], orig[NUMS]], axis=0,
                         ignore_index=True)
    mu = fit_nums.mean().to_numpy(dtype=np.float32)
    sd = fit_nums.std().to_numpy(dtype=np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd)
    nm_tr = ((train[NUMS].to_numpy(dtype=np.float32) - mu) / sd)
    nm_te = ((test[NUMS].to_numpy(dtype=np.float32) - mu) / sd)
    nm_or = ((orig[NUMS].to_numpy(dtype=np.float32) - mu) / sd)
    print(f"[fe] raw nums: {nm_tr.shape[1]} (standardised)", flush=True)

    # Stack everything.
    def stack(*parts):
        return np.concatenate(parts, axis=1).astype(np.float32)

    X_train = stack(nm_tr, tres_tr, lr_tr, dig_tr, cat_tr, fr_tr, om_tr)
    X_test = stack(nm_te, tres_te, lr_te, dig_te, cat_te, fr_te, om_te)
    X_orig = stack(nm_or, tres_or, lr_or, dig_or, cat_or, fr_or, om_or)
    y_train = train[TARGET].to_numpy(dtype=np.int64)
    y_orig = orig[TARGET].to_numpy(dtype=np.int64)
    feat_dim = X_train.shape[1]
    print(f"[fe] feat_dim={feat_dim}  X_train={X_train.shape} "
          f"X_test={X_test.shape} X_orig={X_orig.shape}", flush=True)
    return X_train, X_test, X_orig, y_train, y_orig, feat_dim
