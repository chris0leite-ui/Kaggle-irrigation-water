"""L2 — load + engineer the recipe FE matrix on Kaggle."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_and_engineer():
    log("loading train / test / orig")
    train = pd.read_csv(_find_one("train.csv"))
    test = pd.read_csv(_find_one("test.csv"))
    # 10k original — uploaded as private dataset.
    orig_path = None
    for cand in (
        "Irrigation_Prediction*.csv",
        "irrigation_prediction*.csv",
        "archive*.zip",
    ):
        try:
            orig_path = _find_one(cand)
            break
        except FileNotFoundError:
            continue
    if orig_path is None:
        raise FileNotFoundError("no original dataset under /kaggle/input")
    if orig_path.suffix == ".zip":
        orig = pd.read_csv(orig_path)
    else:
        orig = pd.read_csv(orig_path)

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)

    if IS_SMOKE:
        log("IS_SMOKE — subsampling")
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

    # Factorize raw cats once train+test+orig are aligned (after combos+freq).
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]; test[c] = codes[s:t]; orig[c] = codes[t:]

    tres = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]
    logits_cols = ["logit_P_Low", "logit_P_Medium", "logit_P_High"]
    info = dict(nums=nums, tres=tres, logits=logits_cols, freq=freq,
                orig_stats=orig_stats,
                te_cols=cats + combos + digits + num_as_cat + tres)
    return train, test, info, test_ids


def build_feat_matrix(train, test, info):
    """OTE on full train (representation only, downstream NCM is OOF-validated)."""
    log("fitting OrderedTE on full train")
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
    return (train_fe[feat_cols].to_numpy(np.float32),
            test_fe[feat_cols].to_numpy(np.float32),
            feat_cols)
