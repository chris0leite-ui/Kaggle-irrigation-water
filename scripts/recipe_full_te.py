"""Full public-notebook recipe: all FE + OrderedTE + heavy-reg XGB.

Combines:
  - cat x cat pair combos         (28 cols)
  - digit features -4..+3         (~70 cols after dropping test-constants)
  - num-as-cat                    (11 cols)
  - threshold flags               (4 cols)
  - LR-formula logits             (3 numeric cols)
  - FREQ per cat                  (~44 cols: all cats + combos)
  - ORIG mean/std per col         (~48 numeric cols)
  - OrderedTE (a=1) on every categorical feature (3 cls each)

XGB: max_depth=4, alpha=5, reg_lambda=5, max_leaves=30, lr=0.1,
max_bin=10000, n_estimators=50000, early_stopping_rounds=500,
eval_metric=balanced_accuracy. Sample-weight = class-balanced.

5-fold StratifiedKFold (seed=42) aligned with every other OOF on disk.
Outputs:
  scripts/artifacts/oof_recipe_full_te.npy
  scripts/artifacts/test_recipe_full_te.npy
  scripts/artifacts/recipe_full_te_results.json
  submissions/submission_recipe_full_te.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_features import (  # noqa: E402
    add_cat_pair_combos, add_decimal_fractions, add_digit_features,
    add_domain_interactions, add_freq_features, add_groupby_cat_num_stats,
    add_lr_formula_logits, add_num_as_cat, add_orig_mean_std,
    add_threshold_flags, add_w8_block,
)
from recipe_ote import OrderedTE  # noqa: E402

import os

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

# SMOKE=1 → shrink to 20k train rows, 1 fold, fewer XGB iters. For bug-hunting.
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

# Variant knobs (default preserves the LB-best pipeline):
#   OTE_ALPHA    — shrinkage toward prior in OrderedTE (default 1.0)
#   XGB_BOOSTER  — "gbtree" (default) or "dart" (tree dropout)
#   FOLD_SEED    — seed for StratifiedKFold split (default 42). Override to
#                  7 / 123 / etc. for multi-seed pseudo-label experiments
#                  where the labeler's fold structure is decoupled from
#                  the downstream target model's fold split.
# When any differs from default, output paths get a suffix like
# "_a01", "_a10_dart", "_seed7", "_seed7_dart" so the LB-best artefacts
# stay untouched.
OTE_ALPHA = float(os.environ.get("OTE_ALPHA", "1.0"))
XGB_BOOSTER = os.environ.get("XGB_BOOSTER", "gbtree")
FOLD_SEED = int(os.environ.get("FOLD_SEED", str(SEED)))
# DAE_EMBED_PATH: optional path to a 128-d SwapNoise-DAE train embedding
# (.npy, shape (n_train, K)). The matching test array is derived by
# replacing "oof_" → "test_". When set, the K columns are added as extra
# numerics. Integration for A2 / P1.
DAE_EMBED_PATH = os.environ.get("DAE_EMBED_PATH", "")
# 10k-anchor features (built by build_10k_anchor_features.py).
# EXTRA_OOD=1   -> +3 cols [GMM_neg_logp, IsoForest, kNN_dist] from
#                 oof_ood3_train.npy + test_ood3.npy
# EXTRA_KNN10K=1-> +8 cols [p_low,p_med,p_high,nbr0_y,d_low,d_med,d_high,margin]
#                 from oof_knn10k_train.npy + test_knn10k.npy
EXTRA_OOD = os.environ.get("EXTRA_OOD", "") == "1"
EXTRA_KNN10K = os.environ.get("EXTRA_KNN10K", "") == "1"
# EXTRA_OOD9=1 -> +9 cols (per-class GMM density + per-class kNN dist +
# per-class Mahalanobis to centroid) from oof_ood9_train.npy + test_ood9.npy
# Built by build_expanded_10k_anchor.py.
EXTRA_OOD9 = os.environ.get("EXTRA_OOD9", "") == "1"
# EXTRA_FE: A4 FE transplant from public kernels.
#   ""       — baseline recipe (no extra FE)
#   "domain" — +11 utaazu-style ratio/product features (moist_rain, ET_proxy...)
#   "decimal"— +5 decimal-fraction features `(col % 1).round(2)` on numerics
#   "both"   — both sets (16 new numeric features total)
# Suffix "_fex{variant}" on outputs so LB-best artefacts stay untouched.
EXTRA_FE = os.environ.get("EXTRA_FE", "")
assert EXTRA_FE in ("", "domain", "decimal", "both", "w8"), (
    f"EXTRA_FE must be ''|domain|decimal|both|w8, got {EXTRA_FE!r}"
)
# GBY: rohit8527-style group-by cat x num stats on the SYNTHETIC 630k pool.
# When set to "1", adds per-cat-group mean/std of each numeric, merged onto
# train+test. Distinct from add_orig_mean_std which aggregates TARGET on
# 10k original. 8 cats x 11 nums x 2 stats = 176 extra numeric features.
# Suffix "_gby" on outputs.
GBY = os.environ.get("GBY", "") == "1"

# INSTAB: P3 counterfactual rule-instability features. When set to "1", adds
# 5 features (rule_instability total + per-axis flips for sm/rf/tc/ws) that
# count cell-flips under {±2%, ±5%, ±10%, ±20%} perturbation of each rule
# axis. Captures multi-axis simultaneous closeness to the rule-cell topology
# in a way per-axis distance features cannot. Suffix "_instab" on outputs.
INSTAB = os.environ.get("INSTAB", "") == "1"
# Cleanlab intervention: modify training rows flagged as label-noise.
#   ""           — baseline (no intervention)
#   "drop"       — drop flagged rows from each fold's train set
#   "downweight" — multiply flagged rows' sample_weight by CLEANLAB_WEIGHT
#   "relabel"    — replace flagged rows' y with teacher argmax
# Expected mask path: scripts/artifacts/cleanlab_issues_prune_by_noise_rate.npy
CLEANLAB_TREATMENT = os.environ.get("CLEANLAB_TREATMENT", "")
CLEANLAB_WEIGHT = float(os.environ.get("CLEANLAB_WEIGHT", "0.3"))
# DROP_SCORES: comma-separated dgp_score values to remove from XGB training
# AND override at inference with rule=Low. Mirrors the 2026-04-21 v3/v4
# routed-XGB lever onto the recipe pipeline. e.g. "0,1,2" or "1,2".
# Empty string = no drop (default; preserves LB-best pipeline).
# Mech B: anchor-uncertainty-weighted retraining. When set, multiply
# class-balanced sample_weight by (1 + ALPHA × (1 − max_prob_LB4stack[i]))
# per train row. Up-weights rows where the LB-best 4-stack is uncertain,
# focusing recipe XGB capacity on boundary regions. Different from
# CLEANLAB_TREATMENT=downweight which DOWN-weights ambiguous rows
# interpreted as label noise. Suffix "_anchwα{α}" on outputs.
ANCHOR_WEIGHT_ALPHA = float(os.environ.get("ANCHOR_WEIGHT_ALPHA", "0"))

# Mech A: boundary-confined test-time augmentation. When set, identify
# boundary rows via max_prob(LB-best 4-stack) < TTA_BOUNDARY_THRESH (default
# 0.95) and replace their val/test predictions with the K-perturbation
# average. Perturbations are σ × IQR Gaussian on the 4 rule axes
# (Soil_Moisture, Rainfall_mm, Temperature_C, Wind_Speed_kmh); axis-derived
# FE (sm/rf/tc/ws_dist + abs, dry/norain/hot/windy, dgp_score, LR-formula
# logits) is recomputed per perturbation. Digits/num_as_cat/OTE keys are
# NOT recomputed (perturbations are small enough that key changes are rare).
# Suffix "_btta" on outputs. Mutually exclusive with anchor weight + cleanlab.
TTA_BOUNDARY = os.environ.get("TTA_BOUNDARY", "") == "1"
TTA_BOUNDARY_THRESH = float(os.environ.get("TTA_BOUNDARY_THRESH", "0.95"))
TTA_K = int(os.environ.get("TTA_K", "10"))
TTA_SIGMA = float(os.environ.get("TTA_SIGMA", "0.05"))

DROP_SCORES = os.environ.get("DROP_SCORES", "")
DROP_SCORE_SET: set[int] = set()
if DROP_SCORES:
    DROP_SCORE_SET = {int(s) for s in DROP_SCORES.split(",") if s.strip() != ""}
    assert all(0 <= s <= 9 for s in DROP_SCORE_SET), \
        f"DROP_SCORES must be in 0..9, got {DROP_SCORE_SET}"
    assert not CLEANLAB_TREATMENT, "DROP_SCORES is mutually exclusive with CLEANLAB_TREATMENT"
assert XGB_BOOSTER in ("gbtree", "dart"), f"XGB_BOOSTER must be gbtree|dart, got {XGB_BOOSTER}"
assert CLEANLAB_TREATMENT in ("", "drop", "downweight", "relabel"), CLEANLAB_TREATMENT

_parts = []
if OTE_ALPHA != 1.0:
    _parts.append("a" + f"{OTE_ALPHA:g}".replace(".", ""))
if XGB_BOOSTER != "gbtree":
    _parts.append(XGB_BOOSTER)
if FOLD_SEED != SEED:
    _parts.append(f"seed{FOLD_SEED}")
if CLEANLAB_TREATMENT:
    _parts.append(f"cl{CLEANLAB_TREATMENT}")
if DAE_EMBED_PATH:
    _parts.append("dae")
if EXTRA_OOD:
    _parts.append("ood")
if EXTRA_KNN10K:
    _parts.append("knn10k")
if EXTRA_OOD9:
    _parts.append("ood9")
if EXTRA_FE:
    _parts.append(f"fex{EXTRA_FE}")
if GBY:
    _parts.append("gby")
if INSTAB:
    _parts.append("instab")
if DROP_SCORE_SET:
    _parts.append("ds" + "".join(str(s) for s in sorted(DROP_SCORE_SET)))
if ANCHOR_WEIGHT_ALPHA != 0:
    _parts.append(f"anchw{('m' if ANCHOR_WEIGHT_ALPHA < 0 else '') + str(int(abs(ANCHOR_WEIGHT_ALPHA)*10)).zfill(2)}")
if TTA_BOUNDARY:
    _parts.append(f"btta{int(TTA_BOUNDARY_THRESH*100):03d}k{TTA_K}s{int(TTA_SIGMA*100):03d}")
VARIANT_SUFFIX = ("_" + "_".join(_parts)) if _parts else ""

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------- data + features
def load_and_engineer() -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    log("loading train / test / orig")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    # Track original positions so DAE embeddings (produced on full 630k/270k)
    # can be subsampled to match the SMOKE subset.
    train_subset_idx = None
    test_subset_idx = None
    if SMOKE:
        log("SMOKE=1 — subsampling to 20k train, 10k test")
        train_s = train.sample(20_000, random_state=SEED)
        train_subset_idx = train_s.index.to_numpy()
        train = train_s.reset_index(drop=True)
        test_s = test.sample(10_000, random_state=SEED)
        test_subset_idx = test_s.index.to_numpy()
        test = test_s.reset_index(drop=True)
        test_ids = test_ids[test_subset_idx]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)}  cats={len(cats)}  "
        f"train={len(train)}  test={len(test)}  orig={len(orig)}")

    # Threshold flags + LR-formula logits computed from raw numerics/strings
    # BEFORE any factorization. The LR formula needs stage/mulch as strings.
    log("adding threshold flags + LR-formula logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)

    # Compute dgp_score on train+test (orig only needed if downstream FE wanted
    # it). Stage/mulch are still strings here. Score formula matches the rule
    # in CLAUDE.md: 2*(dry+norain) + (hot+windy+nomulch) + Kc, where Kc = 2 iff
    # Crop_Growth_Stage in {Flowering, Vegetative}. Stored as int8 column.
    if DROP_SCORE_SET:
        log(f"computing dgp_score for DROP_SCORES={sorted(DROP_SCORE_SET)} routing")
        for df in (train, test):
            nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
            stage_str = df["Crop_Growth_Stage"].astype(str).values
            kc = np.where(np.isin(stage_str, ("Flowering", "Vegetative")), 2, 0).astype(np.int8)
            score = (2 * (df["soil_lt_25"].values + df["rain_lt_300"].values)
                     + df["temp_gt_30"].values + df["wind_gt_10"].values + nomulch
                     + kc).astype(np.int8)
            df["dgp_score"] = score
        # Sanity-log distribution on train.
        vc = pd.Series(train["dgp_score"]).value_counts().sort_index()
        log(f"  train dgp_score dist: {dict(vc)}")

    # A4 FE transplant: extra numeric features from public kernels.
    # Added BEFORE digit extraction so they're included as "extra nums" for
    # the tree but NOT digit-expanded (would explode feature count).
    extra_domain: list[str] = []
    extra_decimal: list[str] = []
    extra_w8: list[str] = []
    if EXTRA_FE in ("domain", "both"):
        log("A4 FE transplant: +11 domain interaction features (utaazu)")
        for df in (train, test, orig):
            extra_domain = add_domain_interactions(df)
    if EXTRA_FE in ("decimal", "both"):
        log("A4 FE transplant: +5 decimal-fraction features")
        for df in (train, test, orig):
            extra_decimal = add_decimal_fractions(df)
    if EXTRA_FE == "w8":
        log("W8 FE block: +15 novel-on-recipe cross-products + per-score z-scores")
        for df in (train, test, orig):
            extra_w8 = add_w8_block(df)

    # Pair combos (concat string values; factorized across combined).
    log("adding cat x cat pair combos")
    combos = add_cat_pair_combos(train, test, orig, cats)

    # Digit features on raw numerics.
    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)

    # Num-as-cat (factorized across combined).
    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)

    # FREQ per raw cat + per combo (computed on train+test+orig combined).
    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)

    # ORIG mean/std per column (leak-free — external source only).
    log("adding ORIG mean/std per col")
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    # rohit8527 group-by cat x num stats on synthetic 630k pool.
    # Distinct from orig_stats (which aggregates TARGET on 10k orig).
    gby_cols: list[str] = []
    if GBY:
        log("adding group-by cat x num stats (mean/std) on synthetic 630k")
        gby_cols = add_groupby_cat_num_stats(train, test, cats, nums)
        log(f"  gby_cols={len(gby_cols)}")

    # P3 counterfactual rule-instability (~5 cols). Pure function of raw
    # numerics + Mulching_Used + Crop_Growth_Stage; safe to compute pre-fold
    # and pre-factorize. Must run BEFORE the cat-factorize block below since
    # it reads Mulching_Used / Crop_Growth_Stage as strings.
    instab_cols: list[str] = []
    if INSTAB:
        from p3_instability import add_instability
        log("adding P3 rule-instability features")
        train = add_instability(train)
        test = add_instability(test)
        instab_cols = ["rule_inst_sm", "rule_inst_rf", "rule_inst_tc",
                       "rule_inst_ws", "rule_instability"]
        log(f"  instab_cols={len(instab_cols)}")

    # Factorize raw cats AFTER all FE that needs string values is done.
    # (OrderedTE groupby handles int keys just fine.)
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    # Optional DAE embeddings (A2 / P1). Load after all categorical-string
    # FE is done so raw-dtype expectations are stable.
    dae_cols: list[str] = []
    if DAE_EMBED_PATH:
        log(f"loading DAE embeddings from {DAE_EMBED_PATH}")
        dae_tr = np.load(DAE_EMBED_PATH).astype(np.float32)
        test_path_guess = DAE_EMBED_PATH.replace("oof_", "test_")
        dae_te = np.load(test_path_guess).astype(np.float32)
        log(f"  dae_train={dae_tr.shape}  dae_test={dae_te.shape}")
        if SMOKE:
            assert train_subset_idx is not None and test_subset_idx is not None
            dae_tr = dae_tr[train_subset_idx]
            dae_te = dae_te[test_subset_idx]
            log(f"  SMOKE-subsampled dae: {dae_tr.shape}, {dae_te.shape}")
        assert dae_tr.shape[0] == len(train), (dae_tr.shape, len(train))
        assert dae_te.shape[0] == len(test), (dae_te.shape, len(test))
        n_dae = dae_tr.shape[1]
        dae_cols = [f"dae_{i}" for i in range(n_dae)]
        for i, c in enumerate(dae_cols):
            train[c] = dae_tr[:, i]
            test[c] = dae_te[:, i]

    # 10k-anchor features (built by build_10k_anchor_features.py).
    ood_cols: list[str] = []
    knn10k_cols: list[str] = []
    if EXTRA_OOD:
        log("loading 10k-OOD features (GMM, IsoForest, kNN-density)")
        a_tr = np.load("scripts/artifacts/oof_ood3_train.npy").astype(np.float32)
        a_te = np.load("scripts/artifacts/test_ood3.npy").astype(np.float32)
        if SMOKE:
            assert train_subset_idx is not None and test_subset_idx is not None
            a_tr = a_tr[train_subset_idx]; a_te = a_te[test_subset_idx]
        assert a_tr.shape == (len(train), 3) and a_te.shape == (len(test), 3)
        ood_cols = ["ood_gmm", "ood_iso", "ood_knn"]
        for i, c in enumerate(ood_cols):
            train[c] = a_tr[:, i]; test[c] = a_te[:, i]
        log(f"  +{len(ood_cols)} OOD cols")
    if EXTRA_KNN10K:
        log("loading kNN-from-10k geometric features (k=20)")
        a_tr = np.load("scripts/artifacts/oof_knn10k_train.npy").astype(np.float32)
        a_te = np.load("scripts/artifacts/test_knn10k.npy").astype(np.float32)
        if SMOKE:
            assert train_subset_idx is not None and test_subset_idx is not None
            a_tr = a_tr[train_subset_idx]; a_te = a_te[test_subset_idx]
        assert a_tr.shape == (len(train), 8) and a_te.shape == (len(test), 8)
        knn10k_cols = ["k10_pL","k10_pM","k10_pH","k10_nbr0",
                       "k10_dL","k10_dM","k10_dH","k10_margin"]
        for i, c in enumerate(knn10k_cols):
            train[c] = a_tr[:, i]; test[c] = a_te[:, i]
        log(f"  +{len(knn10k_cols)} kNN10k cols")
    ood9_cols: list[str] = []
    if EXTRA_OOD9:
        log("loading expanded 10k-anchor features (per-class GMM/kNN/Maha)")
        a_tr = np.load("scripts/artifacts/oof_ood9_train.npy").astype(np.float32)
        a_te = np.load("scripts/artifacts/test_ood9.npy").astype(np.float32)
        if SMOKE:
            assert train_subset_idx is not None and test_subset_idx is not None
            a_tr = a_tr[train_subset_idx]; a_te = a_te[test_subset_idx]
        assert a_tr.shape == (len(train), 9) and a_te.shape == (len(test), 9)
        ood9_cols = ["gmm_L","gmm_M","gmm_H","knn_d_L","knn_d_M","knn_d_H",
                     "maha_L","maha_M","maha_H"]
        for i, c in enumerate(ood9_cols):
            train[c] = a_tr[:, i]; test[c] = a_te[:, i]
        log(f"  +{len(ood9_cols)} ood9 cols")

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats_cols, dae_embed=dae_cols,
        extra_domain=extra_domain, extra_decimal=extra_decimal,
        extra_w8=extra_w8,
        gby=gby_cols, instab=instab_cols,
        ood=ood_cols, knn10k=knn10k_cols, ood9=ood9_cols,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: "
        f"cats={len(cats)} combos={len(combos)} digits={len(digits)} "
        f"num_as_cat={len(num_as_cat)} tres={len(tres)} logits={len(logits)} "
        f"freq={len(freq)} orig_stats={len(orig_stats_cols)} "
        f"dae_embed={len(dae_cols)} "
        f"extra_domain={len(extra_domain)} extra_decimal={len(extra_decimal)} "
        f"extra_w8={len(extra_w8)} "
        f"gby={len(gby_cols)} "
        f"te_cols={len(info['te_cols'])}")
    return train, test, info, test_ids


_anchor_uncertainty: np.ndarray | None = None  # populated by run_cv when Mech B / Mech A active
_anchor_uncertainty_test: np.ndarray | None = None  # populated when Mech A active


def _build_anchor_uncertainty_test(n_test: int = -1) -> np.ndarray:
    """Test-side anchor uncertainty (for Mech A boundary-row identification).

    Returns zero-vector under SMOKE since test is subsampled and we can't
    recover the indices here. TTA mask = 0 → boundary mask all False →
    no perturbation applied (test_pred unchanged).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from common import log_blend  # noqa: E402
    from sklearn.isotonic import IsotonicRegression  # noqa: E402

    def _normed(a):
        return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)

    def _iso_cal(oof, test_arr, y_lab):
        oo = np.zeros_like(oof, dtype=np.float32)
        tt = np.zeros_like(test_arr, dtype=np.float32)
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[:, c], (y_lab == c).astype(np.float32))
            oo[:, c] = ir.predict(oof[:, c])
            tt[:, c] = ir.predict(test_arr[:, c])
        return _normed(oo), _normed(tt)

    train_full = pd.read_csv("data/train.csv")
    y_full = train_full[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    r_o = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    r_t = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1_o = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1_t = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7_o = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7_t = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm_o = _normed(np.load(ART / "oof_realmlp.npy"))
    rm_t = _normed(np.load(ART / "test_realmlp.npy"))
    nr_o = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nr_t = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    nr_o_iso, nr_t_iso = _iso_cal(nr_o, nr_t, y_full)
    mv_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    mv_t = _normed(np.load(ART / "test_xgb_metastack.npy"))
    mv_o_iso, mv_t_iso = _iso_cal(mv_o, mv_t, y_full)
    lb3_t = log_blend([r_t, s1_t, s7_t], np.array([0.25, 0.35, 0.40]))
    s2_t = log_blend([lb3_t, rm_t], np.array([0.8, 0.2]))
    s3_t = log_blend([s2_t, nr_t_iso], np.array([0.925, 0.075]))
    lb4_t = log_blend([s3_t, mv_t_iso], np.array([0.7, 0.3]))
    out = (1.0 - lb4_t.max(axis=1)).astype(np.float32)
    if n_test > 0 and out.shape[0] != n_test:
        # SMOKE subsample mismatch — return zero so boundary mask is empty.
        return np.zeros(n_test, dtype=np.float32)
    return out


def _apply_btta(model, feat_cols, base_pred, X_df, mask, fold_seed):
    """Replace predictions at boundary rows with K-perturbation average.

    Perturbations: σ × IQR Gaussian on 4 axis numerics. Recompute the
    directly-derived axis FE (sm/rf/tc/ws_dist + abs + tres flags +
    dgp_score + LR-formula logits) per perturbation. Predict via model,
    average K predictions, replace base_pred[mask] in-place via copy.
    """
    if mask.sum() == 0:
        return base_pred
    out = base_pred.copy()
    sub = X_df.iloc[mask].copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED + fold_seed * 11)

    # IQR per axis from the train OR test slice we have at hand
    iqrs = {}
    for ax in ("Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"):
        v = X_df[ax].astype(float).to_numpy()
        iqrs[ax] = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))

    accum = np.zeros((mask.sum(), 3), dtype=np.float32)
    for k in range(TTA_K):
        pert = sub.copy()
        for ax in iqrs:
            pert[ax] = (sub[ax].astype(float).to_numpy()
                        + rng.normal(0, TTA_SIGMA * iqrs[ax], size=len(sub))).astype(np.float32)
        # Recompute axis-derived FE
        sm = pert["Soil_Moisture"].astype(float).to_numpy()
        rf = pert["Rainfall_mm"].astype(float).to_numpy()
        tc = pert["Temperature_C"].astype(float).to_numpy()
        ws = pert["Wind_Speed_kmh"].astype(float).to_numpy()
        pert["sm_dist"] = (sm - 25.0).astype(np.float32)
        pert["rf_dist"] = (rf - 300.0).astype(np.float32)
        pert["tc_dist"] = (tc - 30.0).astype(np.float32)
        pert["ws_dist"] = (ws - 10.0).astype(np.float32)
        for col in ("sm_dist", "rf_dist", "tc_dist", "ws_dist"):
            pert[col.replace("_dist", "_abs")] = np.abs(pert[col].to_numpy()).astype(np.float32)
        dry = (sm < 25.0).astype(np.int8)
        norain = (rf < 300.0).astype(np.int8)
        hot = (tc > 30.0).astype(np.int8)
        windy = (ws > 10.0).astype(np.int8)
        nomulch = pert["nomulch"].to_numpy() if "nomulch" in pert.columns else np.zeros(len(pert), dtype=np.int8)
        kc_act = pert["kc_active"].to_numpy() if "kc_active" in pert.columns else np.zeros(len(pert), dtype=np.int8)
        pert["dry"] = dry
        pert["norain"] = norain
        pert["hot"] = hot
        pert["windy"] = windy
        pert["dgp_score"] = (2 * (dry + norain) + hot + windy + nomulch + 2 * kc_act).astype(np.int8)

        prd = model.predict_proba(pert[feat_cols]).astype(np.float32)
        accum += prd / TTA_K
    out[mask] = accum
    return out


def _build_anchor_uncertainty(n_train: int) -> np.ndarray:
    """Per-train-row anchor uncertainty = (1 − max P(class)) of LB-best 4-stack.

    Reconstructs LB-best primary on the fly from saved component OOFs, so
    Mech B doesn't require a separate ground-truth file. SMOKE is handled
    via row subsampling in load_and_engineer.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from common import log_blend  # noqa: E402
    from sklearn.isotonic import IsotonicRegression  # noqa: E402

    def _normed(a):
        return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)

    def _iso_cal(oof, test_arr, y_lab):
        oo = np.zeros_like(oof, dtype=np.float32)
        tt = np.zeros_like(test_arr, dtype=np.float32)
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[:, c], (y_lab == c).astype(np.float32))
            oo[:, c] = ir.predict(oof[:, c])
            tt[:, c] = ir.predict(test_arr[:, c])
        return _normed(oo), _normed(tt)

    train_full = pd.read_csv("data/train.csv")
    y_full = train_full[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    r = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    s1 = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s7 = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    rm = _normed(np.load(ART / "oof_realmlp.npy"))
    nr = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nr_iso, _ = _iso_cal(nr, np.zeros_like(nr), y_full)
    mv1 = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    mv1_iso, _ = _iso_cal(mv1, np.zeros_like(mv1), y_full)
    lb3 = log_blend([r, s1, s7], np.array([0.25, 0.35, 0.40]))
    s2 = log_blend([lb3, rm], np.array([0.8, 0.2]))
    s3 = log_blend([s2, nr_iso], np.array([0.925, 0.075]))
    lb4 = log_blend([s3, mv1_iso], np.array([0.7, 0.3]))
    u_full = (1.0 - lb4.max(axis=1)).astype(np.float32)  # uncertainty in [0, 0.667]
    if n_train != len(u_full):
        # SMOKE subset path: caller will pass row indices via global SMOKE
        # subsample applied in load_and_engineer. We can't recover those
        # indices here, so SMOKE+ANCHOR_WEIGHT_ALPHA path is unsupported;
        # return uniform zero so weights collapse to balanced.
        return np.zeros(n_train, dtype=np.float32)
    return u_full


# --------------------------------------------------------- training loop
def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           a_ote: float = 1.0) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    # Mech B / Mech A: precompute anchor-uncertainty weights ONCE
    # (full-train); indexed per-fold inside the loop.
    global _anchor_uncertainty, _anchor_uncertainty_test
    if ANCHOR_WEIGHT_ALPHA != 0 or TTA_BOUNDARY:
        _anchor_uncertainty = _build_anchor_uncertainty(len(train))
        log(f"  Mech B/A: anchor-uncertainty precomputed, "
            f"mean={_anchor_uncertainty.mean():.4f} "
            f"(0 means SMOKE-fallback or zero-input)")
    if TTA_BOUNDARY:
        _anchor_uncertainty_test = _build_anchor_uncertainty_test(len(test))
        log(f"  Mech A: test-side anchor-uncertainty precomputed, "
            f"mean={_anchor_uncertainty_test.mean():.4f}, "
            f"boundary mask = (1-max_p) > {1.0 - TTA_BOUNDARY_THRESH}")

    # Load cleanlab mask + teacher (if relabel) once, up-front.
    cleanlab_mask = None
    teacher_pred = None
    if CLEANLAB_TREATMENT and SMOKE:
        log("SMOKE=1: synthesising a fake cleanlab mask (~0.3% True) so the "
            "plumbing runs, without needing the real 630k-row mask")
        rng = np.random.default_rng(SEED)
        cleanlab_mask = rng.random(len(train)) < 0.003
        if CLEANLAB_TREATMENT == "relabel":
            teacher_pred = rng.integers(0, 3, size=len(train), dtype=np.int64)
    elif CLEANLAB_TREATMENT:
        mask_path = ART / "cleanlab_issues_prune_by_noise_rate.npy"
        cleanlab_mask = np.load(mask_path)
        assert cleanlab_mask.shape == (len(train),), cleanlab_mask.shape
        log(f"cleanlab treatment={CLEANLAB_TREATMENT} "
            f"flagged={int(cleanlab_mask.sum()):,} "
            f"({100*cleanlab_mask.mean():.3f}%)")
        if CLEANLAB_TREATMENT == "relabel":
            eps = 1e-12
            teach_a = np.load(ART / "oof_recipe_full_te.npy")
            teach_b = np.load(ART / "oof_recipe_pseudolabel.npy")
            la = np.log(np.clip(teach_a, eps, 1.0))
            lb = np.log(np.clip(teach_b, eps, 1.0))
            z = 0.5 * la + 0.5 * lb
            z = z - z.max(axis=1, keepdims=True)
            ez = np.exp(z)
            teacher = ez / ez.sum(axis=1, keepdims=True)
            teacher_pred = teacher.argmax(axis=1).astype(np.int64)
            log(f"  relabel teacher loaded; {(teacher_pred != y).sum():,} "
                f"rows where teacher disagrees with observed")

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"]
                     + info.get("dae_embed", [])
                     + info.get("extra_domain", [])
                     + info.get("extra_decimal", [])
                     + info.get("extra_w8", [])
                     + info.get("gby", [])
                     + info.get("instab", [])
                     + info.get("ood", [])
                     + info.get("knn10k", [])
                     + info.get("ood9", []))
    drop_after_te = info["te_cols"]  # raw cats dropped; only TE cols retained

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    # Per-fold checkpoints (rehydrate-resilient): saves each fold's val
    # OOF + test pred immediately on completion, restores on rerun.
    ck_prefix = f"recipe_full_te{VARIANT_SUFFIX}"
    cached = set()
    for fold_check in range(1, N_FOLDS + 1):
        ck_oof = ART / f"oof_{ck_prefix}_fold{fold_check}.npy"
        ck_test = ART / f"test_{ck_prefix}_fold{fold_check}.npy"
        if ck_oof.exists() and ck_test.exists():
            cached.add(fold_check)
    if cached:
        log(f"  resume: {len(cached)} folds cached: {sorted(cached)}")

    # GPU recipe uses max_bin=10000 / n_est=50000. On CPU we cap both to
    # keep wall-time feasible while preserving most of the split quality
    # (max_bin=1024 still gives >99% of max_bin=10000 split AUC on this data).
    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        enable_categorical=False, n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
        booster=XGB_BOOSTER,
    )
    if XGB_BOOSTER == "dart":
        # DART knobs tuned for cost/benefit: rate_drop=0.1 gives moderate
        # variance from gbtree, skip_drop=0.5 halves expected wall-time by
        # skipping dropout on 50% of rounds. Cap tree budget vs gbtree
        # since DART's per-tree cost grows linearly with round index.
        xgb_params.update(
            rate_drop=0.1, skip_drop=0.5,
            sample_type="uniform", normalize_type="tree",
            n_estimators=200 if SMOKE else 800,
            early_stopping_rounds=50 if SMOKE else 150,
        )

    train_scores = train["dgp_score"].values if DROP_SCORE_SET else None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        if fold in cached:
            ck_oof = ART / f"oof_{ck_prefix}_fold{fold}.npy"
            ck_test = ART / f"test_{ck_prefix}_fold{fold}.npy"
            vp = np.load(ck_oof)
            tp = np.load(ck_test)
            oof[va_idx] = vp
            test_pred += tp / N_FOLDS
            bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
            fold_scores.append(bal)
            log(f"  fold {fold} CACHED  argmax_bal_acc = {bal:.5f}")
            continue
        # Training-distribution rebalancing: drop rows whose dgp_score is in
        # DROP_SCORE_SET. Mirrors the 2026-04-21 v3 routed-XGB lever; OTE is
        # then fitted on the filtered training set so its statistics see only
        # the kept rows (still works because dropped rows are rule-trivial-Low).
        if DROP_SCORE_SET:
            keep_score = ~np.isin(train_scores[tr_idx], list(DROP_SCORE_SET))
            n_drop = int((~keep_score).sum())
            log(f"  DROP_SCORES={sorted(DROP_SCORE_SET)}: dropping {n_drop:,} of "
                f"{len(tr_idx):,} train rows ({100*n_drop/len(tr_idx):.2f}%)")
            tr_idx = tr_idx[keep_score]
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        # Cleanlab treatment is applied BEFORE OTE fit so that the target-
        # encoding statistics see the modified training set (matters for
        # `drop` and `relabel`; `downweight` only affects XGB sample_weight).
        fold_flagged = None
        if cleanlab_mask is not None:
            fold_flagged = cleanlab_mask[tr_idx]
            if CLEANLAB_TREATMENT == "drop":
                keep = ~fold_flagged
                X_tr = X_tr.iloc[keep].reset_index(drop=True)
                fold_tr_idx = tr_idx[keep]
                log(f"  drop: kept {len(X_tr):,} / {len(tr_idx):,} "
                    f"({int((~keep).sum()):,} flagged rows removed)")
            elif CLEANLAB_TREATMENT == "relabel":
                flagged_local = np.where(fold_flagged)[0]
                new_labels = teacher_pred[tr_idx][fold_flagged]
                X_tr.loc[flagged_local, TARGET] = new_labels
                fold_tr_idx = tr_idx
                log(f"  relabel: {len(flagged_local):,} rows reassigned "
                    f"to teacher argmax")
            else:  # downweight
                fold_tr_idx = tr_idx
        else:
            fold_tr_idx = tr_idx

        # OrderedTE on train only; apply to val+test.
        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a_ote)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        # Unshuffle back into original order for consistent row alignment.
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        y_tr = y[fold_tr_idx].copy() if CLEANLAB_TREATMENT != "relabel" else X_tr[TARGET].to_numpy()
        sw = compute_sample_weight("balanced", y_tr)
        if CLEANLAB_TREATMENT == "downweight":
            sw[fold_flagged] *= CLEANLAB_WEIGHT
            log(f"  downweight: {int(fold_flagged.sum()):,} flagged rows "
                f"weight ×{CLEANLAB_WEIGHT}")
        if ANCHOR_WEIGHT_ALPHA != 0:
            # Mech B: up-weight rows where LB-best 4-stack is uncertain.
            # Loaded once at module level via _anchor_uncertainty (computed
            # below), indexed into fold_tr_idx to get per-fold weights.
            au = _anchor_uncertainty[fold_tr_idx]
            sw = sw * (1.0 + ANCHOR_WEIGHT_ALPHA * au)
            log(f"  anchor-uncertainty weight α={ANCHOR_WEIGHT_ALPHA:.1f}: "
                f"u-mean={au.mean():.4f} u-max={au.max():.4f} "
                f"weight-range=[{sw.min():.3f},{sw.max():.3f}]")

        log(f"  training XGB on {len(feat_cols)} features, {len(X_tr):,} rows")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y_tr,
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        vp = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        tp = model.predict_proba(X_te[feat_cols]).astype(np.float32)

        # Mech A: boundary-confined TTA. Replace predictions at boundary
        # rows with K-perturbation average; non-boundary predictions stay
        # untouched. Boundary defined by anchor max-prob threshold.
        if TTA_BOUNDARY:
            au_va = _anchor_uncertainty[va_idx]  # uncertainty = 1 − max_prob
            mask_va = au_va > (1.0 - TTA_BOUNDARY_THRESH)
            au_te = _anchor_uncertainty_test
            mask_te = au_te > (1.0 - TTA_BOUNDARY_THRESH)
            n_b_va = int(mask_va.sum())
            n_b_te = int(mask_te.sum())
            log(f"  TTA: boundary rows va={n_b_va}/{len(va_idx)} ({100*n_b_va/len(va_idx):.1f}%), "
                f"te={n_b_te}/{len(test)} ({100*n_b_te/len(test):.1f}%)  σ={TTA_SIGMA} K={TTA_K}")
            if n_b_va > 0 or n_b_te > 0:
                vp = _apply_btta(model, feat_cols, vp, X_va, mask_va, fold)
                tp = _apply_btta(model, feat_cols, tp, X_te, mask_te, fold + 100)

        oof[va_idx] = vp
        test_pred += tp / N_FOLDS
        # checkpoint immediately for rehydrate resilience
        np.save(ART / f"oof_{ck_prefix}_fold{fold}.npy", vp)
        np.save(ART / f"test_{ck_prefix}_fold{fold}.npy", tp)
        bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={model.best_iteration}")

    # Inference-side rule routing: override predictions for rows whose
    # dgp_score is in DROP_SCORE_SET with deterministic Low (the rule's
    # output for scores ≤ 3). Applied to BOTH oof and test for OOF/LB
    # alignment; matches the 2026-04-21 v3 routed-infer pattern.
    if DROP_SCORE_SET:
        test_scores = test["dgp_score"].values
        val_route = np.isin(train_scores, list(DROP_SCORE_SET))
        test_route = np.isin(test_scores, list(DROP_SCORE_SET))
        rule_low = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        log(f"routing inference: {int(val_route.sum()):,} OOF rows + "
            f"{int(test_route.sum()):,} test rows -> Low (rule)")
        # OOF: only override rows that fell in val folds (they're all the train rows).
        oof[val_route] = rule_low
        test_pred[test_route] = rule_low

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    log(f"config: OTE_ALPHA={OTE_ALPHA}  XGB_BOOSTER={XGB_BOOSTER}  "
        f"suffix={VARIANT_SUFFIX!r}  smoke={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info, a_ote=OTE_ALPHA)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / f"oof_recipe_full_te{VARIANT_SUFFIX}.npy"
    test_path = ART / f"test_recipe_full_te{VARIANT_SUFFIX}.npy"
    np.save(oof_path, result["oof"])
    np.save(test_path, result["test"])
    log(f"wrote {oof_path} + {test_path}")

    # Build submission using tuned log-bias on test probs.
    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / f"submission_recipe_full_te{VARIANT_SUFFIX}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")
    log(f"  pred dist: {dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        variant_suffix=VARIANT_SUFFIX,
        ote_alpha=OTE_ALPHA, xgb_booster=XGB_BOOSTER,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
        feature_group_sizes={k: len(v) if isinstance(v, list) else v
                             for k, v in info.items() if k != "te_cols"},
        te_col_count=len(info["te_cols"]),
    )
    res_path = ART / f"recipe_full_te{VARIANT_SUFFIX}_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
