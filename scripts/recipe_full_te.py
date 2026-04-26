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
if EXTRA_FE:
    _parts.append(f"fex{EXTRA_FE}")
if GBY:
    _parts.append("gby")
if DROP_SCORE_SET:
    _parts.append("ds" + "".join(str(s) for s in sorted(DROP_SCORE_SET)))
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

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats_cols, dae_embed=dae_cols,
        extra_domain=extra_domain, extra_decimal=extra_decimal,
        extra_w8=extra_w8,
        gby=gby_cols,
        ood=ood_cols, knn10k=knn10k_cols,
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


# --------------------------------------------------------- training loop
def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           a_ote: float = 1.0) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

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
                     + info.get("ood", [])
                     + info.get("knn10k", []))
    drop_after_te = info["te_cols"]  # raw cats dropped; only TE cols retained

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

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

        log(f"  training XGB on {len(feat_cols)} features, {len(X_tr):,} rows")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y_tr,
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
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
