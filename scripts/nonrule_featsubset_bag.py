"""Feature-subset bagging on the top non-rule features.

Train 5 XGB sub-models, each on a different 4-feature subset drawn from
the 7 most-informative non-rule features (4 continuous with significant
Cohen's d on flipped rows + 3 key categoricals). Each model sees a
genuinely different view of the data, so their disagreements are
driven by information access rather than architecture — a stronger
diversity source than seed-bagging or LGBM/XGB swaps.

Ensemble = log-space average across the 5 sub-models.

Protocol: same 5-fold stratified split (seed=42), fixed greedy bias,
log-blend sweep. Submission if fixed-bias OOF lifts >= +0.0003 over
current base (greedy + XGB-nonrule @ alpha=0.15, OOF 0.97421).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

# Top 7 non-rule features (by flip-row Cohen's d and domain signal).
# Continuous signal-bearers:
TOP_CONT = ["Humidity", "Previous_Irrigation_mm",
            "Electrical_Conductivity", "Field_Area_hectare"]
# Categoricals w/ plausible regional/crop structure:
TOP_CAT = ["Region", "Crop_Type", "Soil_Type"]
TOP_FEATS = TOP_CONT + TOP_CAT  # 7 features total

# 5 subsets of 4 features each. Coverage: each feature appears 2-3x.
SUBSETS = {
    "A": ["Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity", "Region"],
    "B": ["Humidity", "Field_Area_hectare", "Crop_Type", "Soil_Type"],
    "C": ["Previous_Irrigation_mm", "Field_Area_hectare", "Soil_Type", "Region"],
    "D": ["Electrical_Conductivity", "Crop_Type", "Region", "Soil_Type"],
    "E": ["Humidity", "Previous_Irrigation_mm", "Crop_Type", "Field_Area_hectare"],
}

ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
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
    return bias, float(best)


def log_blend_n(probs_list, weights):
    total = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        total = total + w * np.log(np.clip(p, 1e-9, 1.0))
    total -= total.max(1, keepdims=True)
    e = np.exp(total)
    return e / e.sum(1, keepdims=True)


def train_subset(X_full, X_test_full, y, feats, cat_cols, n_folds=N_FOLDS):
    """Return OOF (N, 3) and test (Nt, 3) for a single 4-feature XGB."""
    X = X_full[feats].copy()
    X_test = X_test_full[feats].copy()
    for c in feats:
        if c in cat_cols:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test = np.zeros((len(X_test), 3), dtype=np.float64)

    params = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=1.0,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=SEED,
    )
    dte = xgb.DMatrix(X_test, enable_categorical=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            params, dtr, num_boost_round=2000,
            evals=[(dva, "val")],
            early_stopping_rounds=80,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test += booster.predict(dte, iteration_range=(0, bi + 1)) / n_folds
        log(f"    fold {fold+1}/{n_folds} best_iter={bi}  "
            f"argmax={balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1)):.5f}  "
            f"({time.time()-t0:.1f}s)")
    return oof, test


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    # Encode all potentially-used cat cols once.
    X = tr.drop(columns=[ID, TARGET]).copy()
    X_test = te.drop(columns=[ID]).copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32")
        X_test[c] = te[c].map(mapping).astype("int32")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # Sanity: all top feats are in X.
    for f in TOP_FEATS:
        assert f in X.columns, f

    # Train 5 sub-models.
    oofs = {}
    tests = {}
    standalones = {}
    for name, feats in SUBSETS.items():
        log(f"subset {name}: {feats}")
        oof, test = train_subset(X, X_test, y, feats, cat_cols)
        oofs[name] = oof
        tests[name] = test
        argmax = balanced_accuracy_score(y, oof.argmax(1))
        _, tuned = tune_log_bias(oof, y, prior)
        standalones[name] = {"argmax": float(argmax), "tuned": float(tuned)}
        log(f"  subset {name} standalone  argmax={argmax:.5f}  tuned={tuned:.5f}")
        np.save(ART / f"oof_featsubset_{name}.npy", oof)
        np.save(ART / f"test_featsubset_{name}.npy", test)

    # Log-mean ensemble across 5 subsets.
    log("log-mean ensemble across 5 subsets")
    ens_oof = log_blend_n(list(oofs.values()), [1.0 / len(oofs)] * len(oofs))
    ens_test = log_blend_n(list(tests.values()), [1.0 / len(tests)] * len(tests))
    ens_argmax = balanced_accuracy_score(y, ens_oof.argmax(1))
    _, ens_tuned = tune_log_bias(ens_oof, y, prior)
    log(f"5-subset ensemble  argmax={ens_argmax:.5f}  tuned={ens_tuned:.5f}")
    np.save(ART / "oof_featsubset_ens.npy", ens_oof)
    np.save(ART / "test_featsubset_ens.npy", ens_test)

    # Load greedy + base.
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_xgb_nr = np.load(ART / "oof_xgb_nonrule.npy")
    test_xgb_nr = np.load(ART / "test_xgb_nonrule.npy")
    greedy_res = json.loads(Path(ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = greedy_res["greedy_tuned_oof"]

    base = log_blend_n([oof_xgb_nr, oof_greedy], [0.15, 0.85])
    base_test = log_blend_n([test_xgb_nr, test_greedy], [0.15, 0.85])
    base_ba = balanced_accuracy_score(
        y, (np.log(np.clip(base, 1e-9, 1.0)) + bias_greedy).argmax(1)
    )
    log(f"greedy OOF        = {tuned_greedy:.5f}")
    log(f"base (greedy+XGBnr 0.15) OOF = {base_ba:.5f}")

    results = {
        "subsets": {k: list(v) for k, v in SUBSETS.items()},
        "standalones": standalones,
        "ensemble_standalone": {"argmax": float(ens_argmax), "tuned": float(ens_tuned)},
        "greedy_tuned_oof": tuned_greedy,
        "base_blend_oof": float(base_ba),
        "greedy_bias": bias_greedy.tolist(),
        "sweeps": {},
    }

    # (a) Ensemble alone onto greedy.
    log("sweep (a): 5-subset ensemble onto greedy (fixed bias)")
    sa = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b = log_blend_n([ens_oof, oof_greedy], [alpha, 1 - alpha])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(1)
        )
        sa.append({"alpha": alpha, "oof": float(ba),
                   "delta_vs_greedy": float(ba - tuned_greedy)})
        log(f"  alpha_ens={alpha:.2f}  OOF={ba:.5f}  Δ greedy={ba - tuned_greedy:+.5f}")
    results["sweeps"]["ens_onto_greedy"] = sa

    # (b) Ensemble stacked onto base (greedy + XGB-nonrule @ 0.15).
    log("sweep (b): 5-subset ensemble onto base (fixed bias)")
    sb = []
    for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        b = log_blend_n([ens_oof, base], [beta, 1 - beta])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(1)
        )
        sb.append({"beta": beta, "oof": float(ba),
                   "delta_vs_base": float(ba - base_ba)})
        log(f"  beta_ens={beta:.2f}  OOF={ba:.5f}  Δ base={ba - base_ba:+.5f}")
    results["sweeps"]["ens_onto_base"] = sb

    # (c) 3-way: XGB-nonrule + ensemble + greedy.
    log("sweep (c): XGB-nonrule + ensemble + greedy (fixed bias)")
    sc = []
    best_c = {"oof": -1.0}
    for alpha in [0.05, 0.10, 0.15]:
        for beta in [0.05, 0.10, 0.15, 0.20, 0.25]:
            w_g = 1 - alpha - beta
            if w_g < 0.4:
                continue
            b = log_blend_n([oof_xgb_nr, ens_oof, oof_greedy], [alpha, beta, w_g])
            ba = balanced_accuracy_score(
                y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(1)
            )
            entry = {"alpha_xgb": alpha, "beta_ens": beta, "w_greedy": w_g,
                     "oof": float(ba),
                     "delta_vs_base": float(ba - base_ba)}
            sc.append(entry)
            log(f"  a_xgb={alpha:.2f} b_ens={beta:.2f} w_gr={w_g:.2f}  "
                f"OOF={ba:.5f}  Δ base={ba - base_ba:+.5f}")
            if ba > best_c["oof"]:
                best_c = entry
    results["sweeps"]["xgb_plus_ens_onto_greedy"] = sc
    results["best_2d"] = best_c

    # (d) Individual-subset diagnostic: each onto base with small weight.
    log("sweep (d): each individual subset onto base at beta=0.10")
    sd = []
    for name, oof in oofs.items():
        b = log_blend_n([oof, base], [0.10, 0.90])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(1)
        )
        sd.append({"subset": name, "beta": 0.10, "oof": float(ba),
                   "delta_vs_base": float(ba - base_ba)})
        log(f"  {name} β=0.10  OOF={ba:.5f}  Δ base={ba - base_ba:+.5f}")
    results["sweeps"]["individual_subsets_onto_base"] = sd

    best_a = max(sa, key=lambda d: d["oof"])
    best_b = max(sb, key=lambda d: d["oof"])
    log(f"\nbest ens onto greedy:  α={best_a['alpha']:.2f}  OOF={best_a['oof']:.5f}  "
        f"Δ greedy={best_a['delta_vs_greedy']:+.5f}")
    log(f"best ens onto base:    β={best_b['beta']:.2f}  OOF={best_b['oof']:.5f}  "
        f"Δ base={best_b['delta_vs_base']:+.5f}")
    log(f"best 3-way:            a={best_c.get('alpha_xgb',0):.2f} "
        f"b={best_c.get('beta_ens',0):.2f} g={best_c.get('w_greedy',0):.2f}  "
        f"OOF={best_c['oof']:.5f}  Δ base={best_c.get('delta_vs_base',0):+.5f}")

    # Submission gate.
    cands = []
    if best_b["delta_vs_base"] >= 3e-4:
        cands.append(("ens_onto_base", best_b,
                      log_blend_n([ens_test, base_test],
                                  [best_b["beta"], 1 - best_b["beta"]])))
    if best_c.get("delta_vs_base", 0) >= 3e-4:
        cands.append(("3way_xgb_ens_greedy", best_c, log_blend_n(
            [test_xgb_nr, ens_test, test_greedy],
            [best_c["alpha_xgb"], best_c["beta_ens"], best_c["w_greedy"]]
        )))
    if best_a["delta_vs_greedy"] >= 3e-4 and best_a["oof"] > base_ba + 3e-4:
        cands.append(("ens_onto_greedy", best_a,
                      log_blend_n([ens_test, test_greedy],
                                  [best_a["alpha"], 1 - best_a["alpha"]])))

    if cands:
        kind, info, test_blend = max(cands, key=lambda t: t[1]["oof"])
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(1)
        sub_path = OUT / f"submission_greedy_featsubset_{kind}.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}  (kind={kind})")
        results["action"] = f"ready_to_submit_{kind}"
        results["submission_path"] = str(sub_path)
    else:
        log("no OOF lift clears +0.0003 threshold — no submission")
        results["action"] = "no_submission"

    with open(ART / "nonrule_featsubset_bag_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/nonrule_featsubset_bag_results.json")


if __name__ == "__main__":
    main()
