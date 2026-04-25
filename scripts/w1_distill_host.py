"""W1 — Distill-the-host MLP sweep.

Hypothesis: the host's label-generating NN (`brief.md:74`) was likely
a SMALL standard tabular MLP (3 layers, 128 hidden, ReLU). We've never
tried to MATCH that specific class — only kitchen-sink architectures
(FT-T, TabPFN, RealMLP, Trompt). Sweep narrow configs on raw features.

Decision rule per CLAUDE.md:
  - SMOKE first: 1 fold × ~12 configs on 200k subsample, max_iter=20.
  - If any config has tuned OOF >= 0.97 AND Jaccard < 0.80 vs LB-best:
    → run that config in production (5-fold, full data, max_iter=50).
  - Otherwise: kill the lever, log null.
"""
from __future__ import annotations
import os
import json
import time
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from scripts.dgp_formula import dgp_score


# ---------- config ----------
SMOKE = os.environ.get("SMOKE", "1") == "1"
SUBSAMPLE = 200_000 if SMOKE else None
N_FOLDS = 1 if SMOKE else 5
MAX_ITER = 20 if SMOKE else 50
ART = "scripts/artifacts/"
OUT_PREFIX = "w1_smoke" if SMOKE else "w1_prod"


# ---------- features (mimic the 19 raw cols the host's NN saw) ----------
NUMS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
        "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare"]
CATS = ["Mulching_Used", "Crop_Growth_Stage", "Crop_Type", "Soil_Type",
        "Season", "Irrigation_Type", "Water_Source", "Region"]


def build_X(df: pd.DataFrame, cat_maps: dict | None = None) -> tuple[np.ndarray, dict]:
    """One-hot cats + raw nums + DGP score as a feature."""
    parts = []
    if cat_maps is None:
        cat_maps = {}
        for c in CATS:
            cat_maps[c] = sorted(df[c].astype(str).unique().tolist())
    for c in CATS:
        vals = df[c].astype(str).values
        oh = np.zeros((len(df), len(cat_maps[c])), dtype=np.float32)
        for j, v in enumerate(cat_maps[c]):
            oh[:, j] = (vals == v).astype(np.float32)
        parts.append(oh)
    parts.append(df[NUMS].to_numpy(dtype=np.float32))
    score = dgp_score(df).astype(np.float32).reshape(-1, 1)
    parts.append(score)
    return np.hstack(parts), cat_maps


# ---------- sweep ----------
def sweep_configs(smoke: bool):
    """Generate config grid."""
    if smoke:
        return [
            {"hidden_layer_sizes": (64, 64),       "activation": "relu", "alpha": 1e-3},
            {"hidden_layer_sizes": (128, 128),     "activation": "relu", "alpha": 1e-3},
            {"hidden_layer_sizes": (128, 64),      "activation": "relu", "alpha": 1e-3},
            {"hidden_layer_sizes": (256, 128),     "activation": "relu", "alpha": 1e-3},
            {"hidden_layer_sizes": (128, 128, 64), "activation": "relu", "alpha": 1e-3},
            {"hidden_layer_sizes": (64, 64),       "activation": "tanh", "alpha": 1e-3},
            {"hidden_layer_sizes": (128, 128),     "activation": "tanh", "alpha": 1e-3},
            {"hidden_layer_sizes": (128, 64),      "activation": "tanh", "alpha": 1e-3},
            {"hidden_layer_sizes": (256, 128),     "activation": "tanh", "alpha": 1e-3},
            {"hidden_layer_sizes": (128, 128, 64), "activation": "tanh", "alpha": 1e-3},
            {"hidden_layer_sizes": (128, 128),     "activation": "logistic", "alpha": 1e-3},
            {"hidden_layer_sizes": (256, 128, 64), "activation": "relu", "alpha": 1e-2},
        ]
    # production: top configs from smoke get re-run by hand on full data
    raise NotImplementedError("set SMOKE=1 first; pick top configs by hand.")


# ---------- main ----------
def main():
    print(f"=== W1 distill-the-host (SMOKE={SMOKE}, MAX_ITER={MAX_ITER}, N_FOLDS={N_FOLDS}) ===")
    t0 = time.time()
    train = pd.read_csv("data/train.csv", dtype_backend="numpy_nullable")
    test = pd.read_csv("data/test.csv", dtype_backend="numpy_nullable")
    y_str = train["Irrigation_Need"].astype(str).values
    y = pd.Series(y_str).map({"Low": 0, "Medium": 1, "High": 2}).values.astype(np.int64)
    print(f"loaded train {len(train)} test {len(test)} in {time.time()-t0:.1f}s")

    if SUBSAMPLE is not None:
        rng = np.random.default_rng(42)
        # stratified subsample to match overall prior
        idx = []
        for k in [0, 1, 2]:
            ki = np.where(y == k)[0]
            n_k = int(round(SUBSAMPLE * (y == k).mean()))
            idx.extend(rng.choice(ki, size=min(n_k, len(ki)), replace=False))
        idx = np.array(idx); rng.shuffle(idx)
        train_sub = train.iloc[idx].reset_index(drop=True)
        y_sub = y[idx]
    else:
        train_sub = train; y_sub = y

    Xtr_full, cat_maps = build_X(train_sub)
    Xte, _ = build_X(test, cat_maps)
    print(f"features: {Xtr_full.shape[1]} cols ({Xtr_full.shape[0]} train, {Xte.shape[0]} test)")

    # Load LB-best 4-stack OOF predictions for Jaccard comparison
    from sklearn.isotonic import IsotonicRegression
    def L(n): return np.load(ART + n)
    def iso(p, y_):
        o = np.zeros_like(p)
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
            o[:, k] = ir.fit_transform(p[:, k], (y_ == k).astype(float))
        return o / o.sum(1, keepdims=True).clip(1e-9)
    def lbf(*pw):
        out = sum(w * np.log(p.clip(1e-12)) for p, w in pw)
        out = np.exp(out - out.max(1, keepdims=True))
        return out / out.sum(1, keepdims=True)

    lb3 = lbf((L("oof_recipe_full_te.npy"), .25),
              (L("oof_recipe_pseudolabel.npy"), .35),
              (L("oof_recipe_pseudolabel_seed7labeler.npy"), .40))
    s1 = lbf((lb3, .80), (L("oof_realmlp.npy"), .20))
    s2 = lbf((s1, .925), (iso(L("oof_xgb_nonrule.npy"), y), .075))
    final = lbf((s2, .70), (iso(L("oof_xgb_metastack.npy"), y), .30))
    bias = np.array([1.4324, 1.4689, 3.4008])
    lbbest_pred = (np.log(final.clip(1e-12)) + bias).argmax(1)
    print(f"LB-best 4-stack baseline tuned OOF: {balanced_accuracy_score(y, lbbest_pred):.5f}")

    # If smoke: subsample-aligned LB-best for Jaccard reference
    if SUBSAMPLE is not None:
        lbbest_ref = lbbest_pred[idx]
        y_ref = y_sub
    else:
        lbbest_ref = lbbest_pred
        y_ref = y

    skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=42)
    fold_iter = list(skf.split(np.zeros(len(y_sub)), y_sub))[:N_FOLDS]

    results = []
    configs = sweep_configs(SMOKE)
    print(f"sweeping {len(configs)} configs...")

    for ci, cfg in enumerate(configs):
        oof_p = np.zeros((len(y_sub), 3), dtype=np.float32)
        oof_pred_int = np.full(len(y_sub), -1, dtype=np.int64)
        t1 = time.time()
        for f, (tr, va) in enumerate(fold_iter):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr_full[tr])
            Xva = scaler.transform(Xtr_full[va])
            mlp = MLPClassifier(
                hidden_layer_sizes=cfg["hidden_layer_sizes"],
                activation=cfg["activation"],
                alpha=cfg["alpha"],
                max_iter=MAX_ITER,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                solver="adam",
                learning_rate_init=0.001,
                batch_size=4096,
            )
            mlp.fit(Xtr, y_sub[tr])
            oof_p[va] = mlp.predict_proba(Xva)
            oof_pred_int[va] = mlp.predict(Xva)
        # tuned bal_acc via per-class log-bias coord ascent
        from itertools import product
        best_ba = balanced_accuracy_score(y_sub[oof_pred_int >= 0], oof_pred_int[oof_pred_int >= 0])
        best_bias = (0., 0., 0.)
        # fast coord-ascent
        valid = oof_pred_int >= 0
        log_p = np.log(oof_p[valid].clip(1e-12))
        cur_b = np.array([0., 0., 0.])
        improved = True
        rounds = 0
        while improved and rounds < 8:
            improved = False
            for k in range(3):
                for db in np.linspace(-3.5, 3.5, 15):
                    test_b = cur_b.copy(); test_b[k] = db
                    pp = (log_p + test_b).argmax(1)
                    ba_t = balanced_accuracy_score(y_sub[valid], pp)
                    if ba_t > best_ba + 1e-6:
                        best_ba = ba_t
                        cur_b = test_b
                        improved = True
            rounds += 1
        tuned_pred = (log_p + cur_b).argmax(1)
        # Jaccard vs LB-best on the eval rows
        lbbest_eval = lbbest_ref[valid]
        e_us = (tuned_pred != y_sub[valid])
        e_lb = (lbbest_eval != y_sub[valid])
        inter = (e_us & e_lb).sum()
        union = (e_us | e_lb).sum()
        jaccard = inter / union if union > 0 else 1.0

        n_errs = e_us.sum()
        n_lb_errs = e_lb.sum()

        rec = {
            "config_idx": ci,
            "hidden": str(cfg["hidden_layer_sizes"]),
            "activation": cfg["activation"],
            "alpha": cfg["alpha"],
            "tuned_oof": float(best_ba),
            "bias": cur_b.tolist(),
            "errs": int(n_errs),
            "lbbest_errs_on_eval": int(n_lb_errs),
            "jaccard_vs_lbbest": float(jaccard),
            "wall_s": time.time() - t1,
        }
        results.append(rec)
        flag = ""
        if best_ba >= 0.97 and jaccard < 0.80 and n_errs <= n_lb_errs:
            flag = "  *** PASS GATE ***"
        elif best_ba >= 0.97 and jaccard < 0.85:
            flag = "  (close: low Jaccard but check magnitude)"
        print(f"  cfg{ci:2d} hid={str(cfg['hidden_layer_sizes']):<18} "
              f"act={cfg['activation']:<8} α={cfg['alpha']:.0e} "
              f"tuned={best_ba:.5f} errs={n_errs}/{n_lb_errs} J={jaccard:.3f} "
              f"({rec['wall_s']:.1f}s){flag}")

    # save
    out_json = ART + f"{OUT_PREFIX}_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "smoke": SMOKE,
            "subsample": SUBSAMPLE,
            "n_folds": N_FOLDS,
            "max_iter": MAX_ITER,
            "n_configs": len(configs),
            "results": results,
            "total_wall_s": time.time() - t0,
        }, f, indent=2)
    print(f"\nresults → {out_json}, total wall {(time.time()-t0)/60:.1f} min")

    # Sort + summary
    print("\n=== SMOKE summary (sorted by tuned_oof desc) ===")
    sorted_results = sorted(results, key=lambda r: -r["tuned_oof"])
    for r in sorted_results[:5]:
        print(f"  tuned={r['tuned_oof']:.5f} J={r['jaccard_vs_lbbest']:.3f} errs={r['errs']} "
              f"hid={r['hidden']} act={r['activation']}")

    # Decision
    passing = [r for r in results if r["tuned_oof"] >= 0.97
               and r["jaccard_vs_lbbest"] < 0.80
               and r["errs"] <= r["lbbest_errs_on_eval"]]
    if passing:
        print(f"\n*** {len(passing)} config(s) PASS gate — pick for production rerun ***")
    else:
        print(f"\nNULL: 0 configs passed (tuned≥0.97 AND J<0.80 AND errs≤anchor)")


if __name__ == "__main__":
    main()
