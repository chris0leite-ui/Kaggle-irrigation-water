"""W1 production rerun: top smoke configs on full 504k × 5-fold."""
from __future__ import annotations
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression
from scripts.dgp_formula import dgp_score
from scripts.w1_distill_host import build_X, NUMS, CATS

ART = "scripts/artifacts/"

# Top 3 smoke configs (sorted by tuned_oof, all tanh)
CONFIGS = [
    {"hidden_layer_sizes": (128, 64),      "activation": "tanh", "alpha": 1e-3, "name": "tanh_128_64"},
    {"hidden_layer_sizes": (64, 64),       "activation": "tanh", "alpha": 1e-3, "name": "tanh_64_64"},
    {"hidden_layer_sizes": (128, 128, 64), "activation": "tanh", "alpha": 1e-3, "name": "tanh_128_128_64"},
]
N_FOLDS = 5
MAX_ITER = 60


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


def coord_ascent_bias(log_p, y, init=None):
    cur = np.array(init) if init is not None else np.array([0., 0., 0.])
    best = balanced_accuracy_score(y, (log_p + cur).argmax(1))
    rounds = 0
    improved = True
    while improved and rounds < 8:
        improved = False
        for k in range(3):
            for db in np.linspace(-3.5, 3.5, 15):
                test_b = cur.copy(); test_b[k] = db
                ba_t = balanced_accuracy_score(y, (log_p + test_b).argmax(1))
                if ba_t > best + 1e-6:
                    best = ba_t; cur = test_b; improved = True
        rounds += 1
    return best, cur


def main():
    print(f"=== W1 PRODUCTION: {len(CONFIGS)} configs × {N_FOLDS}-fold × full data, max_iter={MAX_ITER} ===")
    t0 = time.time()
    train = pd.read_csv("data/train.csv", dtype_backend="numpy_nullable")
    test = pd.read_csv("data/test.csv", dtype_backend="numpy_nullable")
    y = train["Irrigation_Need"].astype(str).map({"Low": 0, "Medium": 1, "High": 2}).values.astype(np.int64)

    Xtr_full, cat_maps = build_X(train)
    Xte, _ = build_X(test, cat_maps)
    print(f"features={Xtr_full.shape[1]} train={Xtr_full.shape[0]} test={Xte.shape[0]} ({time.time()-t0:.1f}s)")

    # LB-best 4-stack reference
    lb3 = lbf((L("oof_recipe_full_te.npy"), .25),
              (L("oof_recipe_pseudolabel.npy"), .35),
              (L("oof_recipe_pseudolabel_seed7labeler.npy"), .40))
    s1 = lbf((lb3, .80), (L("oof_realmlp.npy"), .20))
    s2 = lbf((s1, .925), (iso(L("oof_xgb_nonrule.npy"), y), .075))
    final_lb = lbf((s2, .70), (iso(L("oof_xgb_metastack.npy"), y), .30))
    bias_lb = np.array([1.4324, 1.4689, 3.4008])
    lbbest_pred = (np.log(final_lb.clip(1e-12)) + bias_lb).argmax(1)
    lbbest_oof = balanced_accuracy_score(y, lbbest_pred)
    print(f"LB-best 4-stack tuned OOF: {lbbest_oof:.5f} ({(y != lbbest_pred).sum()} errs)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_iter = list(skf.split(np.zeros(len(y)), y))

    all_results = []
    for cfg in CONFIGS:
        print(f"\n--- config: {cfg['name']} ---")
        t1 = time.time()
        oof_p = np.zeros((len(y), 3), dtype=np.float32)
        test_p = np.zeros((len(Xte), 3), dtype=np.float32)
        for f, (tr, va) in enumerate(fold_iter):
            tf = time.time()
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr_full[tr])
            Xva = scaler.transform(Xtr_full[va])
            Xte_s = scaler.transform(Xte)
            mlp = MLPClassifier(
                hidden_layer_sizes=cfg["hidden_layer_sizes"],
                activation=cfg["activation"],
                alpha=cfg["alpha"],
                max_iter=MAX_ITER,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=42,
                solver="adam",
                learning_rate_init=0.001,
                batch_size=4096,
            )
            mlp.fit(Xtr, y[tr])
            oof_p[va] = mlp.predict_proba(Xva)
            test_p += mlp.predict_proba(Xte_s) / N_FOLDS
            print(f"  fold {f+1}: n_iter={mlp.n_iter_} loss={mlp.loss_:.4f} val_score={mlp.best_validation_score_:.4f} ({time.time()-tf:.1f}s)")
        # eval
        argmax_pred = oof_p.argmax(1)
        argmax_ba = balanced_accuracy_score(y, argmax_pred)
        log_p = np.log(oof_p.clip(1e-12))
        tuned_ba, tuned_bias = coord_ascent_bias(log_p, y)
        tuned_pred = (log_p + tuned_bias).argmax(1)
        e_us = (tuned_pred != y); e_lb = (lbbest_pred != y)
        jaccard = (e_us & e_lb).sum() / max((e_us | e_lb).sum(), 1)
        rec = {
            "name": cfg["name"], "hidden": str(cfg["hidden_layer_sizes"]), "activation": cfg["activation"],
            "argmax_oof": float(argmax_ba), "tuned_oof": float(tuned_ba),
            "tuned_bias": tuned_bias.tolist(),
            "errs": int(e_us.sum()), "lbbest_errs": int(e_lb.sum()),
            "jaccard_vs_lbbest": float(jaccard),
            "wall_s": time.time() - t1,
        }
        all_results.append(rec)
        # save oof + test
        np.save(ART + f"oof_w1_{cfg['name']}.npy", oof_p)
        np.save(ART + f"test_w1_{cfg['name']}.npy", test_p)
        flag = ""
        if tuned_ba >= 0.97 and jaccard < 0.80 and rec["errs"] <= rec["lbbest_errs"]:
            flag = "  *** PASS GATE ***"
        elif tuned_ba >= 0.97 and jaccard < 0.85:
            flag = "  (close)"
        print(f"  argmax {argmax_ba:.5f}  tuned {tuned_ba:.5f}  errs {rec['errs']}/{rec['lbbest_errs']}  J {jaccard:.3f}  ({rec['wall_s']/60:.1f} min){flag}")

    out = ART + "w1_prod_results.json"
    with open(out, "w") as f:
        json.dump({"lbbest_oof": float(lbbest_oof), "n_folds": N_FOLDS, "max_iter": MAX_ITER,
                   "results": all_results, "total_wall_s": time.time() - t0}, f, indent=2)
    print(f"\nresults → {out}, total wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
