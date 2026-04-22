"""EBM variant of the non-rule-features-only model + blend.

EBM (Explainable Boosting Machine) from interpretml is architecturally
distinct from LGBM/XGB: GA2M — shape functions per feature + explicit
pairwise interactions, no deep tree-interactions. On the same 13 non-
rule features, EBM's decision surface is fundamentally different from
gradient-boosted trees — if the NN generator's flip signal has shape-
+-pairwise structure, EBM may capture what tree models miss.

Protocol: same 5-fold stratified split (seed=42), fixed-greedy-bias
blend sweep. LB-probe if lifts >= +0.0003 vs current base
(greedy + XGB-nonrule @ alpha=0.15, OOF 0.97421).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {ID, TARGET}

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


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"non-rule features ({len(nonrule_cols)}): {nonrule_cols}")

    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    # Integer-encode categoricals; EBM will treat them as 'nominal' via
    # feature_types param.
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32")
        X_test[c] = te[c].map(mapping).astype("int32")
    for c in num_cols:
        X[c] = X[c].astype(np.float32)
        X_test[c] = X_test[c].astype(np.float32)

    feat_types = ["nominal" if c in cat_cols else "continuous" for c in nonrule_cols]
    log(f"EBM feature types: "
        f"{sum(t == 'nominal' for t in feat_types)} nominal, "
        f"{sum(t == 'continuous' for t in feat_types)} continuous")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    log("training 5-fold EBM 3-class on non-rule features")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    X_np = X.values
    X_test_np = X_test.values

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y)):
        t0 = time.time()
        ebm = ExplainableBoostingClassifier(
            feature_names=nonrule_cols,
            feature_types=feat_types,
            interactions=10,      # pairwise interactions
            outer_bags=4,         # reduced from 8 default for speed
            learning_rate=0.02,
            max_bins=256,
            min_samples_leaf=50,
            random_state=SEED,
            n_jobs=-1,
        )
        ebm.fit(X_np[tr_idx], y[tr_idx])
        oof[va_idx] = ebm.predict_proba(X_np[va_idx])
        test_pred += ebm.predict_proba(X_test_np) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  argmax bal={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"EBM nonrule standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")
    np.save(ART / "oof_ebm_nonrule.npy", oof)
    np.save(ART / "test_ebm_nonrule.npy", test_pred)

    log("loading greedy + existing XGB-nonrule + LGBM-nonrule OOFs")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_xgb = np.load(ART / "oof_xgb_nonrule.npy")
    test_xgb = np.load(ART / "test_xgb_nonrule.npy")
    greedy_res = json.loads(Path(ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = greedy_res["greedy_tuned_oof"]

    base = log_blend_n([oof_xgb, oof_greedy], [0.15, 0.85])
    base_test = log_blend_n([test_xgb, test_greedy], [0.15, 0.85])
    lp = np.log(np.clip(base, 1e-9, 1.0))
    base_ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
    log(f"baseline (greedy + XGB-nonrule 0.15) OOF = {base_ba:.5f}")

    results = {
        "ebm_standalone_argmax": float(argmax_bal),
        "ebm_standalone_tuned":  float(tuned_bal),
        "greedy_tuned_oof": tuned_greedy,
        "base_xgb_nonrule_blend_oof": float(base_ba),
        "sweeps": {},
    }

    # (a) EBM alone onto greedy.
    log("sweep (a): EBM alone onto greedy (fixed bias)")
    sa = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b = log_blend_n([oof, oof_greedy], [alpha, 1 - alpha])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        sa.append({"alpha": alpha, "oof": float(ba),
                   "delta_vs_greedy": float(ba - tuned_greedy)})
        log(f"  alpha_ebm={alpha:.2f}  OOF = {ba:.5f}  "
            f"Δ greedy = {ba - tuned_greedy:+.5f}")
    results["sweeps"]["ebm_alone_onto_greedy"] = sa

    # (b) EBM stacked onto (greedy + XGB-nonrule @ 0.15).
    log("sweep (b): EBM stacked onto (greedy + XGB-nonrule @ 0.15)")
    sb = []
    for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        b = log_blend_n([oof, base], [beta, 1 - beta])
        ba = balanced_accuracy_score(
            y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        sb.append({"beta": beta, "oof": float(ba),
                   "delta_vs_base": float(ba - base_ba)})
        log(f"  beta_ebm={beta:.2f}  OOF = {ba:.5f}  Δ base = {ba - base_ba:+.5f}")
    results["sweeps"]["ebm_onto_base"] = sb

    # (c) 3-way: XGB + EBM + greedy, sweep both weights.
    log("sweep (c): XGB + EBM + greedy 2D sweep (fixed bias)")
    sc = []
    best_c = {"oof": -1.0}
    for alpha in [0.05, 0.10, 0.15]:
        for beta in [0.05, 0.10, 0.15, 0.20]:
            w_g = 1 - alpha - beta
            if w_g < 0.4:
                continue
            bl = log_blend_n([oof_xgb, oof, oof_greedy], [alpha, beta, w_g])
            ba = balanced_accuracy_score(
                y, (np.log(np.clip(bl, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
            )
            entry = {"alpha_xgb": alpha, "beta_ebm": beta, "w_greedy": w_g,
                     "oof": float(ba),
                     "delta_vs_base": float(ba - base_ba)}
            sc.append(entry)
            log(f"  a_xgb={alpha:.2f} b_ebm={beta:.2f} w_gr={w_g:.2f}  "
                f"OOF = {ba:.5f}  Δ base = {ba - base_ba:+.5f}")
            if ba > best_c["oof"]:
                best_c = entry
    results["sweeps"]["xgb_plus_ebm_onto_greedy"] = sc
    results["best_2d"] = best_c

    best_a = max(sa, key=lambda d: d["oof"])
    best_b = max(sb, key=lambda d: d["oof"])
    log(f"\nbest onto greedy:   α={best_a['alpha']:.2f}  OOF={best_a['oof']:.5f}  "
        f"Δ greedy={best_a['delta_vs_greedy']:+.5f}")
    log(f"best onto base:     β={best_b['beta']:.2f}  OOF={best_b['oof']:.5f}  "
        f"Δ base={best_b['delta_vs_base']:+.5f}")
    log(f"best 3-way:         a={best_c.get('alpha_xgb', 0):.2f} "
        f"b={best_c.get('beta_ebm', 0):.2f} g={best_c.get('w_greedy', 0):.2f}  "
        f"OOF={best_c['oof']:.5f}  Δ base={best_c.get('delta_vs_base', 0):+.5f}")

    # Pick best candidate to submit (>= +0.0003 over base).
    cands = []
    if best_b["delta_vs_base"] >= 3e-4:
        cands.append(("onto_base_1d", best_b,
                      log_blend_n([test_pred, base_test],
                                  [best_b["beta"], 1 - best_b["beta"]])))
    if best_c.get("delta_vs_base", 0) >= 3e-4:
        cands.append(("3way_xgb_ebm_greedy", best_c, log_blend_n(
            [test_xgb, test_pred, test_greedy],
            [best_c["alpha_xgb"], best_c["beta_ebm"], best_c["w_greedy"]]
        )))
    if best_a["delta_vs_greedy"] >= 3e-4 and best_a["oof"] > base_ba + 3e-4:
        cands.append(("ebm_onto_greedy", best_a,
                      log_blend_n([test_pred, test_greedy],
                                  [best_a["alpha"], 1 - best_a["alpha"]])))

    if cands:
        kind, info, test_blend = max(cands, key=lambda t: t[1]["oof"])
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub_path = OUT / f"submission_greedy_nonrule_ebm_{kind}.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}  (kind={kind})")
        results["action"] = f"ready_to_submit_{kind}"
        results["submission_path"] = str(sub_path)

        blend_oof = {
            "onto_base_1d": log_blend_n([oof, base],
                                        [info.get("beta", 0), 1 - info.get("beta", 0)]),
            "3way_xgb_ebm_greedy": log_blend_n(
                [oof_xgb, oof, oof_greedy],
                [info.get("alpha_xgb", 0), info.get("beta_ebm", 0), info.get("w_greedy", 0)]
            ),
            "ebm_onto_greedy": log_blend_n([oof, oof_greedy],
                                           [info.get("alpha", 0), 1 - info.get("alpha", 0)]),
        }[kind]
        cm = confusion_matrix(
            y, (np.log(np.clip(blend_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        log(f"OOF confusion matrix (chosen blend):\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
    else:
        log("no OOF lift clears +0.0003 threshold — no submission")
        results["action"] = "no_submission"

    with open(ART / "nonrule_ebm_blend_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/nonrule_ebm_blend_results.json")


if __name__ == "__main__":
    main()
