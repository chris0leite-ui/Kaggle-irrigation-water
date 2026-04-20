"""
Final LGBM evaluation: run the best Optuna-found params on full 630k
with the same 5-fold stratified CV and log-bias coord-ascent as
benchmark.py, so the result is a drop-in replacement comparable to the
0.97097 baseline.

Reads `scripts/artifacts/hyperopt_lgbm_200k.json` (or --params-file).
Writes `oof_lgbm_tuned.npy`, `test_lgbm_tuned.npy`,
`bench_tuned_results.json`, and the argmax / tuned-bias submission CSVs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
SUB = ROOT / "submissions"
SUB.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params-file", type=str,
                   default=str(ART / "hyperopt_lgbm_200k.json"))
    p.add_argument("--tag", type=str, default="tuned")
    args = p.parse_args()

    cfg = json.loads(Path(args.params_file).read_text())
    best = cfg["best_params"]
    log(f"using best params from {args.params_file}")
    log(f"hyperopt bal_acc (prior-reweight, 200k): {cfg['best_value']:.5f}")

    log("loading data")
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")

    num_cols = [c for c in tr.select_dtypes(include=[np.number]).columns
                if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        m = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(m).astype("int32")
        te[c] = te[c].map(m).astype("int32")

    X = tr[num_cols + cat_cols].copy()
    X_test = te[num_cols + cat_cols].copy()
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"priors: {dict(zip(CLASSES, prior.round(4)))}")

    params = dict(
        objective="multiclass",
        num_class=len(CLASSES),
        metric="multi_logloss",
        verbose=-1,
        seed=SEED,
        **best,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                          categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx],
                          categorical_feature=cat_cols, reference=dtr)
        model = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(150, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    results: list[dict] = []

    def bench(name: str, pred: np.ndarray) -> None:
        bal = balanced_accuracy_score(y, pred)
        cm = confusion_matrix(y, pred).tolist()
        results.append({"name": name, "bal_acc": float(bal), "cm": cm})

    bench("LGBM-tuned + argmax", oof.argmax(axis=1))
    bench("LGBM-tuned + prior-reweight argmax", (oof / prior).argmax(axis=1))

    log("coord-ascent over per-class log-bias")
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))

    def score_bias(b: np.ndarray) -> float:
        return balanced_accuracy_score(y, (log_oof + b).argmax(axis=1))

    bias = -np.log(prior)
    best_score = score_bias(bias)
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(score_bias(base))
            j = int(np.argmax(scores))
            if scores[j] > best_score + 1e-6:
                bias[k] = bias[k] + grid[j]
                best_score = scores[j]
                improved = True
        if not improved:
            break
    log(f"  best bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={best_score:.5f}")
    bench("LGBM-tuned + tuned log-bias", (log_oof + bias).argmax(axis=1))

    print("\n=== results (OOF balanced accuracy) ===")
    w = max(len(r["name"]) for r in results)
    for r in results:
        print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")

    print("\nbaseline (bench_results.json): LGBM + tuned log-bias = 0.97097")
    best_rule = max(results, key=lambda r: r["bal_acc"])
    print(f"delta vs baseline: {best_rule['bal_acc'] - 0.97097:+.5f}")

    np.save(ART / f"oof_lgbm_{args.tag}.npy", oof)
    np.save(ART / f"test_lgbm_{args.tag}.npy", test_pred)
    out = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "params": best,
        "class_priors": prior.tolist(),
        "log_bias": bias.tolist(),
        "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
        "best_rule": best_rule["name"],
        "baseline_tuned_log_bias": 0.97097,
    }
    (ART / f"bench_{args.tag}_results.json").write_text(json.dumps(out, indent=2))
    log(f"saved: {ART}/bench_{args.tag}_results.json")

    pd.DataFrame(
        {ID: te[ID], TARGET: [IDX2CLS[i] for i in test_pred.argmax(axis=1)]}
    ).to_csv(SUB / f"submission_lgbm_{args.tag}_argmax.csv", index=False)
    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame(
        {ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}
    ).to_csv(SUB / f"submission_lgbm_{args.tag}_tuned.csv", index=False)
    log(f"submissions written to {SUB}/")


if __name__ == "__main__":
    main()
