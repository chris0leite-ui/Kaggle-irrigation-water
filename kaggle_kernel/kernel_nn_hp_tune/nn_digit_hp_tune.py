"""NN with HP tuning on digit-enriched features (Kaggle kernel entrypoint).

Pipeline:
  1. Load train/test, add dist + digit features.
  2. Optuna TPE search (N_TRIALS on fold-0, TRIAL_EPOCHS each).
  3. Refit best HP on 5-fold StratifiedKFold(seed=42), REFIT_EPOCHS each.
  4. Fold-1 Jaccard gate vs digit-XGB (LB-best baseline); abort if >= 0.95.
  5. Emit oof_nn_digit.npy, test_nn_digit.npy, submission CSV, results JSON.

Sibling modules (same dir): features.py, model.py, train.py, data.py, search.py.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

# GPU shim lives in _bootstrap.py (runs before any torch import at build time).
# For local dev, importing it here is harmless.
sys.path.insert(0, str(Path(__file__).parent))
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import optuna
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from data import load_prepared, CLASSES, IDX2CLS, TARGET, ID
from model import DigitFTTransformer
from train import make_loader, train_one_fold, tune_log_bias, predict_probs
from search import build_objective


SEED = 42
N_FOLDS = 5
N_TRIALS = int(os.environ.get("N_TRIALS", "20"))
TRIAL_EPOCHS = int(os.environ.get("TRIAL_EPOCHS", "8"))
REFIT_EPOCHS = int(os.environ.get("REFIT_EPOCHS", "25"))
JACCARD_KILL = 0.95
JACCARD_WARN = 0.85
ERR_WARN_RATIO = 1.20  # digit-XGB has 8,846 errors; warn if >20% more.

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_one(pattern):
    for p in Path("/kaggle/input").rglob(pattern):
        return p
    return None


def main():
    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={DEVICE}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    train_csv = find_one("train.csv")
    test_csv = find_one("test.csv")
    gate_digit_path = find_one("oof_xgb_dist_digits.npy")
    gate_greedy_path = find_one("oof_greedy_blend.npy")
    test_digit_path = find_one("test_xgb_dist_digits.npy")
    assert train_csv and test_csv, "missing competition CSVs"
    assert gate_digit_path and gate_greedy_path, "missing OOF gate arrays"

    log(f"train={train_csv}  test={test_csv}")
    log(f"gate digit-XGB={gate_digit_path}  gate greedy={gate_greedy_path}")

    data = load_prepared(train_csv, test_csv)
    log(f"features: {len(data['num_cols'])} num + {len(data['dig_cols'])} digit "
        f"+ {len(data['cat_cols'])} cat")
    log(f"priors: {dict(zip(CLASSES, data['prior'].round(4)))}")

    gate_digit = np.load(gate_digit_path)
    gate_greedy = np.load(gate_greedy_path)
    digit_err_total = int((gate_digit.argmax(axis=1) != data["y"]).sum())
    log(f"digit-XGB total OOF errors: {digit_err_total}")

    log(f"=== Optuna HP search: {N_TRIALS} trials x {TRIAL_EPOCHS} epochs ===")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    objective = build_objective(data, DEVICE, TRIAL_EPOCHS, log_fn=log)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best_hp = dict(study.best_params)
    log(f"best trial {study.best_trial.number}  val_bal={study.best_value:.5f}")
    log(f"best HP: {best_hp}")

    # Refit best HP on 5 folds
    y = data["y"]; prior = data["prior"]
    log_prior = torch.from_numpy(np.log(prior).astype(np.float32)).to(DEVICE)
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test_probs = np.zeros((len(data["x_num_te"]), 3), dtype=np.float64)
    fold_logs = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    tens_te = (
        torch.from_numpy(data["x_num_te"]).float(),
        torch.from_numpy(data["x_dig_te"]).long() if data["x_dig_te"] is not None else None,
        torch.from_numpy(data["x_cat_te"]).long() if data["x_cat_te"] is not None else None,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        log(f"=== refit fold {fold+1}/{N_FOLDS}  train {len(tr_idx)}  val {len(va_idx)} ===")
        torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
        model = DigitFTTransformer(
            n_num=data["x_num_tr"].shape[1],
            digit_cards=data["digit_cards"],
            cat_cards=data["cat_cards"],
            d_token=best_hp["d_token"], n_blocks=best_hp["n_blocks"],
            n_heads=best_hp["n_heads"], attn_drop=best_hp["attn_drop"],
            ffn_drop=best_hp["ffn_drop"],
        ).to(DEVICE)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            log(f"  model params: {n_params:,}")
        opt = torch.optim.AdamW(model.parameters(), lr=best_hp["lr"],
                                weight_decay=best_hp["wd"])
        loader = make_loader(
            data["x_num_tr"][tr_idx],
            data["x_dig_tr"][tr_idx] if data["x_dig_tr"] is not None else None,
            data["x_cat_tr"][tr_idx] if data["x_cat_tr"] is not None else None,
            y[tr_idx], batch=best_hp["batch"], shuffle=True,
        )
        tens_va = (
            torch.from_numpy(data["x_num_tr"][va_idx]).float(),
            torch.from_numpy(data["x_dig_tr"][va_idx]).long() if data["x_dig_tr"] is not None else None,
            torch.from_numpy(data["x_cat_tr"][va_idx]).long() if data["x_cat_tr"] is not None else None,
        )
        best_probs, best_bal = train_one_fold(
            model, opt, {"train": loader}, tens_va, y[va_idx], log_prior,
            epochs=REFIT_EPOCHS, device=DEVICE, base_lr=best_hp["lr"],
            log_fn=log, log_prefix="  ",
        )
        oof[va_idx] = best_probs
        test_fold = predict_probs(model, *tens_te, DEVICE)
        test_probs += test_fold / N_FOLDS
        entry = {"fold": fold + 1, "val_bal_acc_best": float(best_bal),
                 "seconds": round(time.time() - t0, 1)}
        fold_logs.append(entry)

        # Gate after fold 1
        if fold == 0:
            nn_pred = oof[va_idx].argmax(axis=1)
            dig_pred = gate_digit[va_idx].argmax(axis=1)
            e_nn = set(va_idx[nn_pred != y[va_idx]])
            e_dig = set(va_idx[dig_pred != y[va_idx]])
            inter = len(e_nn & e_dig)
            union = len(e_nn | e_dig) or 1
            jac = inter / union
            log(f"  fold-1 Jaccard(NN vs digit-XGB) = {jac:.4f}  "
                f"NN errs={len(e_nn)}  digit errs={len(e_dig)}")
            entry["jaccard_vs_digit_xgb"] = jac
            entry["nn_errs_fold1"] = len(e_nn)
            entry["digit_errs_fold1"] = len(e_dig)
            if jac >= JACCARD_KILL:
                log(f"  KILL: Jaccard {jac:.4f} >= {JACCARD_KILL}  --  aborting")
                json.dump({"killed_at_fold": 1, "best_hp": best_hp,
                           "study_best_value": float(study.best_value),
                           "fold_logs": fold_logs},
                          open(OUT / "nn_digit_hp_tune_results.json", "w"),
                          indent=2)
                return
            if len(e_nn) > ERR_WARN_RATIO * len(e_dig):
                log(f"  WARN: NN has {len(e_nn) / max(1,len(e_dig)):.2f}x digit-XGB's errors  "
                    f"-- blend lift unlikely")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"OOF argmax={argmax_bal:.5f}  reweight={reweight_bal:.5f}  "
        f"tuned={tuned_bal:.5f}  bias={bias.round(4).tolist()}")

    np.save(OUT / "oof_nn_digit.npy", oof)
    np.save(OUT / "test_nn_digit.npy", test_probs)
    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log("OOF tuned confusion matrix:\n" + str(pd.DataFrame(cm, index=CLASSES, columns=CLASSES)))

    tuned_test_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: data["id_test"],
                  TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT / "submission_nn_digit_tuned.csv", index=False)

    # Fixed-bias blend preview vs digit-XGB on OOF
    if test_digit_path is not None:
        test_digit = np.load(test_digit_path)
        preview = {}
        lp_nn = np.log(np.clip(oof, 1e-9, 1.0))
        lp_dig = np.log(np.clip(gate_digit, 1e-9, 1.0))
        for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            blended = (1 - a) * lp_dig + a * lp_nn
            _, bal = tune_log_bias(np.exp(blended), y, prior)
            preview[f"alpha={a:.2f}"] = float(bal)
        log(f"blend preview vs digit-XGB (tuned): {preview}")
    else:
        preview = None

    json.dump({
        "best_hp": best_hp,
        "study_best_value": float(study.best_value),
        "n_trials": N_TRIALS,
        "trial_epochs": TRIAL_EPOCHS,
        "refit_epochs": REFIT_EPOCHS,
        "class_priors": prior.tolist(),
        "log_bias": bias.tolist(),
        "argmax_bal_acc": float(argmax_bal),
        "reweight_bal_acc": float(reweight_bal),
        "tuned_bal_acc": float(tuned_bal),
        "blend_preview_vs_digit_xgb": preview,
        "fold_logs": fold_logs,
    }, open(OUT / "nn_digit_hp_tune_results.json", "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
