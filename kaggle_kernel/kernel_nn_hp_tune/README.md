# NN HP tuning on digit-enriched features

FT-Transformer over num + digit + cat tokens, with Optuna HP search
on fold-0 and 5-fold refit. Designed to test whether a NN can break
the tree-ensemble ceiling (LB 0.97468) when it sees the same digit
quantisation signal that made digit-XGB the current best.

## Layout (keep modular — `../CLAUDE.md` § "keep files short")

- `features.py` — dist + digit FE (mirrors `scripts/benchmark_dist.py`
  and `scripts/digit_features.py`).
- `model.py` — FT-Transformer with NumericalTokenizer + CategoricalTokenizer
  (reused for digit cols, one 10-way embedding per column-position).
- `train.py` — fold training loop, Balanced Softmax, log-bias tuning.
- `data.py` — CSV → (num, dig, cat) arrays + cat vocabularies.
- `search.py` — Optuna objective sampling d_token / n_blocks / n_heads /
  dropouts / lr / wd / batch.
- `nn_digit_hp_tune.py` — Kaggle kernel entrypoint.
- `build.py` — concatenates the above into `dist/` for Kaggle upload.

## Push to Kaggle

```bash
# One-time: upload the baseline OOFs as a private dataset
cd ../ds_nn_baselines && kaggle datasets create -p . --dir-mode zip

# Build + push
cd ../kernel_nn_hp_tune && python build.py
cd dist && kaggle kernels push
```

The kernel reads `/kaggle/input/playground-series-s6e4/{train,test}.csv`
and `/kaggle/input/irrigation-nn-baselines/{oof_greedy_blend.npy,
oof_xgb_dist_digits.npy,test_xgb_dist_digits.npy}`.

## Env-var overrides

- `N_TRIALS` (default 20) — Optuna trials on fold-0.
- `TRIAL_EPOCHS` (default 8) — epochs per trial.
- `REFIT_EPOCHS` (default 25) — epochs per fold at refit.

## Gates

- Fold-1 Jaccard vs digit-XGB `>= 0.95` → abort (redundant predictor).
- Fold-1 NN errs `> 1.2×` digit-XGB's → WARN (blend lift unlikely per
  the 2026-04-22 "Jaccard necessary but not sufficient" rule).

## Outputs (`/kaggle/working/`)

- `oof_nn_digit.npy`, `test_nn_digit.npy` — aligned with the 5-fold
  `StratifiedKFold(seed=42)` split used for every other OOF.
- `submission_nn_digit_tuned.csv` — tuned log-bias standalone submit.
- `nn_digit_hp_tune_results.json` — best HP, study value, per-fold
  logs, blend-preview vs digit-XGB at α ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}.

## After download

Port OOF/test arrays into `scripts/artifacts/` and run a
`fixed-bias` α-sweep blend (mirror `scripts/blend_digits_ote.py`)
vs `oof_xgb_dist_digits.npy`. If blend peak `Δ >= +5e-4`, probe LB;
otherwise null result, log in `CLAUDE.md`.
