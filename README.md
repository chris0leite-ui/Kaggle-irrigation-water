# Predicting Irrigation Need — Playground Series S6E4

<https://www.kaggle.com/competitions/playground-series-s6e4>

3-class classification (`Low` / `Medium` / `High`) on 19 tabular agronomy
features. Train 630k rows, test 270k rows. Metric: **balanced accuracy**.
Severe class imbalance: 58.7 / 37.9 / **3.3** %.

## Pipeline

Two trainings of the same heavy-reg-XGB recipe + a single 50/50 log-blend:

```
recipe_full_te.py   →  oof_recipe_full_te.npy,   test_recipe_full_te.npy   (LB 0.97939)
recipe_pseudolabel  →  oof_recipe_pseudolabel.npy, test_recipe_pseudolabel.npy
                       (uses recipe's test probs as τ=0.98 pseudo-labels)
blend.py            →  submission_recipe_pseudolabel_blend.csv               (LB 0.97998)
```

Decision rule: `argmax(0.5·log(test_a) + 0.5·log(test_b) + [1.4324, 1.4689, 3.4008])`
where the bias is recipe's per-class log-bias from coord-ascent on OOF.

This is within ~1σ of the prior LB-best 4-stack (0.98094) under fold-std
σ≈0.00088, with one-tenth the moving parts.

## Reproduce

```bash
./bootstrap.sh                      # installs deps + downloads data
python scripts/recipe_full_te.py    # ~55 min CPU, fold OOF + test probs
python scripts/recipe_pseudolabel.py # ~48 min CPU, augmented retrain
python scripts/blend.py             # ~5 s, writes the final submission
```

`SMOKE=1` shrinks each step to a 5-min sanity run (20k rows × 2 folds).

## Layout

```
scripts/
  recipe_features.py    feature-engineering blocks
  recipe_ote.py         OrderedTE (per-class cumulative shuffled target encoder)
  recipe_full_te.py     XGB on FE + OTE (foundation model)
  recipe_pseudolabel.py same pipeline, augmented with τ=0.98 pseudo-labels
  common.py             tune_log_bias, log_blend, fast_bal_acc
  blend.py              50/50 log-blend → submission CSV
data/                   competition data + 10k original (gitignored)
submissions/            LB-confirmed reference CSVs
```

## Feature blocks (recipe_full_te)

8 raw cats + 11 raw nums + threshold flags (4) + LR-formula logits (3) +
cat×cat pair combos (28) + digit features `floor(v·10^k) mod 10` for k=−4..+3
(~70) + num-as-cat (11) + FREQ over cats+combos (~44) + ORIG mean/std on the
10k original (~48), then **OrderedTE (a=1)** on every categorical (~117 keys ×
3 classes ≈ 350 OTE features). Total ≈ 440 features fed to XGB.

XGB: `max_depth=4, max_leaves=30, lr=0.1, reg_alpha=5, reg_lambda=5,
subsample=0.8, colsample_bytree=0.8, n_estimators=3000, early_stopping=200`,
class-balanced sample weights. 5-fold StratifiedKFold(seed=42).
