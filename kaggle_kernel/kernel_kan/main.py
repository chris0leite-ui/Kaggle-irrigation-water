"""Thin orchestrator — imports from sibling modules + calls run.

Entrypoint referenced in kernel-metadata.json. For Kaggle push,
build.py inlines all sibling modules into a single distributable
code_file (kaggle ignores siblings in a kernel push); local
`python main.py` works directly via the imports below.
"""
from __future__ import annotations

from boot import boot
from config import (
    SMOKE, PROBE, N_FOLDS, MAX_FOLDS, N_EPOCHS,
    BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTHING,
    HIDDEN, GRID_SIZE, SPLINE_ORDER, GRID_RANGE, DROPOUT,
    FOLD1_KILL_SEC, TOTAL_KILL_SEC,
    KAGGLE_INPUT, OUT_DIR,
)
from features import load_data, build_arrays
from cv import run_cv, save_outputs


def main() -> None:
    boot()
    print(f"[main] SMOKE={SMOKE} PROBE={PROBE} N_FOLDS={N_FOLDS} "
          f"MAX_FOLDS={MAX_FOLDS} N_EPOCHS={N_EPOCHS}", flush=True)
    train, test, orig, test_ids = load_data(KAGGLE_INPUT, SMOKE)
    X_tr, X_te, X_or, y_tr, y_or, feat_dim = build_arrays(train, test, orig)
    suffix = "_smoke" if SMOKE else ("_probe" if PROBE else "")
    oof, test_pred, fold_ba, folds_done = run_cv(
        X_tr, X_te, X_or, y_tr, y_or,
        n_folds=N_FOLDS, max_folds=MAX_FOLDS, n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        hidden=HIDDEN, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER,
        grid_range=GRID_RANGE, dropout=DROPOUT,
        label_smoothing=LABEL_SMOOTHING,
        fold1_kill_s=FOLD1_KILL_SEC, total_kill_s=TOTAL_KILL_SEC,
        out_dir=OUT_DIR, suffix=suffix,
    )
    save_outputs(OUT_DIR, oof, test_pred, test_ids, fold_ba,
                 folds_done, N_FOLDS, y_tr, suffix=suffix)


if __name__ == "__main__":
    main()
