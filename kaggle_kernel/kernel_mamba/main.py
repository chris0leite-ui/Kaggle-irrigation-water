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
    BATCH_SIZE, LR, WEIGHT_DECAY,
    D_MODEL, N_LAYERS, D_STATE, D_CONV, EXPAND, DROPOUT,
    FOLD1_KILL_SEC, TOTAL_KILL_SEC, PROBE_SUBSAMPLE,
    KAGGLE_INPUT, OUT_DIR,
)
from features import load_data, build_frame
from cv import run_cv, save_outputs


def main() -> None:
    boot()
    print(f"[main] SMOKE={SMOKE} PROBE={PROBE} N_FOLDS={N_FOLDS} "
          f"MAX_FOLDS={MAX_FOLDS} N_EPOCHS={N_EPOCHS}", flush=True)
    train, test, orig, test_ids = load_data(KAGGLE_INPUT, SMOKE)
    train, test, orig = build_frame(train, test, orig)
    suffix = "_smoke" if SMOKE else ("_probe" if PROBE else "")
    oof, test_pred, fold_ba, folds_done, y = run_cv(
        train, test, orig,
        n_folds=N_FOLDS, max_folds=MAX_FOLDS, n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
        d_conv=D_CONV, expand=EXPAND, dropout=DROPOUT,
        fold1_kill_s=FOLD1_KILL_SEC, total_kill_s=TOTAL_KILL_SEC,
        out_dir=OUT_DIR, suffix=suffix,
        probe_subsample=PROBE_SUBSAMPLE if PROBE else 0,
    )
    save_outputs(OUT_DIR, oof, test_pred, test_ids, fold_ba,
                 folds_done, N_FOLDS, y, suffix=suffix)


if __name__ == "__main__":
    main()
