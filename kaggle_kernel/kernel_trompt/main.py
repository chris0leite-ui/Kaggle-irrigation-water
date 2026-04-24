"""Thin orchestrator — imports from sibling modules + calls run.

This file is the entrypoint referenced in kernel-metadata.json. For
Kaggle push, the build.py step inlines all sibling modules into a
single distributable code_file (kaggle ignores sibling files in a
kernel push), but local `python main.py` works directly via the
imports below.
"""
from __future__ import annotations
import torch

from boot import boot
from config import (
    IS_SMOKE, SMOKE, N_FOLDS, N_EPOCHS,
    FOLD1_KILL_SEC, TOTAL_KILL_SEC,
    KAGGLE_INPUT, OUT_DIR,
)
from features import load_data, build_frame
from cv import run_cv, save_outputs


def main() -> None:
    boot()
    print(f"[main] SMOKE={SMOKE} N_FOLDS={N_FOLDS} N_EPOCHS={N_EPOCHS}",
          flush=True)
    train, test, orig, test_ids = load_data(KAGGLE_INPUT, SMOKE)
    train, test, orig = build_frame(train, test, orig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}", flush=True)
    oof, test_pred, fold_ba, folds_done, y = run_cv(
        train, test, orig, device,
        n_folds=N_FOLDS, n_epochs=N_EPOCHS,
        fold1_kill_s=FOLD1_KILL_SEC, total_kill_s=TOTAL_KILL_SEC,
    )
    save_outputs(OUT_DIR, oof, test_pred, test_ids, fold_ba,
                 folds_done, N_FOLDS, y)


if __name__ == "__main__":
    main()
