"""Aggregate per-fold SVGP checkpoints into the final OOF + test arrays.

After running RUN_FOLD=1..5 sequentially (each saves a per-fold checkpoint),
this script: (a) loads each fold's val OOF rows and inserts into a global
(630000, 3) OOF array using the SAME StratifiedKFold(seed=42) split,
(b) averages the 5 test predictions, (c) writes the aggregate OOF + test
+ a results JSON, ready for svgp_blend_gate.py and svgp_minimal_check.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import CLS2IDX  # noqa: E402
from tier1b_helpers import ART, BIAS, bal_at_bias, load_y  # noqa: E402

SEED = 42
N_FOLDS = 5
SUFFIX = "_svgp"


def main():
    t0 = time.time()
    y = load_y()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # Need a dummy X aligned with y for split; use indices.
    X_dummy = np.zeros((len(y), 1), dtype=np.float32)

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds = []
    fold_argmax = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dummy, y)):
        f_oof_p = ART / f"oof_xgb_metastack{SUFFIX}_fold{fold+1}.npy"
        f_test_p = ART / f"test_xgb_metastack{SUFFIX}_fold{fold+1}.npy"
        if not (f_oof_p.exists() and f_test_p.exists()):
            print(f"[agg] MISSING fold {fold+1}: {f_oof_p}")
            sys.exit(1)
        v = np.load(f_oof_p).astype(np.float32)
        if v.shape[0] != len(va_idx):
            print(f"[agg] fold {fold+1} val shape mismatch: {v.shape[0]} vs {len(va_idx)}")
            sys.exit(1)
        oof[va_idx] = v
        test_folds.append(np.load(f_test_p).astype(np.float32))
        b = balanced_accuracy_score(y[va_idx], v.argmax(1))
        fold_argmax.append(b)
        print(f"[agg] fold {fold+1} val_argmax={b:.5f}")

    test = np.mean(test_folds, axis=0).astype(np.float32)
    np.save(ART / f"oof_xgb_metastack{SUFFIX}.npy", oof)
    np.save(ART / f"test_xgb_metastack{SUFFIX}.npy", test)

    argmax_oof = balanced_accuracy_score(y, oof.argmax(1))
    tuned_oof = bal_at_bias(oof, y)
    print(f"\n=== aggregated SVGP meta ===")
    print(f"  argmax OOF        = {argmax_oof:.5f}")
    print(f"  tuned @recipe-bias = {tuned_oof:.5f}")
    print(f"  fold argmax std   = {np.std(fold_argmax):.5f}")
    out = dict(
        argmax_oof=float(argmax_oof), tuned_oof=float(tuned_oof),
        fold_argmax=[float(x) for x in fold_argmax],
        elapsed=time.time() - t0,
    )
    (ART / f"svgp_metastack{SUFFIX}_results.json").write_text(json.dumps(out, indent=2))
    print(f"[agg] wrote results JSON, wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
