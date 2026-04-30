"""L2 — fold loop + final test predict."""
from __future__ import annotations

import json
import time
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def fit_predict_ncm(emb_tr, y_tr, emb_eval):
    return MahalanobisNCM(n_classes=3).fit(emb_tr, y_tr).predict_proba_macro_recall(emb_eval)


def run_pipeline(start_t: float):
    log(f"config: SMOKE={IS_SMOKE} EPOCHS={EPOCHS} BS={BATCH_SIZE} "
        f"EMBED_DIM={EMBED_DIM} N_FOLDS={N_FOLDS}")
    device = _select_device()
    log(f"device: {device}")

    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()
    X_tr_all, X_te, _ = build_feat_matrix(train, test, info)
    log(f"feat matrix: train={X_tr_all.shape}  test={X_te.shape}")

    scaler = StandardScaler().fit(X_tr_all)
    X_tr_all = scaler.transform(X_tr_all).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    fold_metrics = []

    splits = list(skf.split(X_tr_all, y))[:N_FOLDS]
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        if time.time() - start_t > TOTAL_KILL_SEC:
            log(f"WALL kill at fold {fold} (>{TOTAL_KILL_SEC}s)")
            break
        log(f"=== fold {fold}/{N_FOLDS}: tr={len(tr_idx)} va={len(va_idx)} ===")
        emb = Embedder(X_tr_all.shape[1], embed_dim=EMBED_DIM, device=device)
        emb.fit(X_tr_all[tr_idx], y[tr_idx], EPOCHS, BATCH_SIZE)
        emb_tr = emb.transform(X_tr_all[tr_idx])
        emb_va = emb.transform(X_tr_all[va_idx])
        log("  fitting MahalanobisNCM + predicting val")
        probs_va = fit_predict_ncm(emb_tr, y[tr_idx], emb_va)
        oof[va_idx] = probs_va
        bal = balanced_accuracy_score(y[va_idx], probs_va.argmax(1))
        fold_metrics.append(float(bal))
        log(f"  fold {fold} argmax bal_acc = {bal:.5f}")
        # Per-fold checkpoint (rehydrate-resilient).
        np.save(OUT_DIR / f"oof_l2_supcon_ncm_partial.npy", oof)

    # Final test prediction: train embedding on FULL train.
    log("=== final test prediction: embed on full train ===")
    emb = Embedder(X_tr_all.shape[1], embed_dim=EMBED_DIM, device=device)
    emb.fit(X_tr_all, y, EPOCHS, BATCH_SIZE)
    emb_tr_full = emb.transform(X_tr_all)
    emb_te = emb.transform(X_te)
    log("  fitting MahalanobisNCM on full train + predicting test")
    test_probs = fit_predict_ncm(emb_tr_full, y, emb_te)

    results = dict(smoke=IS_SMOKE, n_folds=len(fold_metrics), epochs=EPOCHS,
                   embed_dim=EMBED_DIM, fold_argmax_bal=fold_metrics)
    if len(fold_metrics) == 5:
        argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
        results["argmax_bal_acc"] = float(argmax_bal)
        log(f"OOF argmax = {argmax_bal:.5f}")

    suffix = "_smoke" if IS_SMOKE else ""
    np.save(OUT_DIR / f"oof_l2_supcon_ncm{suffix}.npy", oof)
    np.save(OUT_DIR / f"test_l2_supcon_ncm{suffix}.npy", test_probs)
    with open(OUT_DIR / f"l2_supcon_ncm_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    log("done")
