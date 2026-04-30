"""Option B: FAISS KNN-based geometric voter as 3rd OTHER.

Mechanism:
  Per row, find k=50 nearest neighbors in TRAIN feature space, take
  majority train label as KNN-vote. OOF: leave-one-fold-out (fit index
  on tr rows, query va). Test: fit on full train, query test.

Use the resulting KNN-vote argmax as a 3rd voter alongside {raw, tier1b}
for k=2 plurality / unanimous overrides on top of 4b (LB 0.98150).

Why this is structurally novel:
  - All prior OTHERS are probabilistic classifiers (XGB, CatBoost, RF, NN)
  - KNN is non-parametric, gradient-free
  - Errors concentrate on sparse-neighborhood rows, different failure mode
  - First geometric similarity voter on this problem

Features: 36 numeric distance/rule features (cheaper than full recipe FE,
captures rule-axis geometry that drives override decisions).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, add_distance_features  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
SEED = 42
K_NN = 50


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)
    n_tr, n_te = len(train), len(test)

    log("Building features (36-dim dist set)")
    train_dist = add_distance_features(train.drop(columns=[TARGET]))
    test_dist = add_distance_features(test)
    feat_cols = [c for c in train_dist.columns
                 if pd.api.types.is_numeric_dtype(train_dist[c])
                 and train_dist[c].dtype != bool]
    feat_cols = [c for c in feat_cols if train_dist[c].dtype.kind in "fiub"]
    log(f"  {len(feat_cols)} features")
    Xtr = train_dist[feat_cols].to_numpy().astype(np.float32)
    Xte = test_dist[feat_cols].to_numpy().astype(np.float32)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    d = Xtr_s.shape[1]
    log(f"  shapes: train {Xtr_s.shape}, test {Xte_s.shape}")

    # ===== OOF KNN votes (5-fold seed=42 leave-one-fold-out) =====
    log(f"OOF KNN: 5-fold leave-one-fold-out, k={K_NN}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_votes = np.zeros((n_tr, 3), dtype=np.float32)  # softvote per class
    oof_argmax = np.full(n_tr, -1, dtype=np.int32)

    for fold, (tr, va) in enumerate(skf.split(Xtr_s, y)):
        t0 = time.time()
        # FAISS Flat L2 (exact). With 36 dims and ~500k rows it's fast.
        index = faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(Xtr_s[tr]))
        # Query val rows for K nearest neighbors among tr rows
        D, I = index.search(np.ascontiguousarray(Xtr_s[va]), K_NN)
        # I is indices INTO tr; map to absolute row ids
        nn_labels = y[tr][I]  # (n_va, K)
        # Per-row majority class
        for c in range(3):
            oof_votes[va, c] = (nn_labels == c).mean(axis=1)
        oof_argmax[va] = oof_votes[va].argmax(1)
        log(f"  fold {fold+1}: query {len(va)} rows, {time.time()-t0:.1f}s")

    # OOF KNN argmax bal_acc
    bal_oof = balanced_accuracy_score(y, oof_argmax)
    pcr_oof = per_class_recall(y, oof_argmax)
    bias, tuned = tune_log_bias(oof_votes, y, prior)
    log(f"\nKNN OOF argmax bal_acc: {bal_oof:.5f}")
    log(f"KNN OOF tuned (with bias): {tuned:.5f}  bias={bias.round(3).tolist()}")
    log(f"PCR=[L={pcr_oof[0]:.4f} M={pcr_oof[1]:.4f} H={pcr_oof[2]:.4f}]")

    # ===== Test KNN votes (full train index) =====
    log("Test KNN: building index on full train")
    index = faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(Xtr_s))
    t0 = time.time()
    D, I = index.search(np.ascontiguousarray(Xte_s), K_NN)
    log(f"  Test query: {time.time()-t0:.1f}s")
    nn_labels_te = y[I]  # (n_te, K)
    test_votes = np.zeros((n_te, 3), dtype=np.float32)
    for c in range(3):
        test_votes[:, c] = (nn_labels_te == c).mean(axis=1)
    test_argmax = test_votes.argmax(1)
    cnt = np.bincount(test_argmax, minlength=3)
    log(f"Test KNN argmax dist: L={cnt[0]} M={cnt[1]} H={cnt[2]}")

    # Save
    np.save(ART / "oof_n15_knn_voter.npy", oof_votes)
    np.save(ART / "test_n15_knn_voter.npy", test_votes)
    log(f"Saved KNN voter OOF + test")

    # ===== Build override candidates with KNN as 3rd voter =====
    log("\n=== Override candidates: 2-of-3 plurality (raw, tier1b, KNN) on top of 4b ===")
    # Load 4b anchor (LB 0.98150)
    anchor_4b = pd.read_csv(SUB / "submission_idea4b_selective_override.csv")[TARGET].map(CLS2IDX).to_numpy()
    # Load raw, tier1b argmaxes (test side)
    raw_t = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)
    raw_t /= np.clip(raw_t.sum(1, keepdims=True), 1e-9, None)
    t1b_t = np.load(ART / "test_tier1b_greedy_meta.npy").astype(np.float32)
    t1b_t /= np.clip(t1b_t.sum(1, keepdims=True), 1e-9, None)
    raw_o = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_o /= np.clip(raw_o.sum(1, keepdims=True), 1e-9, None)
    t1b_o = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    t1b_o /= np.clip(t1b_o.sum(1, keepdims=True), 1e-9, None)
    braw, _ = tune_log_bias(raw_o, y, prior)
    bt1b, _ = tune_log_bias(t1b_o, y, prior)
    rawtt = (np.log(np.clip(raw_t, 1e-9, 1.0)) + braw).argmax(1)
    t1btt = (np.log(np.clip(t1b_t, 1e-9, 1.0)) + bt1b).argmax(1)
    knn_test = test_argmax

    # 2-of-3 plurality
    oargs = np.stack([rawtt, t1btt, knn_test], axis=1)
    votes_test = np.zeros((n_te, 3), dtype=np.int32)
    for c in range(3):
        votes_test[:, c] = (oargs == c).sum(axis=1)
    not_anchor = (np.arange(3)[None, :] != anchor_4b[:, None])
    elig = (votes_test >= 2) & not_anchor
    votes_elig = np.where(elig, votes_test, -1)
    any_elig = elig.any(axis=1)
    chosen = votes_elig.argmax(axis=1)

    # Variants by direction
    log(f"\nDirection breakdown of 2-of-3 plurality (raw, tier1b, KNN) on 4b:")
    direction_counts = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            cnt = int((any_elig & (anchor_4b == a) & (chosen == c)).sum())
            if cnt > 0:
                direction_counts[(a, c)] = cnt
                log(f"  {IDX2CLS[a]}->{IDX2CLS[c]}: {cnt}")

    # OOF analog: 2-of-3 plurality on B (= v1 + k=2 raw,t1b unanimous OOF)
    v1_o = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_o /= np.clip(v1_o.sum(1, keepdims=True), 1e-9, None)
    bv1, _ = tune_log_bias(v1_o, y, prior)
    v1o = (np.log(np.clip(v1_o, 1e-9, 1.0)) + bv1).argmax(1)
    rawo = (np.log(np.clip(raw_o, 1e-9, 1.0)) + braw).argmax(1)
    t1bo = (np.log(np.clip(t1b_o, 1e-9, 1.0)) + bt1b).argmax(1)
    # B_oof = v1 + k=2 unanimous (raw, t1b)
    k2_mask = (rawo == t1bo) & (rawo != v1o)
    B_oof = v1o.copy()
    B_oof[k2_mask] = rawo[k2_mask]

    # Apply 2-of-3 plurality with KNN OOF voter
    oargs_oof = np.stack([rawo, t1bo, oof_argmax], axis=1)
    votes_oof = np.zeros((n_tr, 3), dtype=np.int32)
    for c in range(3):
        votes_oof[:, c] = (oargs_oof == c).sum(axis=1)
    not_B = (np.arange(3)[None, :] != B_oof[:, None])
    elig_oof = (votes_oof >= 2) & not_B
    votes_elig_oof = np.where(elig_oof, votes_oof, -1)
    any_elig_oof = elig_oof.any(axis=1)
    chosen_oof = votes_elig_oof.argmax(axis=1)

    # Per-direction OOF precision
    log(f"\n=== Per-direction OOF precision (2-of-3 plurality on B, KNN as 3rd) ===")
    direction_oof_stats = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            m = any_elig_oof & (B_oof == a) & (chosen_oof == c)
            n_d = int(m.sum())
            if n_d == 0: continue
            n_correct = int((y[m] == c).sum())
            prec = n_correct / n_d
            be = prior[c] / (prior[a] + prior[c])
            direction_oof_stats[(a, c)] = dict(
                n=n_d, prec=float(prec), be=float(be), margin=float(prec - be))
            log(f"  {IDX2CLS[a]}->{IDX2CLS[c]}: n={n_d:4d}  prec={prec:.4f}  BE={be:.4f}  margin={prec-be:+.4f}")

    # Build candidate variants on test
    log(f"\n=== Building test-side candidates ===")
    base_v1 = balanced_accuracy_score(y, v1o)
    B_bal = balanced_accuracy_score(y, B_oof)
    # Apply 2-of-3 plurality with all directions
    pred_full = B_oof.copy()
    for (a, c), s in direction_oof_stats.items():
        m = any_elig_oof & (B_oof == a) & (chosen_oof == c)
        pred_full[m] = c
    bal_full = balanced_accuracy_score(y, pred_full)
    # STRONG: only directions with margin > 0
    strong_dirs = [d for d, s in direction_oof_stats.items() if s["margin"] > 0]
    pred_strong = B_oof.copy()
    for (a, c) in strong_dirs:
        m = any_elig_oof & (B_oof == a) & (chosen_oof == c)
        pred_strong[m] = c
    bal_strong = balanced_accuracy_score(y, pred_strong)
    log(f"  v1 baseline:                      {base_v1:.5f}")
    log(f"  B (k=2 unanimous, LB 0.98140):    {B_bal:.5f}  Δ={B_bal-base_v1:+.5f}")
    log(f"  B + 2of3 KNN ALL directions:      {bal_full:.5f}  Δ vs B={bal_full-B_bal:+.5f}")
    log(f"  B + 2of3 KNN STRONG ({len(strong_dirs)} dirs):  {bal_strong:.5f}  Δ vs B={bal_strong-B_bal:+.5f}")

    # Test side
    pred_test_all = anchor_4b.copy()
    pred_test_strong = anchor_4b.copy()
    for (a, c), cnt in direction_counts.items():
        m = any_elig & (anchor_4b == a) & (chosen == c)
        pred_test_all[m] = c
        if (a, c) in strong_dirs:
            pred_test_strong[m] = c
    n_overrides_all = int((pred_test_all != anchor_4b).sum())
    n_overrides_strong = int((pred_test_strong != anchor_4b).sum())

    # Save submissions
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in pred_test_all]}).to_csv(
        SUB / "submission_n15_knn_layered_4b_ALL.csv", index=False)
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in pred_test_strong]}).to_csv(
        SUB / "submission_n15_knn_layered_4b_STRONG.csv", index=False)
    log(f"\nTest overrides: ALL={n_overrides_all}, STRONG={n_overrides_strong}")
    log(f"Saved candidate submissions")

    summary = {
        "knn_oof_argmax_bal_acc": float(bal_oof),
        "knn_oof_tuned": float(tuned),
        "knn_pcr": pcr_oof.tolist(),
        "B_oof_bal_acc": float(B_bal),
        "B_plus_knn_all_oof": float(bal_full),
        "B_plus_knn_strong_oof": float(bal_strong),
        "delta_all_vs_B": float(bal_full - B_bal),
        "delta_strong_vs_B": float(bal_strong - B_bal),
        "n_test_overrides_ALL": n_overrides_all,
        "n_test_overrides_STRONG": n_overrides_strong,
        "test_direction_counts": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": v
                                  for (a, c), v in direction_counts.items()},
        "oof_direction_stats": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": s
                                for (a, c), s in direction_oof_stats.items()},
    }
    with open(ART / "n15_knn_voter_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary")


if __name__ == "__main__":
    main()
