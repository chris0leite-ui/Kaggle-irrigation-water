"""Full-train diagnostic: AUC of (1 - P(synth)) for predicting rule-flip.

Earlier av_predicts_flip.py probed only the 10k AV-training subsample
(n_flip = 177). With av_full_predict.py we now have leak-free P(synth)
on all 630k train rows. Recompute the headline AUC + per-score
breakdown at full sample size.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.dist_shift.flip_manifold import _bits, _rule, _score
from scripts.dist_shift.loader import ARTI, load


def main() -> None:
    train, _, _ = load()
    p_synth = np.load(ARTI / "oof_av_p_synth_train.npy")
    p_orig = 1.0 - p_synth

    b = _bits(train)
    s = _score(b).to_numpy()
    rule = _rule(s)
    flip = (rule != train["Irrigation_Need"].to_numpy()).astype(int)

    n = len(train)
    n_flip = int(flip.sum())
    print(f"Full train n={n}  n_flip={n_flip} ({n_flip/n*100:.4f}%)")

    auc = roc_auc_score(flip, p_orig)
    print(f"\n=== HEADLINE: AUC(P(orig), flip) on full 630k = {auc:.4f} ===")

    a = p_orig[flip == 0]
    bb = p_orig[flip == 1]
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + bb.var(ddof=1)))
    d = (bb.mean() - a.mean()) / pooled if pooled > 0 else 0.0
    print(f"P(orig) flip vs clean:")
    print(f"  mean clean = {a.mean():.4f}  n = {len(a)}")
    print(f"  mean flip  = {bb.mean():.4f}  n = {len(bb)}")
    print(f"  Cohen's d  = {d:.3f}")

    # Per-score
    print("\nPer-score AUC of P(orig) for flip (full train):")
    rows = []
    for sc in sorted(set(s)):
        mask = (s == sc)
        n_m = int(mask.sum())
        n_fm = int(flip[mask].sum())
        if n_fm < 5 or n_fm == n_m:
            rows.append({"score": int(sc), "n": n_m, "n_flip": n_fm, "auc": "n/a"})
            continue
        a_sc = roc_auc_score(flip[mask], p_orig[mask])
        rows.append({"score": int(sc), "n": n_m, "n_flip": n_fm,
                     "auc": round(float(a_sc), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    # Top-K precision: sort by P(orig) desc, what fraction of top-K are flips?
    print("\nTop-K precision (sort by P(orig) desc):")
    order = np.argsort(-p_orig)
    flip_sorted = flip[order]
    base = n_flip / n
    pk_rows = []
    for k in [100, 500, 1000, 5000, 10000, 20000]:
        prec = float(flip_sorted[:k].mean())
        pk_rows.append({"K": k, "n_flip_in_top_K": int(flip_sorted[:k].sum()),
                        "precision": round(prec, 4),
                        "lift_vs_base": round(prec / base, 2)})
    print(pd.DataFrame(pk_rows).to_string(index=False))

    out = {
        "auc_p_orig_predicts_flip_full_train": float(auc),
        "cohen_d_p_orig_flip_vs_clean": float(d),
        "n_train": n, "n_flip": n_flip,
        "p_orig_mean_clean": float(a.mean()),
        "p_orig_mean_flip": float(bb.mean()),
        "per_score_auc_full": rows,
        "top_k_precision": pk_rows,
        "base_rate": float(base),
    }
    (ARTI / "av_predicts_flip_full_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {ARTI/'av_predicts_flip_full_results.json'}")


if __name__ == "__main__":
    main()
