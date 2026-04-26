"""Stage 1: manifold characterization on score=6 ∩ rule=Medium override domain.

Goal: figure out whether true-H rows are feature-separable from true-M rows
inside the score=6 ∩ rule_pred=Medium override slice. v2 specialist hits AUC
0.938 here; the question is whether that's a model-capacity issue (room to
push to 0.97+ with better FE / model) or an information ceiling (the
features don't carry the signal needed to clear break-even precision 8.8%).

Outputs:
  - per-feature AUC + Cohen's d + KS-stat (single-feature ranking)
  - missed-H residual analysis: are bottom-50% true-H by v2 prob feature-
    distinct from found-H, or feature-indistinguishable from true-M?
  - top-K simple non-linear combinations (ratio / log / power) ranked by
    AUC on the binary M-vs-H task within the override slice
  - L2-LR oracle AUC on the full feature set (best-case linear ceiling)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import CLS2IDX, add_distance_features  # noqa: E402

ART = Path("scripts/artifacts")
OUT = ART / "score6_manifold_stage1_results.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc_directional(y: np.ndarray, x: np.ndarray) -> float:
    """Direction-agnostic AUC (max of x and -x as scorer)."""
    a = roc_auc_score(y, x)
    return max(a, 1.0 - a)


def topn_precision(y: np.ndarray, score: np.ndarray, ns: list[int]) -> dict:
    order = np.argsort(-score)
    out = {}
    n_pos = int(y.sum())
    for n in ns:
        if n > len(y):
            continue
        top_idx = order[:n]
        prec = float(y[top_idx].mean())
        recall = float(y[top_idx].sum() / max(n_pos, 1))
        out[f"n_{n}"] = {"prec": prec, "recall": recall, "correct": int(y[top_idx].sum())}
    return out


def main() -> None:
    log("loading train")
    tr = pd.read_csv("data/train.csv")
    tr = add_distance_features(tr)

    y_full = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int8)
    score = tr["dgp_score"].to_numpy().astype(np.int8)
    rule = tr["rule_pred"].to_numpy().astype(np.int8)

    domain = (score == 6) & (rule == 1)  # rule=Medium at score=6
    n_dom = int(domain.sum())
    n_l = int(((y_full == 0) & domain).sum())
    n_m = int(((y_full == 1) & domain).sum())
    n_h = int(((y_full == 2) & domain).sum())
    log(f"override domain (score=6, rule=Medium): n={n_dom:,}  L={n_l}  M={n_m}  H={n_h}")
    log(f"  break-even precision = H/(H+M) = {n_h/(n_h+n_m):.4f}  (ignoring L)")

    # Restrict to true M/H rows for the binary characterization
    mh_mask = domain & ((y_full == 1) | (y_full == 2))
    is_h = (y_full == 2)[mh_mask].astype(np.int8)
    log(f"M+H rows in domain: {mh_mask.sum():,}  ({is_h.sum()} positives)")

    # v2 baseline AUC + top-N precision
    oof_v2 = np.load(ART / "oof_spec6_mh_v2.npy")[mh_mask]
    auc_v2 = roc_auc_score(is_h, oof_v2)
    log(f"v2 specialist AUC on this slice: {auc_v2:.5f}")
    v2_topn = topn_precision(is_h, oof_v2, [50, 100, 200, 500, 1000, 2000])
    for n_str, d in v2_topn.items():
        log(f"  v2 {n_str}: correct={d['correct']:>4}  prec={d['prec']:.4f}  recall={d['recall']:.4f}")

    # Per-feature univariate analysis
    feats = [
        "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
        "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "sm_x_rf", "tc_x_ws",
    ]
    df_dom = tr.loc[mh_mask, feats].reset_index(drop=True).astype(np.float32)
    log(f"single-feature AUC ranking ({len(feats)} feats):")
    per_feat = []
    for f in feats:
        x = df_dom[f].to_numpy()
        if np.isnan(x).any() or x.std() < 1e-12:
            continue
        au = auc_directional(is_h, x)
        d = (x[is_h == 1].mean() - x[is_h == 0].mean()) / max(x.std(), 1e-9)
        ks, p = ks_2samp(x[is_h == 1], x[is_h == 0])
        per_feat.append({"feat": f, "auc": float(au), "cohen_d": float(d),
                         "ks": float(ks), "p": float(p)})
    per_feat.sort(key=lambda r: -r["auc"])
    for r in per_feat[:10]:
        log(f"  {r['feat']:30s} auc={r['auc']:.4f}  d={r['cohen_d']:+.3f}  ks={r['ks']:.3f}")

    # Simple non-linear combinations: ratios + log + signed sqrt of pairs of top-6
    log("non-linear pair combinations (ratio / log-diff / signed-sqrt-product):")
    top6 = [r["feat"] for r in per_feat[:6]]
    pair_results = []
    for i, a in enumerate(top6):
        for b in top6[i+1:]:
            x_a = df_dom[a].to_numpy()
            x_b = df_dom[b].to_numpy()
            for kind, x in [
                ("ratio", x_a / (x_b + 1e-6)),
                ("log_diff", np.log(np.abs(x_a) + 1) - np.log(np.abs(x_b) + 1)),
                ("ssp", np.sign(x_a * x_b) * np.sqrt(np.abs(x_a * x_b))),
                ("plus", x_a + x_b),
                ("minus", x_a - x_b),
            ]:
                if np.isnan(x).any() or np.isinf(x).any() or x.std() < 1e-12:
                    continue
                try:
                    au = auc_directional(is_h, x)
                    pair_results.append({"feats": (a, b), "kind": kind, "auc": float(au)})
                except Exception:
                    continue
    pair_results.sort(key=lambda r: -r["auc"])
    for r in pair_results[:15]:
        log(f"  {r['kind']:9s} {r['feats'][0]:24s} × {r['feats'][1]:24s}  auc={r['auc']:.4f}")

    # L2-LR oracle on full feature set: best-case LINEAR ceiling
    log("L2-LR oracle (5-fold) on standardized features:")
    from sklearn.model_selection import StratifiedKFold
    X = df_dom.to_numpy()
    y = is_h
    oof_lr = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        Xs = sc.transform(X[tr_idx])
        Xv = sc.transform(X[va_idx])
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(Xs, y[tr_idx])
        oof_lr[va_idx] = lr.predict_proba(Xv)[:, 1]
    auc_lr = roc_auc_score(y, oof_lr)
    log(f"  LR oracle AUC: {auc_lr:.5f}  (vs v2: {auc_v2:.5f})")
    lr_topn = topn_precision(y, oof_lr, [50, 100, 200, 500, 1000, 2000])

    # Missed-H residual: bottom-50% of true-H by v2 prob — feature-distinct from
    # found-H, or feature-indistinguishable from true-M?
    log("missed-H residual analysis (bottom-50% of true-H by v2 prob):")
    h_idx = np.where(is_h == 1)[0]
    h_v2 = oof_v2[h_idx]
    med_h = np.median(h_v2)
    missed_h_idx = h_idx[h_v2 < med_h]
    found_h_idx = h_idx[h_v2 >= med_h]
    m_idx = np.where(is_h == 0)[0]
    log(f"  missed-H: n={len(missed_h_idx)}  found-H: n={len(found_h_idx)}  M: n={len(m_idx)}")

    # For each top-10 feature, how does missed-H mean compare to M mean and found-H mean?
    # If missed-H mean ~ M mean, missed-H rows are feature-indistinguishable from M
    # (information ceiling). If missed-H mean ~ found-H mean, v2 under-fits.
    residual_table = []
    for r in per_feat[:10]:
        f = r["feat"]
        x = df_dom[f].to_numpy()
        m_mean = x[m_idx].mean()
        m_std = x[m_idx].std()
        f_mean = x[found_h_idx].mean()
        u_mean = x[missed_h_idx].mean()
        # Z-distance from M mean (in M std units)
        z_found = (f_mean - m_mean) / max(m_std, 1e-9)
        z_missed = (u_mean - m_mean) / max(m_std, 1e-9)
        ratio = z_missed / z_found if abs(z_found) > 1e-6 else float("nan")
        residual_table.append({
            "feat": f, "m_mean": float(m_mean), "found_h_mean": float(f_mean),
            "missed_h_mean": float(u_mean), "z_found": float(z_found),
            "z_missed": float(z_missed), "missed_to_found_ratio": float(ratio),
        })
    log(f"  feat                        z(found-H from M)   z(missed-H from M)   ratio")
    for r in residual_table:
        log(f"  {r['feat']:28s} {r['z_found']:+.3f}              {r['z_missed']:+.3f}             {r['missed_to_found_ratio']:+.2f}")

    # Save everything
    out = {
        "domain": {"n": n_dom, "L": n_l, "M": n_m, "H": n_h,
                   "break_even_precision": n_h / (n_h + n_m)},
        "v2": {"auc": float(auc_v2), "topn": v2_topn},
        "lr_oracle": {"auc": float(auc_lr), "topn": lr_topn},
        "per_feature": per_feat,
        "pair_combinations_top15": pair_results[:15],
        "missed_H_residual": residual_table,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
