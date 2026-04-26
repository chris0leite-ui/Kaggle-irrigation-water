"""Stage 1b: refocus on the ACTUAL override domain.

Stage 1a used (score=6 ∩ rule=Medium) which is the loose pre-teacher slice.
The real override target is (score=6 ∩ teacher_argmax=Medium) — the rows
where the teacher (LB-best 3-way) is already saying Medium and the
specialist would flip them to High.

Two questions:
  1. On the teacher-residual domain, what's v2's AUC, top-N precision,
     and theoretical override capacity at break-even (~8.1%)?
  2. Are the missed-H rows in this domain feature-distinguishable from
     M rows when we INCLUDE teacher meta-features (P_M, P_H, margin)?
     If teacher-meta makes them distinguishable, room exists for a
     stronger specialist; otherwise the lever is information-bounded.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import CLS2IDX, add_distance_features, log_blend  # noqa: E402

ART = Path("scripts/artifacts")
OUT = ART / "score6_manifold_stage1b_results.json"

# Macro-recall break-even precision (from CLAUDE.md):
#   each correct flip M→H = +1/N_H, each wrong flip = -1/N_M_total
#   break_even = N_H_total / (N_H_total + N_M_total) = 21009 / 260083 ≈ 0.0808
N_H_TOTAL = 21009
N_M_TOTAL = 239074
BE_PRECISION = N_H_TOTAL / (N_H_TOTAL + N_M_TOTAL)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc_dir(y, x):
    a = roc_auc_score(y, x)
    return max(a, 1.0 - a)


def topn(y, score, ns):
    order = np.argsort(-score)
    out = {}
    n_pos = int(y.sum())
    for n in ns:
        if n > len(y):
            continue
        idx = order[:n]
        c = int(y[idx].sum())
        out[f"n_{n}"] = {"correct": c, "prec": c / n,
                         "recall": c / max(n_pos, 1)}
    return out


def macro_recall_delta(y_full_in_domain: np.ndarray, override_idx_local: np.ndarray) -> float:
    """Δ macro-recall if we flip these in-domain rows from M to H predictions.

    y_full_in_domain: 0/1/2 labels for in-domain rows.
    override_idx_local: local indices (within in-domain) to flip to H.
    Returns Δ in macro-recall metric.
    """
    correct_h = int((y_full_in_domain[override_idx_local] == 2).sum())
    wrong_m = int((y_full_in_domain[override_idx_local] == 1).sum())
    # ignore wrong_l flips (overriding a true-L from M to H is also wrong but rare)
    delta_h = correct_h / N_H_TOTAL
    delta_m = -wrong_m / N_M_TOTAL
    return (delta_h + delta_m) / 3.0  # 3 classes in macro avg


def main() -> None:
    log("loading train")
    tr = pd.read_csv("data/train.csv")
    tr = add_distance_features(tr)

    y_full = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int8)
    score = tr["dgp_score"].to_numpy().astype(np.int8)

    log("building LB-best 3-way teacher (recipe + pseudo_s1 + pseudo_s7)")
    oof_r = np.load(ART / "oof_recipe_full_te.npy")
    oof_s1 = np.load(ART / "oof_recipe_pseudolabel.npy")
    oof_s7 = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    teacher = log_blend([oof_r, oof_s1, oof_s7], np.array([0.25, 0.35, 0.40]))
    # apply LB-best fixed bias [1.4324, 1.4689, 3.4008]
    bias = np.array([1.4324, 1.4689, 3.4008])
    teacher_pred = np.argmax(np.log(np.clip(teacher, 1e-9, 1.0)) + bias, axis=1).astype(np.int8)
    log(f"teacher argmax dist: {np.bincount(teacher_pred, minlength=3)}")

    # Override domain: score=6 ∩ teacher_pred=Medium
    domain = (score == 6) & (teacher_pred == 1)
    n_dom = int(domain.sum())
    in_h = (y_full == 2) & domain
    in_m = (y_full == 1) & domain
    in_l = (y_full == 0) & domain
    log(f"override domain (score=6, teacher=Medium):")
    log(f"  n={n_dom:,}  L={in_l.sum()}  M={in_m.sum()}  H={in_h.sum()}")
    log(f"  break-even precision (macro-recall): {BE_PRECISION:.4f}")
    log(f"  random-baseline precision in domain: {in_h.sum()/n_dom:.4f}")

    # We can override toward H. Wrong override on a true-L is also bad (it's
    # currently called M; flipping to H replaces M-correct with H-wrong).
    # Strict scorer: any non-H in domain is a wrong flip.
    is_h = (y_full == 2)[domain].astype(np.int8)
    is_m = (y_full == 1)[domain].astype(np.int8)
    is_l = (y_full == 0)[domain].astype(np.int8)
    log(f"  positive prevalence in domain: {is_h.mean():.4f}")

    # v2 specialist OOF, restricted to domain
    oof_v2 = np.load(ART / "oof_spec6_mh_v2.npy")[domain]
    auc_v2 = roc_auc_score(is_h, oof_v2)
    log(f"v2 specialist AUC on teacher-residual domain: {auc_v2:.5f}")

    # Top-N precision + macro-recall Δ at each cutoff
    log("v2 top-N performance (with strict M+L wrong-flip accounting):")
    log(f"  n_overrides | correct_H | wrong_M | wrong_L | prec(H) | macro_recall_delta")
    y_dom_full = y_full[domain]
    v2_topn = {}
    order = np.argsort(-oof_v2)
    for n in [10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000]:
        if n > n_dom:
            continue
        top_idx = order[:n]
        c_h = int((y_dom_full[top_idx] == 2).sum())
        w_m = int((y_dom_full[top_idx] == 1).sum())
        w_l = int((y_dom_full[top_idx] == 0).sum())
        prec = c_h / n
        delta = macro_recall_delta(y_dom_full, top_idx)
        v2_topn[f"n_{n}"] = {"correct": c_h, "wrong_m": w_m, "wrong_l": w_l,
                             "prec": prec, "macro_delta": delta}
        log(f"  {n:>5d}        | {c_h:>5d}    | {w_m:>5d}  | {w_l:>5d}  | {prec:.4f} | {delta:+.6f}")

    # What's the optimal override count under v2's ranking?
    best_n = max(v2_topn.items(), key=lambda kv: kv[1]["macro_delta"])
    log(f"v2 OPTIMAL override count under macro-recall: {best_n[0]} → "
        f"Δ = {best_n[1]['macro_delta']:+.6f}")

    # Univariate per-feature AUC on this domain
    feats = [
        "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
        "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "sm_x_rf", "tc_x_ws",
    ]
    df_dom = tr.loc[domain, feats].reset_index(drop=True).astype(np.float32)
    log(f"single-feature AUC ranking on teacher-residual domain:")
    per_feat = []
    for f in feats:
        x = df_dom[f].to_numpy()
        if np.isnan(x).any() or x.std() < 1e-12:
            continue
        au = auc_dir(is_h, x)
        d = (x[is_h == 1].mean() - x[is_h == 0].mean()) / max(x.std(), 1e-9)
        per_feat.append({"feat": f, "auc": float(au), "cohen_d": float(d)})
    per_feat.sort(key=lambda r: -r["auc"])
    for r in per_feat[:10]:
        log(f"  {r['feat']:30s} auc={r['auc']:.4f}  d={r['cohen_d']:+.3f}")

    # Add teacher-meta features and re-rank
    log("adding teacher-meta features to univariate ranking:")
    teacher_dom = teacher[domain]
    df_dom["teacher_PM"] = teacher_dom[:, 1]
    df_dom["teacher_PH"] = teacher_dom[:, 2]
    df_dom["teacher_PL"] = teacher_dom[:, 0]
    df_dom["teacher_mh_margin"] = df_dom["teacher_PM"] - df_dom["teacher_PH"]
    df_dom["teacher_mh_ratio"] = (np.log(np.clip(df_dom["teacher_PH"], 1e-9, 1)) -
                                   np.log(np.clip(df_dom["teacher_PM"], 1e-9, 1)))
    teacher_feats = ["teacher_PM", "teacher_PH", "teacher_PL",
                     "teacher_mh_margin", "teacher_mh_ratio"]
    for f in teacher_feats:
        x = df_dom[f].to_numpy()
        au = auc_dir(is_h, x)
        d = (x[is_h == 1].mean() - x[is_h == 0].mean()) / max(x.std(), 1e-9)
        log(f"  {f:30s} auc={au:.4f}  d={d:+.3f}")

    # L2-LR oracle on teacher-residual domain (best linear ceiling)
    log("L2-LR oracle on teacher-residual domain (raw + teacher-meta):")
    X = df_dom.to_numpy()
    y = is_h
    oof_lr = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(sc.transform(X[tr_idx]), y[tr_idx])
        oof_lr[va_idx] = lr.predict_proba(sc.transform(X[va_idx]))[:, 1]
    auc_lr = roc_auc_score(y, oof_lr)
    log(f"  LR oracle AUC: {auc_lr:.5f}  (vs v2: {auc_v2:.5f})")
    lr_topn = topn(y, oof_lr, [10, 25, 50, 100, 200, 500, 1000])

    # Missed-H residual on the actual domain
    log("missed-H residual: bottom-50% of true-H by v2 prob, vs found-H, vs M:")
    h_idx = np.where(is_h == 1)[0]
    h_v2 = oof_v2[h_idx]
    med = np.median(h_v2)
    miss_idx = h_idx[h_v2 < med]
    found_idx = h_idx[h_v2 >= med]
    m_idx = np.where(is_h == 0)[0]
    log(f"  missed-H={len(miss_idx)}  found-H={len(found_idx)}  M(+L)={len(m_idx)}")

    # ALL features (including teacher-meta) to see if missed-H is recoverable
    all_feats = list(df_dom.columns)
    residual_table = []
    for f in all_feats:
        x = df_dom[f].to_numpy()
        m_mean = x[m_idx].mean()
        m_std = x[m_idx].std()
        f_mean = x[found_idx].mean()
        u_mean = x[miss_idx].mean()
        z_found = (f_mean - m_mean) / max(m_std, 1e-9)
        z_missed = (u_mean - m_mean) / max(m_std, 1e-9)
        ratio = z_missed / z_found if abs(z_found) > 1e-3 else float("nan")
        residual_table.append({
            "feat": f, "z_found": float(z_found), "z_missed": float(z_missed),
            "ratio": float(ratio),
        })
    residual_table.sort(key=lambda r: -abs(r["z_found"]))
    log(f"  feat                          z(found-H)  z(missed-H)   ratio")
    for r in residual_table[:15]:
        log(f"  {r['feat']:30s}  {r['z_found']:+.3f}     {r['z_missed']:+.3f}        {r['ratio']:+.2f}")

    out = {
        "domain": {"n": n_dom, "L": int(in_l.sum()), "M": int(in_m.sum()),
                   "H": int(in_h.sum()),
                   "break_even_precision": BE_PRECISION,
                   "random_baseline_prec": float(is_h.mean())},
        "v2": {"auc": float(auc_v2), "topn": v2_topn,
               "best_n": best_n[0], "best_macro_delta": best_n[1]["macro_delta"]},
        "lr_oracle": {"auc": float(auc_lr), "topn": lr_topn},
        "per_feature_raw": per_feat,
        "teacher_feats_auc": {f: {"auc": auc_dir(is_h, df_dom[f].to_numpy())}
                               for f in teacher_feats},
        "missed_H_residual_top15": residual_table[:15],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
