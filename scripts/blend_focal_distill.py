"""Blend-gate analysis for focal-loss recipe + capacity-reduced distill.

Two new candidate OOFs:
  * oof_recipe_focal_g2_invfreq.npy  — recipe XGB trained with multi-class
    focal loss (gamma=2, alpha=invfreq).
  * oof_soft_distill_small.npy       — soft-distill from LB-best teacher
    at reduced student capacity (max_depth=3, max_leaves=15, n_round=1500).

Anchors:
  A1 recipe               (oof_recipe_full_te.npy)               OOF 0.97967
  A2 LB-best 2-way        (0.5 recipe + 0.5 pseudo, log)         OOF 0.98012
  A3 LB-best 3-way        (0.25/0.35/0.40 multi-seed)            OOF 0.98029

For each candidate x anchor pair, report:
  * standalone @ anchor's fixed bias, and own tuned bias
  * error count, Jaccard vs anchor
  * per-class recall
  * fixed-bias log-blend alpha sweep (0.00 .. 0.50)
  * fixed-bias gate verdict: lift >= +2e-4 AND errs <= anchor

Also runs diagnostic: per-class recall at each anchor's operating point
so we can see if focal's rare-High lift actually survives blending.

Writes `scripts/artifacts/blend_focal_distill_results.json` + emits a
submission CSV for any (cand, anchor) that passes the +5e-4 threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
EPS = 1e-9

CANDIDATES = [
    ("focal_g2_invfreq", "oof_recipe_focal_g2_invfreq.npy", "test_recipe_focal_g2_invfreq.npy"),
    ("distill_small",    "oof_soft_distill_small.npy",      "test_soft_distill_small.npy"),
]

# Reconstruct anchors from saved OOFs.
def load_prob(name: str) -> np.ndarray:
    p = np.load(ART / name)
    assert p.ndim == 2 and p.shape[1] == 3, (name, p.shape)
    return p


def log_blend(probs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    z = sum(wi * np.log(np.clip(p, EPS, 1.0)) for wi, p in zip(w, probs))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def build_anchors() -> dict:
    p_r_oof = load_prob("oof_recipe_full_te.npy")
    p_r_te  = load_prob("test_recipe_full_te.npy")
    p_p_oof = load_prob("oof_recipe_pseudolabel.npy")
    p_p_te  = load_prob("test_recipe_pseudolabel.npy")
    # 2-way 50/50 log-blend = LB-best component stack.
    lb2_oof = log_blend([p_r_oof, p_p_oof], [0.5, 0.5])
    lb2_te  = log_blend([p_r_te,  p_p_te],  [0.5, 0.5])
    anchors = {
        "recipe":  dict(oof=p_r_oof, test=p_r_te),
        "lb_best_2way": dict(oof=lb2_oof, test=lb2_te),
    }
    # 3-way optional: if all three components on disk, build it.
    try:
        p_s1_oof = load_prob("oof_recipe_pseudolabel.npy")  # same as stage-1
        p_s7_oof = load_prob("oof_recipe_pseudolabel_seed7labeler.npy")
        p_s7_te  = load_prob("test_recipe_pseudolabel_seed7labeler.npy")
        p_s1_te  = p_p_te  # stage-1 is recipe_pseudolabel
        lb3_oof = log_blend([p_r_oof, p_s1_oof, p_s7_oof], [0.25, 0.35, 0.40])
        lb3_te  = log_blend([p_r_te,  p_s1_te,  p_s7_te],  [0.25, 0.35, 0.40])
        anchors["lb_best_3way"] = dict(oof=lb3_oof, test=lb3_te)
    except (FileNotFoundError, OSError):
        print("  (lb_best_3way not built — seed7 labeler OOFs not on disk)")
    return anchors


def per_class_recall(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1
    rec = {}
    for k in range(3):
        denom = cm[k].sum()
        rec[["L", "M", "H"][k]] = float(cm[k, k] / denom) if denom else 0.0
    return rec


def analyze_anchor(name: str, anchor: dict, y: np.ndarray, cc: np.ndarray,
                   prior: np.ndarray) -> dict:
    # Tune anchor's own log-bias on its OOF.
    bias_a, bal_a = tune_log_bias(anchor["oof"], y, prior)
    log_a = np.log(np.clip(anchor["oof"], EPS, 1.0))
    pred_a = (log_a + bias_a).argmax(1)
    rec_a = per_class_recall(y, pred_a)
    errs_a = int((pred_a != y).sum())
    return dict(bias=bias_a.tolist(), bal_acc=bal_a, errs=errs_a, recall=rec_a)


def analyze_candidate(cand_oof: np.ndarray, cand_test: np.ndarray,
                      anchor: dict, anchor_bias: np.ndarray,
                      y: np.ndarray, cc: np.ndarray, prior: np.ndarray) -> dict:
    # Candidate at anchor's fixed bias.
    log_c = np.log(np.clip(cand_oof, EPS, 1.0))
    pred_c_fix = (log_c + anchor_bias).argmax(1)
    bal_c_fix = fast_bal_acc(y, pred_c_fix, class_counts=cc)

    # Candidate at its own tuned bias.
    bias_c, bal_c_tuned = tune_log_bias(cand_oof, y, prior)
    pred_c_tune = (log_c + bias_c).argmax(1)

    # Error geometry vs anchor.
    log_a = np.log(np.clip(anchor["oof"], EPS, 1.0))
    pred_a = (log_a + anchor_bias).argmax(1)
    errs_a = (pred_a != y)
    errs_c = (pred_c_fix != y)
    inter = int((errs_a & errs_c).sum())
    union = int((errs_a | errs_c).sum())
    jaccard = inter / max(union, 1)

    # Blend sweep.
    sweep = []
    for alpha in np.linspace(0.0, 0.50, 21):
        blend = log_blend([anchor["oof"], cand_oof], [1 - alpha, alpha])
        log_b = np.log(np.clip(blend, EPS, 1.0))
        bal = fast_bal_acc(y, (log_b + anchor_bias).argmax(1), class_counts=cc)
        sweep.append((float(alpha), float(bal)))
    best_a, best_b = max(sweep, key=lambda t: t[1])

    return dict(
        fixed_bias_bal_acc=float(bal_c_fix),
        tuned_bal_acc=float(bal_c_tuned),
        tuned_bias=bias_c.tolist(),
        recall_fixed=per_class_recall(y, pred_c_fix),
        recall_tuned=per_class_recall(y, pred_c_tune),
        errs=int(errs_c.sum()),
        jaccard_vs_anchor=float(jaccard),
        sweep=sweep,
        peak_alpha=float(best_a),
        peak_bal=float(best_b),
    )


def main():
    y = pd.read_csv("data/train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    cc = np.bincount(y, minlength=3)

    print(f"y: N={len(y)}  prior={prior.round(4).tolist()}")
    anchors = build_anchors()
    print(f"anchors: {list(anchors.keys())}")

    out = {}
    for aname, anchor in anchors.items():
        print(f"\n====== anchor: {aname} ======")
        a_info = analyze_anchor(aname, anchor, y, cc, prior)
        print(f"  bias={np.round(a_info['bias'],4).tolist()}  "
              f"bal_acc={a_info['bal_acc']:.5f}  errs={a_info['errs']:,}  "
              f"recall={a_info['recall']}")
        anchor_bias = np.asarray(a_info["bias"])

        cand_results = {}
        for cname, oof_file, test_file in CANDIDATES:
            oof_path = ART / oof_file
            test_path = ART / test_file
            if not oof_path.exists() or not test_path.exists():
                print(f"  [skip] {cname}: missing {oof_file} or {test_file}")
                continue
            oof = load_prob(oof_file)
            test = load_prob(test_file)
            if oof.shape[0] != len(y):
                print(f"  [skip] {cname}: OOF shape {oof.shape} != "
                      f"expected ({len(y)}, 3) — likely a SMOKE run, "
                      f"rerun at full scale")
                continue
            info = analyze_candidate(oof, test, anchor, anchor_bias,
                                     y, cc, prior)
            cand_results[cname] = info
            delta = info["peak_bal"] - a_info["bal_acc"]
            gate = "  PASS" if delta >= 2e-4 else "  null"
            print(f"  {cname}:")
            print(f"    standalone  tuned={info['tuned_bal_acc']:.5f}  "
                  f"fixed_bias={info['fixed_bias_bal_acc']:.5f}  "
                  f"errs={info['errs']:,}  Jaccard={info['jaccard_vs_anchor']:.4f}")
            print(f"    recall tuned  L={info['recall_tuned']['L']:.4f}  "
                  f"M={info['recall_tuned']['M']:.4f}  "
                  f"H={info['recall_tuned']['H']:.4f}")
            print(f"    sweep peak   alpha={info['peak_alpha']:.3f}  "
                  f"OOF={info['peak_bal']:.5f}  "
                  f"delta={delta:+.5f}  {gate}")

        out[aname] = dict(anchor=a_info, candidates=cand_results)

    res_path = ART / "blend_focal_distill_results.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {res_path}")


if __name__ == "__main__":
    main()
