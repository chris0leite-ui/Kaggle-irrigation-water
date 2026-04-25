"""Blend-gate analysis: distill-tiny + distill-small vs anchors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
EPS = 1e-9

CANDIDATES = [
    ("distill_small", "oof_soft_distill_small.npy", "test_soft_distill_small.npy"),
    ("distill_tiny",  "oof_soft_distill_tiny.npy",  "test_soft_distill_tiny.npy"),
    ("recipe_smote2x", "oof_recipe_smote2x.npy", "test_recipe_smote2x.npy"),
]


def log_blend(probs, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    z = sum(wi * np.log(np.clip(p, EPS, 1.0)) for wi, p in zip(w, probs))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def load_prob(name):
    p = np.load(ART / name).astype(np.float32)
    p = p / np.clip(p.sum(1, keepdims=True), 1e-9, None)
    return p


def build_anchors():
    r_oof = load_prob("oof_recipe_full_te.npy")
    r_te = load_prob("test_recipe_full_te.npy")
    p_oof = load_prob("oof_recipe_pseudolabel.npy")
    p_te = load_prob("test_recipe_pseudolabel.npy")
    lb2_oof = log_blend([r_oof, p_oof], [0.5, 0.5])
    lb2_te = log_blend([r_te, p_te], [0.5, 0.5])
    s7_oof = load_prob("oof_recipe_pseudolabel_seed7labeler.npy")
    s7_te = load_prob("test_recipe_pseudolabel_seed7labeler.npy")
    lb3_oof = log_blend([r_oof, p_oof, s7_oof], [0.25, 0.35, 0.40])
    lb3_te = log_blend([r_te, p_te, s7_te], [0.25, 0.35, 0.40])
    rm_oof = load_prob("oof_realmlp.npy")
    rm_te = load_prob("test_realmlp.npy")
    nr_oof = load_prob("oof_xgb_nonrule.npy")
    nr_te = load_prob("test_xgb_nonrule.npy")
    stack_oof = log_blend([lb3_oof, rm_oof, nr_oof], [0.725, 0.200, 0.075])
    stack_te = log_blend([lb3_te, rm_te, nr_te], [0.725, 0.200, 0.075])
    return {
        "recipe": dict(oof=r_oof, test=r_te),
        "lb2_best": dict(oof=lb2_oof, test=lb2_te),
        "lb3_best": dict(oof=lb3_oof, test=lb3_te),
        "lb_best_stack": dict(oof=stack_oof, test=stack_te),
    }


def per_class_recall(y, pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1
    return {k: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
            for i, k in enumerate(["L", "M", "H"])}


def analyze(cand_oof, anchor_oof, anchor_bias, y, cc, prior):
    log_c = np.log(np.clip(cand_oof, EPS, 1.0))
    pred_c_fix = (log_c + anchor_bias).argmax(1)
    bal_c_fix = fast_bal_acc(y, pred_c_fix, class_counts=cc)
    bias_c, bal_c_tuned = tune_log_bias(cand_oof, y, prior)
    pred_c_tune = (log_c + bias_c).argmax(1)

    log_a = np.log(np.clip(anchor_oof, EPS, 1.0))
    pred_a = (log_a + anchor_bias).argmax(1)
    errs_a = pred_a != y
    errs_c = pred_c_fix != y
    inter = int((errs_a & errs_c).sum())
    union = int((errs_a | errs_c).sum())
    jaccard = inter / max(union, 1)

    sweep = []
    for alpha in np.linspace(0.0, 0.40, 17):
        blend = log_blend([anchor_oof, cand_oof], [1 - alpha, alpha])
        log_b = np.log(np.clip(blend, EPS, 1.0))
        bal = fast_bal_acc(y, (log_b + anchor_bias).argmax(1), class_counts=cc)
        sweep.append((float(alpha), float(bal)))
    best_a, best_b = max(sweep, key=lambda t: t[1])

    return dict(
        fixed_bias_bal_acc=float(bal_c_fix),
        tuned_bal_acc=float(bal_c_tuned),
        tuned_bias=bias_c.tolist(),
        recall_tuned=per_class_recall(y, pred_c_tune),
        errs=int(errs_c.sum()),
        jaccard=float(jaccard),
        sweep=sweep,
        peak_alpha=float(best_a),
        peak_bal=float(best_b),
    )


def main():
    y = pd.read_csv("data/train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    cc = np.bincount(y, minlength=3)

    anchors = build_anchors()
    print(f"anchors: {list(anchors.keys())}")

    out = {}
    for aname, anchor in anchors.items():
        print(f"\n====== anchor: {aname} ======")
        bias_a, bal_a = tune_log_bias(anchor["oof"], y, prior)
        pred_a = (np.log(np.clip(anchor["oof"], EPS, 1.0)) + bias_a).argmax(1)
        rec_a = per_class_recall(y, pred_a)
        errs_a = int((pred_a != y).sum())
        print(f"  anchor: bias={np.round(bias_a, 4).tolist()}  "
              f"bal={bal_a:.5f}  errs={errs_a:,}  recall={rec_a}")

        cand_out = {}
        for cname, oof_file, test_file in CANDIDATES:
            if not (ART / oof_file).exists():
                print(f"  [skip] {cname}: {oof_file} missing")
                continue
            cand_oof = load_prob(oof_file)
            info = analyze(cand_oof, anchor["oof"], bias_a, y, cc, prior)
            delta = info["peak_bal"] - bal_a
            gate = "PASS" if delta >= 2e-4 else "null"
            print(f"  {cname}:")
            print(f"    standalone tuned={info['tuned_bal_acc']:.5f}  "
                  f"errs={info['errs']:,}  Jaccard={info['jaccard']:.4f}")
            print(f"    recall tuned  L={info['recall_tuned']['L']:.4f}  "
                  f"M={info['recall_tuned']['M']:.4f}  "
                  f"H={info['recall_tuned']['H']:.4f}")
            print(f"    sweep peak   alpha={info['peak_alpha']:.3f}  "
                  f"OOF={info['peak_bal']:.5f}  delta={delta:+.5f}  {gate}")
            cand_out[cname] = info

        out[aname] = dict(anchor_bal=bal_a, anchor_bias=bias_a.tolist(),
                          anchor_errs=errs_a, candidates=cand_out)

    (ART / "blend_distill_variants_results.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\nwrote scripts/artifacts/blend_distill_variants_results.json")


if __name__ == "__main__":
    main()
