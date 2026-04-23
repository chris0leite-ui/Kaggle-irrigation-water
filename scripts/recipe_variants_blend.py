"""Blend analysis for recipe variants (OTE multi-strength + DART).

Reads the variant OOFs on disk and evaluates blend potential vs:
    1. recipe_full_te (LB 0.97939) — the anchor
    2. greedy 2-way blend: 0.5 × recipe_full_te + 0.5 × recipe_pseudolabel
       (current LB best 0.97998)

For each variant, reports:
    - standalone tuned bal_acc (own log-bias) and fixed-bias at anchor
    - error count + Jaccard vs anchor + vs LB-best
    - pairwise α-sweep vs anchor (fixed bias)
    - pairwise α-sweep vs LB-best (fixed bias)

Then runs greedy forward-selection adding variants on top of LB-best.

Fixed-bias evaluation is the trustworthy signal (binhigh lesson:
tuned-bias retune manufactures OOF lift that blows up LB gap). The
LB-probe gate is Δ ≥ +5e-4 vs the LB-best baseline.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)

TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS2IDX.items()}

VARIANT_SUFFIXES = ["_a01", "_a10", "_dart"]
ALPHAS = np.array([0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25,
                   0.30, 0.35, 0.40, 0.45, 0.50])


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend(probs_list, weights, eps=1e-9):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    logs = np.zeros_like(probs_list[0], dtype=np.float64)
    for wi, p in zip(w, probs_list):
        logs += wi * np.log(np.clip(p, eps, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def fb_ba(probs, y, bias, eps=1e-9):
    lp = np.log(np.clip(probs, eps, 1.0))
    return fast_bal_acc(y.astype(np.int32), (lp + bias).argmax(1))


def err_mask(probs, y, bias, eps=1e-9):
    lp = np.log(np.clip(probs, eps, 1.0))
    return (lp + bias).argmax(1) != y


def jaccard(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / max(1, union)


def main() -> None:
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    anchor_bias = np.array(recipe_res["log_bias"])
    log(f"anchor: recipe_full_te tuned OOF = "
        f"{recipe_res['tuned_log_bias_bal_acc']:.5f}  "
        f"bias={anchor_bias.round(4).tolist()}")

    y = pd.read_csv("data/train.csv")[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    oof_anchor = np.load(ART / "oof_recipe_full_te.npy")
    test_anchor = np.load(ART / "test_recipe_full_te.npy")
    anchor_ba = fb_ba(oof_anchor, y, anchor_bias)
    anchor_err = err_mask(oof_anchor, y, anchor_bias)
    log(f"anchor fixed-bias OOF = {anchor_ba:.5f}  errors={int(anchor_err.sum())}")

    lb_pseudo_path = ART / "oof_recipe_pseudolabel.npy"
    has_lb_best = lb_pseudo_path.exists()
    if has_lb_best:
        oof_pseudo = np.load(lb_pseudo_path)
        test_pseudo = np.load(ART / "test_recipe_pseudolabel.npy")
        oof_lbbest = log_blend([oof_anchor, oof_pseudo], [0.5, 0.5])
        test_lbbest = log_blend([test_anchor, test_pseudo], [0.5, 0.5])
        lbbest_ba = fb_ba(oof_lbbest, y, anchor_bias)
        lbbest_err = err_mask(oof_lbbest, y, anchor_bias)
        log(f"LB-best (2-way blend) fixed-bias OOF = {lbbest_ba:.5f}  "
            f"errors={int(lbbest_err.sum())}")
    else:
        log("recipe_pseudolabel not on disk — only anchor baseline evaluated")
        oof_lbbest = test_lbbest = None
        lbbest_ba = lbbest_err = None

    out = dict(
        anchor=dict(oof_fixed=float(anchor_ba), bias=anchor_bias.tolist(),
                    errors=int(anchor_err.sum())),
        variants={},
    )
    if has_lb_best:
        out["lb_best_2way"] = dict(
            oof_fixed=float(lbbest_ba), errors=int(lbbest_err.sum()),
        )

    # --- per-variant diagnostics + pairwise sweeps -----------------------
    loaded = {}
    for suffix in VARIANT_SUFFIXES:
        name = f"recipe_full_te{suffix}"
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not (oof_p.exists() and test_p.exists()):
            log(f"[skip] {name}: artefact missing")
            continue
        oof_v = np.load(oof_p)
        test_v = np.load(test_p)
        if oof_v.shape != oof_anchor.shape or test_v.shape != test_anchor.shape:
            log(f"[skip] {name}: shape mismatch "
                f"(oof {oof_v.shape} vs {oof_anchor.shape}; "
                f"test {test_v.shape} vs {test_anchor.shape})")
            continue
        loaded[name] = (oof_v, test_v)

        fixed = fb_ba(oof_v, y, anchor_bias)
        _, tuned = tune_log_bias(oof_v, y, prior)
        err_v = err_mask(oof_v, y, anchor_bias)
        j_anchor = jaccard(err_v, anchor_err)
        j_lb = jaccard(err_v, lbbest_err) if has_lb_best else None

        log(f"\n=== {name} ===")
        log(f"  standalone fixed@anchor={fixed:.5f}  tuned={tuned:.5f}  "
            f"errors={int(err_v.sum())}  J(anchor)={j_anchor:.4f}"
            + (f"  J(lb_best)={j_lb:.4f}" if j_lb is not None else ""))

        # Pairwise vs anchor.
        peak_a = (0.0, anchor_ba)
        for a in ALPHAS:
            b = log_blend([oof_v, oof_anchor], [a, 1 - a])
            s = fb_ba(b, y, anchor_bias)
            if s > peak_a[1]:
                peak_a = (float(a), float(s))
        log(f"  vs anchor:  peak α={peak_a[0]:.3f}  OOF={peak_a[1]:.5f}  "
            f"Δ={peak_a[1]-anchor_ba:+.5f}")

        # Pairwise vs LB-best.
        peak_lb = None
        sub_path = None
        if has_lb_best:
            peak = (0.0, lbbest_ba)
            for a in ALPHAS:
                b = log_blend([oof_v, oof_lbbest], [a, 1 - a])
                s = fb_ba(b, y, anchor_bias)
                if s > peak[1]:
                    peak = (float(a), float(s))
            peak_lb = peak
            log(f"  vs lb_best: peak α={peak[0]:.3f}  OOF={peak[1]:.5f}  "
                f"Δ={peak[1]-lbbest_ba:+.5f}")

            if peak[1] - lbbest_ba >= 5e-4:
                a = peak[0]
                blend_test = log_blend([test_v, test_lbbest], [a, 1 - a])
                preds = (np.log(np.clip(blend_test, 1e-9, 1.0))
                         + anchor_bias).argmax(1)
                test_ids = pd.read_csv("data/test.csv")["id"].values
                sub = pd.DataFrame({
                    "id": test_ids,
                    TARGET: [IDX2CLS[i] for i in preds],
                })
                a_tag = f"{a:.3f}".replace(".", "")
                sub_path = (SUB /
                            f"submission_lbbest_x_{name}_a{a_tag}.csv")
                sub.to_csv(sub_path, index=False)
                log(f"  → wrote {sub_path}")

        out["variants"][name] = dict(
            standalone_fixed=float(fixed),
            standalone_tuned=float(tuned),
            errors=int(err_v.sum()),
            jaccard_vs_anchor=float(j_anchor),
            jaccard_vs_lbbest=float(j_lb) if j_lb is not None else None,
            pairwise_vs_anchor=dict(alpha=peak_a[0], oof=peak_a[1],
                                    delta=peak_a[1] - anchor_ba),
            pairwise_vs_lbbest=(dict(alpha=peak_lb[0], oof=peak_lb[1],
                                     delta=peak_lb[1] - lbbest_ba)
                                if peak_lb else None),
            submission=str(sub_path) if sub_path else None,
        )

    # --- greedy forward-selection from LB-best (or anchor) ---------------
    if has_lb_best:
        baseline_oof = oof_lbbest
        baseline_test = test_lbbest
        baseline_ba = lbbest_ba
        baseline_name = "lb_best_2way"
    else:
        baseline_oof = oof_anchor
        baseline_test = test_anchor
        baseline_ba = anchor_ba
        baseline_name = "recipe_full_te"
    current_oof = baseline_oof.copy()
    current_test = baseline_test.copy()
    current_ba = baseline_ba
    picked = []
    log(f"\n--- greedy forward from {baseline_name} "
        f"(fixed-bias OOF {baseline_ba:.5f}) ---")
    while True:
        best = None
        for name, (o, _) in loaded.items():
            if name in picked:
                continue
            for a in ALPHAS:
                b = log_blend([o, current_oof], [a, 1 - a])
                s = fb_ba(b, y, anchor_bias)
                if best is None or s > best[2]:
                    best = (name, float(a), float(s))
        if best is None or best[2] - current_ba < 1e-5:
            log("  no candidate improves by >= 1e-5; stop.")
            break
        name, a, s = best
        cand_o, cand_t = loaded[name]
        current_oof = log_blend([cand_o, current_oof], [a, 1 - a])
        current_test = log_blend([cand_t, current_test], [a, 1 - a])
        picked.append(name)
        current_ba = s
        log(f"  + {name:30s}  α={a:.3f}  fixed={current_ba:.5f}  "
            f"Δ={current_ba - baseline_ba:+.5f}")

    log(f"\nfinal greedy: baseline={baseline_name} + {picked}")
    log(f"fixed-bias OOF = {current_ba:.5f}  "
        f"Δ vs {baseline_name} = {current_ba - baseline_ba:+.5f}")

    greedy_sub = None
    if current_ba - baseline_ba >= 5e-4 and picked:
        preds = (np.log(np.clip(current_test, 1e-9, 1.0))
                 + anchor_bias).argmax(1)
        test_ids = pd.read_csv("data/test.csv")["id"].values
        sub = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in preds],
        })
        tag = "_".join(n.replace("recipe_full_te", "rft") for n in picked)
        greedy_sub = SUB / f"submission_recipe_variants_greedy_{tag}.csv"
        sub.to_csv(greedy_sub, index=False)
        log(f"→ wrote greedy submission {greedy_sub}")

    out["greedy"] = dict(
        baseline=baseline_name,
        baseline_oof=float(baseline_ba),
        picked=picked,
        oof=float(current_ba),
        delta_vs_baseline=float(current_ba - baseline_ba),
        submission=str(greedy_sub) if greedy_sub else None,
    )

    res_path = ART / "recipe_variants_blend_results.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {res_path}")


if __name__ == "__main__":
    main()
