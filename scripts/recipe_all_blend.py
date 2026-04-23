"""Greedy forward log-blend over the recipe-level OOF bank.

Candidates (listed in priority order; missing files are skipped):
    recipe_full_te        (XGB on 443 features — LB 0.97939)
    recipe_lgbm           (LGBM on same 443 features)
    recipe_pseudolabel    (XGB retrained on train + pseudo-labeled test)
    recipe_catboost       (CatBoost on same 443 features — already null on its own)

Anchor bias is recipe_full_te's tuned bias (LB-calibrated). Greedy proceeds:
    1. Start from recipe_full_te (best single component).
    2. For each remaining candidate, sweep α ∈ 0.025..0.5 over log-blend
       α × cand + (1-α) × current_blend. Keep α that maximises tuned-bias
       bal_acc (DIAGNOSTIC) AND fixed-bias bal_acc.
    3. Select the candidate with the largest FIXED-BIAS lift ≥ +1e-4.
    4. Emit a submission at the final blend if Δ vs recipe_full_te ≥ +5e-4.

The fixed-bias gate is the only one that drives submissions (binhigh
lesson — tuned-bias retune manufactures fake OOF lift that regresses LB).
Tuned-bias diagnostic is reported only for understanding whether a null
is bias-driven (fixable) or genuine (not fixable).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)

TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

CANDIDATES = [
    ("recipe_full_te",            "oof_recipe_full_te.npy",              "test_recipe_full_te.npy"),
    ("recipe_allpairs",           "oof_recipe_allpairs.npy",             "test_recipe_allpairs.npy"),
    ("recipe_lgbm",               "oof_recipe_lgbm.npy",                 "test_recipe_lgbm.npy"),
    ("recipe_pseudolabel",        "oof_recipe_pseudolabel.npy",          "test_recipe_pseudolabel.npy"),
    ("recipe_pseudolabel_stage2", "oof_recipe_pseudolabel_stage2.npy",   "test_recipe_pseudolabel_stage2.npy"),
    ("recipe_catboost",           "oof_recipe_catboost.npy",             "test_recipe_catboost.npy"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_blend_list(probs_list, weights) -> np.ndarray:
    logs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        logs = logs + w * np.log(np.clip(p, 1e-9, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def fixed_bias_ba(probs, y, bias) -> float:
    lp = np.log(np.clip(probs, 1e-9, 1.0))
    return fast_bal_acc(y.astype(np.int32), (lp + bias).argmax(1))


def main():
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias = np.array(recipe_res["log_bias"])
    log(f"anchor bias = {bias.round(4).tolist()}  "
        f"(recipe_full_te tuned OOF = {recipe_res['tuned_log_bias_bal_acc']:.5f})")

    comps = {}
    for name, oof_f, test_f in CANDIDATES:
        op, tp = ART / oof_f, ART / test_f
        if not op.exists() or not tp.exists():
            log(f"  skip {name}: missing artefact")
            continue
        comps[name] = dict(oof=np.load(op), test=np.load(tp))
    log(f"loaded {len(comps)} components: {sorted(comps.keys())}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # Standalone at fixed bias + tuned bias.
    log("\n--- standalones ---")
    standalone = {}
    for n, v in comps.items():
        fb = fixed_bias_ba(v["oof"], y, bias)
        _, tb = tune_log_bias(v["oof"], y, prior)
        err = (np.log(np.clip(v["oof"], 1e-9, 1.0)) + bias).argmax(1) != y
        standalone[n] = dict(fixed=fb, tuned=tb, errors=int(err.sum()))
        log(f"  {n:22s}  fixed@anchor={fb:.5f}  tuned={tb:.5f}  "
            f"err_count={err.sum()}")

    # Pairwise Jaccard vs recipe_full_te for diagnostic.
    log("\n--- Jaccard vs recipe_full_te ---")
    if "recipe_full_te" in comps:
        ref_err = ((np.log(np.clip(comps["recipe_full_te"]["oof"], 1e-9, 1.0))
                    + bias).argmax(1) != y)
        for n in comps:
            if n == "recipe_full_te":
                continue
            e = ((np.log(np.clip(comps[n]["oof"], 1e-9, 1.0))
                  + bias).argmax(1) != y)
            j = (e & ref_err).sum() / max(1, (e | ref_err).sum())
            log(f"  {n:22s}  Jaccard={j:.4f}  errs_{n}={e.sum()} "
                f"vs errs_recipe={ref_err.sum()}")

    # Greedy forward on fixed-bias scoring.
    anchor_ba = standalone["recipe_full_te"]["fixed"]
    current_names = ["recipe_full_te"]
    current_weights = [1.0]
    current_blend = comps["recipe_full_te"]["oof"].copy()
    current_test = comps["recipe_full_te"]["test"].copy()
    current_ba = anchor_ba
    log(f"\n--- greedy forward (anchor={anchor_ba:.5f}) ---")

    alpha_grid = np.array([0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25,
                           0.30, 0.35, 0.40, 0.45, 0.50])

    while True:
        best_cand, best_alpha, best_delta = None, 0.0, 0.0
        best_tuned_diag = 0.0
        for cand in comps:
            if cand in current_names:
                continue
            for a in alpha_grid:
                b = log_blend_list([comps[cand]["oof"], current_blend], [a, 1 - a])
                fb = fixed_bias_ba(b, y, bias)
                if fb > current_ba + best_delta:
                    best_delta = fb - current_ba
                    best_cand = cand
                    best_alpha = a
                    _, best_tuned_diag = tune_log_bias(b, y, prior)
        if best_cand is None or best_delta < 1e-5:
            log("  no candidate improves by >= 1e-5; stop.")
            break
        new_w = [(1 - best_alpha) * w for w in current_weights] + [best_alpha]
        current_names = current_names + [best_cand]
        current_weights = new_w
        current_blend = log_blend_list(
            [comps[n]["oof"] for n in current_names], current_weights
        )
        current_test = log_blend_list(
            [comps[n]["test"] for n in current_names], current_weights
        )
        current_ba = fixed_bias_ba(current_blend, y, bias)
        log(f"  + {best_cand:22s}  α={best_alpha:.3f}  fixed={current_ba:.5f}  "
            f"Δ={best_delta:+.5f}  (tuned-diag={best_tuned_diag:.5f})")

    log(f"\nfinal greedy blend: {current_names}")
    for n, w in zip(current_names, current_weights):
        log(f"  {w:.4f}  {n}")
    log(f"fixed-bias OOF = {current_ba:.5f}  "
        f"Δ vs recipe_full_te = {current_ba - anchor_ba:+.5f}")

    # Emit submission if Δ ≥ +5e-4 (standard LB-probe gate).
    action = "no_submission"
    sub_path = None
    if current_ba - anchor_ba >= 5e-4:
        preds = (np.log(np.clip(current_test, 1e-9, 1.0)) + bias).argmax(1)
        sub_path = SUB / f"submission_recipe_greedy_{'_'.join(current_names[1:])}.csv"
        pd.DataFrame({
            "id": te["id"], TARGET: [IDX2CLS[i] for i in preds]
        }).to_csv(sub_path, index=False)
        log(f"wrote {sub_path}")
        action = "submission_ready"
    elif current_ba - anchor_ba >= 1e-4:
        action = "submission_borderline"
    log(f"action: {action}")

    # Also compute best pairwise for each candidate (simpler sweep record).
    log("\n--- pairwise sweeps (diagnostic) ---")
    pairwise = {}
    for cand in comps:
        if cand == "recipe_full_te":
            continue
        best_a, best_ba = 0.0, anchor_ba
        for a in np.linspace(0.0, 0.95, 20):
            b = log_blend_list(
                [comps["recipe_full_te"]["oof"], comps[cand]["oof"]],
                [1 - a, a]
            )
            ba = fixed_bias_ba(b, y, bias)
            if ba > best_ba:
                best_ba = ba
                best_a = a
        pairwise[cand] = dict(alpha=float(best_a), oof=float(best_ba),
                              delta=float(best_ba - anchor_ba))
        log(f"  recipe × {cand:22s}  α={best_a:.3f}  "
            f"OOF={best_ba:.5f}  Δ={best_ba - anchor_ba:+.5f}")

    # Confusion matrix at final blend.
    cm = confusion_matrix(
        y,
        (np.log(np.clip(current_blend, 1e-9, 1.0)) + bias).argmax(1),
    )
    log(f"\nfinal blend confusion matrix:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    out = dict(
        anchor_bias=bias.tolist(),
        anchor_oof=anchor_ba,
        standalone=standalone,
        pairwise=pairwise,
        greedy=dict(
            components=current_names,
            weights=current_weights,
            oof_fixed_bias=current_ba,
            delta_vs_anchor=current_ba - anchor_ba,
        ),
        action=action,
        submission=str(sub_path) if sub_path else None,
    )
    with open(ART / "recipe_all_blend_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {ART}/recipe_all_blend_results.json")


if __name__ == "__main__":
    main()
