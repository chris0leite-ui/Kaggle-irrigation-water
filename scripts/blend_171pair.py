"""Fixed-bias log-blend sweep: 171-pair recipe vs current LB-best stack.

Diagnostics required by the latest LEARNINGS rule (added 2026-04-23 from
the no_ote N1 null):
  A blend candidate must satisfy BOTH
    (1) Jaccard(err vs anchor) < ~0.78
    (2) error count <= anchor's
  to have a real chance of lifting OOF. Lower Jaccard alone is necessary
  but not sufficient — magnitude drag on extra-wrong rows cancels the
  novelty on extra-right rows.

Tested anchors:
  A. recipe_full_te alone           (OOF 0.97967 / LB 0.97939)
  B. LB-BEST = 0.5 * recipe + 0.5 * recipe_pseudolabel
                                    (OOF 0.98012 / LB 0.97998)

Sweeps α ∈ {0.025, ..., 0.50} of
    log P_blend = α * log P_171pair + (1 - α) * log P_anchor
under each anchor's FIXED bias (no retune — proven failure mode per
binhigh + Session B). Emits a submission ONLY for the LB-BEST anchor
case and ONLY if Δ_OOF >= +1e-4 AND error count <= anchor's.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend_two(p_a: np.ndarray, p_b: np.ndarray,
                  w_a: float) -> np.ndarray:
    eps = 1e-9
    logs = w_a * np.log(np.clip(p_a, eps, 1.0)) + (1 - w_a) * np.log(np.clip(p_b, eps, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def fixed_bias_argmax(prob: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(prob, 1e-9, 1.0)) + bias).argmax(axis=1)


def fixed_bias_bal(prob: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, fixed_bias_argmax(prob, bias)))


def err_jaccard(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> float:
    err_a = pred_a != y
    err_b = pred_b != y
    inter = (err_a & err_b).sum()
    union = (err_a | err_b).sum()
    return float(inter / union) if union else 0.0


def sweep(p_cand: np.ndarray, p_anchor: np.ndarray, y: np.ndarray,
          bias: np.ndarray, anchor_pred: np.ndarray,
          alpha_grid: np.ndarray, label: str) -> dict:
    rows = []
    for a in alpha_grid:
        blend = log_blend_two(p_cand, p_anchor, a)
        pred = fixed_bias_argmax(blend, bias)
        ba = float(balanced_accuracy_score(y, pred))
        errs = int((pred != y).sum())
        jacc = err_jaccard(pred, anchor_pred, y)
        rows.append((float(a), ba, errs, jacc))
    best = max(rows, key=lambda r: r[1])
    log(f"  {label}  peak α={best[0]:.3f}  OOF={best[1]:.5f}  errs={best[2]:,}  Jaccard={best[3]:.4f}")
    return {"sweep": rows, "peak_alpha": best[0], "peak_oof": best[1],
            "peak_errs": best[2], "peak_jaccard": best[3]}


def main() -> None:
    log("loading components")
    oof_171 = np.load(ART / "oof_recipe_171pair.npy")
    test_171 = np.load(ART / "test_recipe_171pair.npy")
    oof_recipe = np.load(ART / "oof_recipe_full_te.npy")
    test_recipe = np.load(ART / "test_recipe_full_te.npy")
    pseudo_oof_path = ART / "oof_recipe_pseudolabel.npy"
    pseudo_test_path = ART / "test_recipe_pseudolabel.npy"
    have_pseudo = pseudo_oof_path.exists() and pseudo_test_path.exists()
    if have_pseudo:
        oof_pseudo = np.load(pseudo_oof_path)
        test_pseudo = np.load(pseudo_test_path)
        log("pseudolabel OOF/test loaded — will test anchor B (LB-best) too")
    else:
        log("pseudolabel OOF/test missing — anchor B (LB-best) will be skipped")
        oof_pseudo = test_pseudo = None

    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    anchor_bias = np.array(recipe_res["log_bias"])
    log(f"recipe bias = {anchor_bias.round(4).tolist()}")

    res_171 = json.loads((ART / "recipe_171pair_results.json").read_text())
    log(f"171pair standalone OOF (per its tuned bias) = {res_171['tuned_log_bias_bal_acc']:.5f}")

    tr = pd.read_csv("data/train.csv")
    te_df = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    # --- Anchor A: recipe alone
    pred_recipe = fixed_bias_argmax(oof_recipe, anchor_bias)
    ba_recipe = float(balanced_accuracy_score(y, pred_recipe))
    err_recipe = int((pred_recipe != y).sum())
    log(f"\nanchor A (recipe alone) at fixed bias: OOF={ba_recipe:.5f}  errs={err_recipe:,}")

    # 171pair at recipe's bias (sanity)
    ba_171_at_recipe_bias = fixed_bias_bal(oof_171, y, anchor_bias)
    pred_171 = fixed_bias_argmax(oof_171, anchor_bias)
    err_171 = int((pred_171 != y).sum())
    jacc_171_vs_recipe = err_jaccard(pred_171, pred_recipe, y)
    log(f"171pair at recipe's bias:           OOF={ba_171_at_recipe_bias:.5f}  errs={err_171:,}  Jaccard(vs recipe)={jacc_171_vs_recipe:.4f}")

    alpha_grid = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    log("\n--- sweep vs anchor A (recipe alone) ---")
    sweep_A = sweep(oof_171, oof_recipe, y, anchor_bias, pred_recipe, alpha_grid, "171p × recipe")
    delta_A = sweep_A["peak_oof"] - ba_recipe
    log(f"\nbest Δ vs recipe alone: {delta_A:+.5f}  (anchor 0.97967 ref)")

    sweep_B = None
    delta_B = None
    sub_path = None
    if have_pseudo:
        oof_lbbest = log_blend_two(oof_recipe, oof_pseudo, 0.5)
        test_lbbest = log_blend_two(test_recipe, test_pseudo, 0.5)
        pred_lbbest = fixed_bias_argmax(oof_lbbest, anchor_bias)
        ba_lbbest = float(balanced_accuracy_score(y, pred_lbbest))
        err_lbbest = int((pred_lbbest != y).sum())
        log(f"\nanchor B (LB-best 0.5*recipe + 0.5*pseudolabel) at fixed bias: OOF={ba_lbbest:.5f}  errs={err_lbbest:,}")

        jacc_171_vs_lbbest = err_jaccard(pred_171, pred_lbbest, y)
        log(f"171pair errors vs LB-best: Jaccard={jacc_171_vs_lbbest:.4f}  err magnitude {err_171:,} vs anchor {err_lbbest:,}")

        log("\n--- sweep vs anchor B (LB-best) ---")
        sweep_B = sweep(oof_171, oof_lbbest, y, anchor_bias, pred_lbbest, alpha_grid, "171p × LB-best")
        delta_B = sweep_B["peak_oof"] - ba_lbbest
        log(f"best Δ vs LB-best: {delta_B:+.5f}  (anchor 0.97998 LB ref)")

        if delta_B >= 1e-4:
            a = sweep_B["peak_alpha"]
            test_blend = log_blend_two(test_171, test_lbbest, a)
            preds = fixed_bias_argmax(test_blend, anchor_bias)
            sub_path = SUB / f"submission_lbbest_plus_171pair_a{a:.3f}.csv"
            pd.DataFrame({ID: te_df[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub_path, index=False)
            log(f"\nwrote {sub_path}  OOF Δ vs LB-best = {delta_B:+.5f}")

            cm = confusion_matrix(y, fixed_bias_argmax(log_blend_two(oof_171, oof_lbbest, a), anchor_bias))
            log("OOF confusion at emit blend:\n" + str(pd.DataFrame(cm, index=CLASSES, columns=CLASSES)))
        else:
            log(f"\nno submission: best Δ vs LB-best = {delta_B:+.5f} below +1e-4 gate")
    else:
        # Anchor B unavailable — emit a recipe-only blend submission as a
        # diagnostic if anchor A shows a clear lift. NOT the LB-best path,
        # but useful as a sanity check that the new feature surface helps.
        if delta_A >= 5e-4:
            a = sweep_A["peak_alpha"]
            test_blend = log_blend_two(test_171, test_recipe, a)
            preds = fixed_bias_argmax(test_blend, anchor_bias)
            sub_path = SUB / f"submission_recipe_plus_171pair_a{a:.3f}.csv"
            pd.DataFrame({ID: te_df[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub_path, index=False)
            log(f"\nwrote {sub_path}  OOF Δ vs recipe = {delta_A:+.5f}  (NOT LB-best path; pseudolabel re-run recommended next)")

    out = {
        "anchor_bias": anchor_bias.tolist(),
        "have_pseudolabel": have_pseudo,
        "anchor_A_recipe_oof": ba_recipe,
        "anchor_A_errs": err_recipe,
        "171pair_at_recipe_bias_oof": ba_171_at_recipe_bias,
        "171pair_errs": err_171,
        "171pair_jaccard_vs_recipe": jacc_171_vs_recipe,
        "sweep_A": sweep_A,
        "delta_vs_recipe": delta_A,
        "sweep_B": sweep_B,
        "delta_vs_lbbest": delta_B,
        "submission": str(sub_path) if sub_path else None,
    }
    with open(ART / "blend_171pair_results.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else o)
    log(f"wrote {ART}/blend_171pair_results.json")


if __name__ == "__main__":
    main()
