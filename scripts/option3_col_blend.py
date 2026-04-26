"""Option 3 — per-class column-selective blend.

Both 3-way OTE and NN-dist candidates produced standalone OOFs that
TIED vanilla recipe AND had FEWER total errors than the LB-best
4-stack (NN-dist 9189 errs vs anchor 9415, ratio 0.976). The full-
vector log-blend gate failed because per-class recall trade hurts
High by ~0.006 — wrong direction under macro-recall.

Hypothesis: the candidate's Medium-column probabilities carry
genuine signal (per-class recall Δ M=+0.002) while its Low/High
columns are slightly worse. Surgically blending ONLY the Medium
column extracts the gain without the High loss.

Mechanism:
  P_blend[:,0] = anchor[:,0]                                   # Low: anchor only
  P_blend[:,1] = (1-α)*anchor[:,1] + α*candidate[:,1]          # Medium: blend
  P_blend[:,2] = anchor[:,2]                                   # High: anchor only
  P_blend = renormalize(P_blend)

Then apply fixed recipe bias [1.4324, 1.4689, 3.4008] for argmax.

Per-class recall guardrail: each class ≥ anchor_floor − 5e-4.
Emit submission ONLY if peak Δ ≥ +1e-4 OOF (looser than +2e-4
because column-selective is structurally novel — closes the lever
either way).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, SUB, build_lbbest_stack, iso_cal, load_y, log, normed,
)

CANDIDATES = ["recipe_full_te_3way", "recipe_full_te_nndist"]
GUARDRAIL = 5e-4
EMIT_DELTA = 1e-4
ALPHA_GRID = np.arange(0.025, 0.85, 0.025)


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return np.diag(cm).astype(float) / cm.sum(axis=1).clip(1)


def col_blend(anchor: np.ndarray, cand: np.ndarray, col: int,
              alpha: float) -> np.ndarray:
    """Replace anchor's column `col` with (1-α)*anchor[:,col] + α*cand[:,col],
    then renormalize the row to sum to 1. The other columns scale
    proportionally so the rows still sum to 1.
    """
    out = anchor.copy()
    out[:, col] = (1 - alpha) * anchor[:, col] + alpha * cand[:, col]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)


def evaluate(p, y, anchor_pcr):
    pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    bal = balanced_accuracy_score(y, pred)
    pcr = per_class_recall(y, pred)
    errs = int((pred != y).sum())
    pcr_pass = all(pcr[c] >= anchor_pcr[c] - GUARDRAIL for c in range(3))
    return dict(bal=float(bal), pcr=pcr.tolist(),
                errs=errs, pcr_pass=bool(pcr_pass))


def sweep_one(name, anchor_o, anchor_t, cand_o, cand_t, y, anchor_pcr, anchor_bal):
    """For one candidate, sweep α on each of the 3 column choices."""
    res = {"name": name, "by_col": {}}
    for col, cls in enumerate(["Low", "Medium", "High"]):
        rows = []
        for a in ALPHA_GRID:
            blend_o = col_blend(anchor_o, cand_o, col, float(a))
            ev = evaluate(blend_o, y, anchor_pcr)
            ev["alpha"] = float(a)
            ev["delta"] = ev["bal"] - anchor_bal
            rows.append(ev)
        # Best gate-passing α
        passing = [r for r in rows if r["pcr_pass"]]
        peak = max(passing, key=lambda r: r["delta"]) if passing else None
        res["by_col"][cls] = dict(
            sweep=rows, peak=peak,
            n_pass=int(sum(r["pcr_pass"] for r in rows)),
        )
        log(f"  col={cls}: " + (
            f"PEAK α={peak['alpha']:.3f} Δ={peak['delta']:+.5f} "
            f"errs={peak['errs']} PCR={peak['pcr'][0]:.4f}/{peak['pcr'][1]:.4f}/{peak['pcr'][2]:.4f}"
            if peak else "no α passes guardrail"))
    return res


def emit_submission(name_tag, anchor_t, cand_t, col, alpha, ids):
    blend_t = col_blend(anchor_t, cand_t, col, alpha)
    pred = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)
    cls_map = {0: "Low", 1: "Medium", 2: "High"}
    sub = pd.DataFrame({"id": ids, "Irrigation_Need": [cls_map[i] for i in pred]})
    cls_name = ["Low", "Medium", "High"][col]
    p = SUB / f"submission_opt3_colblend_{name_tag}_col{cls_name}_a{int(alpha*1000):03d}.csv"
    sub.to_csv(p, index=False)
    log(f"  EMITTED {p}  pred dist: {dict(sub['Irrigation_Need'].value_counts())}")
    return str(p)


def main():
    t0 = time.time()
    y = load_y()
    log("loading LB-best 4-stack anchor (3-stack ⊗ xgb_metastack_iso α=0.30)")
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    anchor_o = log_blend([lb3_o, meta_o_iso], np.array([0.70, 0.30]))
    anchor_t = log_blend([lb3_t, meta_t_iso], np.array([0.70, 0.30]))

    pred_anc = (np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS).argmax(1)
    anchor_bal = balanced_accuracy_score(y, pred_anc)
    anchor_pcr = per_class_recall(y, pred_anc)
    log(f"  anchor OOF @ recipe bias = {anchor_bal:.5f}  "
        f"PCR L={anchor_pcr[0]:.4f} M={anchor_pcr[1]:.4f} H={anchor_pcr[2]:.4f}")

    ids = pd.read_csv("submissions/submission_recipe_full_te.csv")["id"].values
    summary = {"anchor_bal": float(anchor_bal),
               "anchor_pcr": anchor_pcr.tolist(),
               "candidates": []}

    for name in CANDIDATES:
        log(f"=== {name} ===")
        oof = normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        tst = normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        # Iso-cal for prob-scale alignment with anchor's recipe-bias point.
        oof_iso, tst_iso = iso_cal(oof, tst, y)
        res = sweep_one(name, anchor_o, anchor_t, oof_iso, tst_iso, y,
                        anchor_pcr, anchor_bal)
        # Emit submission for the BEST gate-passing column (highest Δ across
        # all 3 columns) if Δ ≥ EMIT_DELTA. We emit per-column too if
        # they pass the threshold separately, to maximize LB-test surface.
        best_overall = None
        for cls in ("Low", "Medium", "High"):
            peak = res["by_col"][cls]["peak"]
            if peak is None:
                continue
            if best_overall is None or peak["delta"] > best_overall["delta"]:
                best_overall = dict(peak, col=cls)
        if best_overall is not None and best_overall["delta"] >= EMIT_DELTA:
            col_idx = ["Low", "Medium", "High"].index(best_overall["col"])
            tag = name.replace("recipe_full_te_", "")
            res["submission"] = emit_submission(
                tag, anchor_t, tst_iso, col_idx,
                best_overall["alpha"], ids)
            res["best_overall"] = best_overall
        else:
            res["submission"] = None
            res["best_overall"] = best_overall
            log(f"  no emit (best Δ {best_overall['delta'] if best_overall else 'N/A'})")
        summary["candidates"].append(res)

    out = ART / "opt3_colblend_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")
    log(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
