"""LB-ready submission emitter for tomorrow's probes.

Three candidates from the senior-FE work, each tuned to maximize the
chance of LB transfer relative to the LB-best 4-stack primary
(submission_tier1b_greedy_meta.csv → LB 0.98094).

OPTION 1 — meta-stacker with 3-way + nndist in pool (NEW LB candidate):
  oof_xgb_metastack_3wnn (this session) iso-cal'd, blended into
  LB-best 3-stack at α ∈ {0.30, 0.50}. The α=0.30 emission mirrors
  the LB-validated v1 mechanism exactly; only the meta itself is
  upgraded. The α=0.50 emission is the OOF-peak (more aggressive,
  higher selection-overfit risk).

OPTION 2 — NN-dist conditional override on rule-disagreement rows:
  Even though the NN-route OOF sweep produced Δ ≈ 0, the mechanism
  is structurally novel (per-row hard override, NOT log-blend). The
  goal here is not OOF lift but LB-test the override mechanism with
  a config that DOES produce test-row diffs vs the LB-best primary.
  Pick the highest-flip config that passes per-class guardrail on OOF.

OPTION 3 — per-class column-selective blend:
  Same logic — OOF Δ ≈ 0, but the mechanism is novel. Find the col×α
  combo with maximum test-row diff vs primary while passing guardrail.

For each option, also save a results JSON documenting:
  - candidate path
  - OOF Δ vs anchor
  - per-class recall delta
  - test-row count differing from primary
  - expected LB delta (via various calibration extrapolations)
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

GUARDRAIL = 5e-4


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return np.diag(cm).astype(float) / cm.sum(axis=1).clip(1)


def predict_anchor(p):
    return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)


def write_sub(name, ids, pred, summary_dict):
    cls_map = {0: "Low", 1: "Medium", 2: "High"}
    sub = pd.DataFrame({"id": ids,
                        "Irrigation_Need": [cls_map[i] for i in pred]})
    p = SUB / f"submission_{name}.csv"
    sub.to_csv(p, index=False)
    summary_dict[name] = dict(
        path=str(p),
        pred_dist=dict(sub["Irrigation_Need"].value_counts()),
    )
    log(f"  emitted {p}  pred dist: {summary_dict[name]['pred_dist']}")
    return sub


def diff_vs_primary(sub_df, primary_path):
    """Count rows where this submission differs from primary."""
    primary = pd.read_csv(primary_path)
    return int((sub_df["Irrigation_Need"].values
                != primary["Irrigation_Need"].values).sum())


# --------------------------------------------------------- Option 1
def opt1_meta_3wnn(y, anchor_o, anchor_t, ids, primary_path, summary):
    log("=== OPTION 1: meta-stacker 3wnn iso-cal blend ===")
    meta_o = normed(np.load(ART / "oof_xgb_metastack_3wnn.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack_3wnn.npy").astype(np.float32))
    # LB-best 3-stack (without the LB-validated meta layer)
    lb3_o, lb3_t = build_lbbest_stack(y)
    lb3_bal = balanced_accuracy_score(y, predict_anchor(lb3_o))
    # iso-cal aligns prob scale with anchor's bias
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    log(f"  meta_iso standalone @ recipe-bias: "
        f"{balanced_accuracy_score(y, predict_anchor(meta_o_iso)):.5f}")

    # Emit at three α values for tomorrow's probe.
    for a in (0.30, 0.40, 0.50):
        blend_o = log_blend([lb3_o, meta_o_iso], np.array([1 - a, a]))
        blend_t = log_blend([lb3_t, meta_t_iso], np.array([1 - a, a]))
        pred_o = predict_anchor(blend_o)
        pred_t = predict_anchor(blend_t)
        bal = balanced_accuracy_score(y, pred_o)
        pcr = per_class_recall(y, pred_o)
        log(f"  α={a:.2f}  OOF={bal:.5f}  Δ vs lb3 = {bal-lb3_bal:+.5f}  "
            f"PCR L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}")
        name = f"opt1_meta3wnn_iso_a{int(a*100):03d}"
        sub = write_sub(name, ids, pred_t, summary)
        summary[name].update(
            oof_bal=float(bal), oof_delta_vs_lb3=float(bal - lb3_bal),
            pcr=pcr.tolist(),
            test_diff_vs_primary=diff_vs_primary(sub, primary_path),
        )


# --------------------------------------------------------- Option 2
def opt2_nn_route(y, anchor_o, anchor_t, ids, primary_path, summary):
    log("=== OPTION 2: NN-route override (force-emit highest-flip configs) ===")
    nnd_oof = np.load(ART / "oof_nn_dist_features.npy").astype(np.float32)
    nnd_test = np.load(ART / "test_nn_dist_features.npy").astype(np.float32)
    frac_high_oof = nnd_oof[:, 4]
    frac_high_test = nnd_test[:, 4]
    nnd_pred_oof = normed(np.load(ART / "oof_recipe_full_te_nndist.npy"))
    nnd_pred_test = normed(np.load(ART / "test_recipe_full_te_nndist.npy"))
    nnd_argmax_oof = predict_anchor(nnd_pred_oof)
    nnd_argmax_test = predict_anchor(nnd_pred_test)
    nnd_high_oof = nnd_pred_oof[:, 2]   # P(High | nndist standalone)
    nnd_high_test = nnd_pred_test[:, 2]

    pred_anc = predict_anchor(anchor_o)
    bal_anc = balanced_accuracy_score(y, pred_anc)
    pcr_anc = per_class_recall(y, pred_anc)

    # Pick highest-flip config that passes guardrail. Use 3 mechanisms:
    #
    # A) HtoNotH conservative: anchor=High AND frac_high < τL_low
    # B) HtoNotH aggressive:   anchor=High AND nndist_argmax != High AND nndist[High] < τM
    # C) NotHtoH soft:         anchor != High AND nndist[High] > τH (soft prob, not argmax)
    #
    # For each, scan a sub-grid and pick the cell maximizing test-flip count
    # subject to OOF guardrail PASS.
    candidates = []

    def apply(pred, mask, target):
        out = pred.copy()
        out[mask] = target
        return out

    # A: HtoNotH conservative
    for tL in (0.05, 0.10, 0.15, 0.20):
        m_o = (pred_anc == 2) & (frac_high_oof < tL)
        if m_o.sum() == 0:
            continue
        # next-best class on these rows
        biased = np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS
        biased_no_h = biased[m_o].copy()
        biased_no_h[:, 2] = -np.inf
        new_o = biased_no_h.argmax(1)
        pred_o_test = pred_anc.copy()
        pred_o_test[m_o] = new_o
        pcr_t = per_class_recall(y, pred_o_test)
        bal_t = balanced_accuracy_score(y, pred_o_test)
        if all(pcr_t[c] >= pcr_anc[c] - GUARDRAIL for c in range(3)):
            # apply to test
            m_te = (predict_anchor(anchor_t) == 2) & (frac_high_test < tL)
            biased_te = np.log(np.clip(anchor_t, 1e-12, 1)) + BIAS
            biased_te_no_h = biased_te[m_te].copy()
            biased_te_no_h[:, 2] = -np.inf
            pred_te = predict_anchor(anchor_t).copy()
            pred_te[m_te] = biased_te_no_h.argmax(1)
            candidates.append(dict(
                name=f"A_HtoNotH_tL{int(tL*100):02d}",
                pred_test=pred_te, oof_bal=bal_t,
                pcr=pcr_t.tolist(),
                ov_oof=int(m_o.sum()), ov_test=int(m_te.sum()),
            ))

    # B: HtoNotH aggressive (gated by nndist disagreement)
    for tM in (0.30, 0.40, 0.50, 0.60):
        m_o = (pred_anc == 2) & (nnd_argmax_oof != 2) & (nnd_high_oof < tM)
        if m_o.sum() == 0:
            continue
        biased = np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS
        biased_no_h = biased[m_o].copy()
        biased_no_h[:, 2] = -np.inf
        pred_o_test = pred_anc.copy()
        pred_o_test[m_o] = biased_no_h.argmax(1)
        pcr_t = per_class_recall(y, pred_o_test)
        bal_t = balanced_accuracy_score(y, pred_o_test)
        if all(pcr_t[c] >= pcr_anc[c] - GUARDRAIL for c in range(3)):
            m_te = (predict_anchor(anchor_t) == 2) & (nnd_argmax_test != 2) & (nnd_high_test < tM)
            biased_te = np.log(np.clip(anchor_t, 1e-12, 1)) + BIAS
            biased_te_no_h = biased_te[m_te].copy()
            biased_te_no_h[:, 2] = -np.inf
            pred_te = predict_anchor(anchor_t).copy()
            pred_te[m_te] = biased_te_no_h.argmax(1)
            candidates.append(dict(
                name=f"B_HtoNotH_aggressive_tM{int(tM*100):02d}",
                pred_test=pred_te, oof_bal=bal_t,
                pcr=pcr_t.tolist(),
                ov_oof=int(m_o.sum()), ov_test=int(m_te.sum()),
            ))

    # C: NotHtoH soft (gated by NN-dist High prob)
    for tH in (0.50, 0.60, 0.70, 0.80, 0.90):
        m_o = (pred_anc != 2) & (nnd_high_oof > tH)
        if m_o.sum() == 0:
            continue
        pred_o_test = pred_anc.copy()
        pred_o_test[m_o] = 2
        pcr_t = per_class_recall(y, pred_o_test)
        bal_t = balanced_accuracy_score(y, pred_o_test)
        if all(pcr_t[c] >= pcr_anc[c] - GUARDRAIL for c in range(3)):
            m_te = (predict_anchor(anchor_t) != 2) & (nnd_high_test > tH)
            pred_te = predict_anchor(anchor_t).copy()
            pred_te[m_te] = 2
            candidates.append(dict(
                name=f"C_NotHtoH_soft_tH{int(tH*100):02d}",
                pred_test=pred_te, oof_bal=bal_t,
                pcr=pcr_t.tolist(),
                ov_oof=int(m_o.sum()), ov_test=int(m_te.sum()),
            ))

    log(f"  {len(candidates)} candidates passed OOF guardrail")
    # Pick top 3 by test-flip count (the ones that actually probe the LB)
    candidates.sort(key=lambda c: c["ov_test"], reverse=True)
    for c in candidates[:3]:
        name = f"opt2_route_{c['name']}"
        sub = write_sub(name, ids, c["pred_test"], summary)
        summary[name].update(
            oof_bal=float(c["oof_bal"]),
            oof_delta_vs_anchor=float(c["oof_bal"] - bal_anc),
            pcr=c["pcr"],
            override_oof=c["ov_oof"], override_test=c["ov_test"],
            test_diff_vs_primary=diff_vs_primary(sub, primary_path),
        )
        log(f"    {name}: OOF Δ={c['oof_bal']-bal_anc:+.5f} "
            f"oof_overrides={c['ov_oof']} test_overrides={c['ov_test']}")


# --------------------------------------------------------- Option 3
def opt3_col_blend(y, anchor_o, anchor_t, ids, primary_path, summary):
    log("=== OPTION 3: column-selective blend (force-emit highest-flip) ===")
    pred_anc = predict_anchor(anchor_o)
    bal_anc = balanced_accuracy_score(y, pred_anc)
    pcr_anc = per_class_recall(y, pred_anc)

    candidates = []
    for cand_name in ("recipe_full_te_3way", "recipe_full_te_nndist"):
        cand_o = normed(np.load(ART / f"oof_{cand_name}.npy").astype(np.float32))
        cand_t = normed(np.load(ART / f"test_{cand_name}.npy").astype(np.float32))
        cand_o_iso, cand_t_iso = iso_cal(cand_o, cand_t, y)
        for col, cls in enumerate(("Low", "Medium", "High")):
            for a in (0.10, 0.20, 0.30, 0.50, 0.70):
                blend_o = anchor_o.copy()
                blend_o[:, col] = (1 - a) * anchor_o[:, col] + a * cand_o_iso[:, col]
                blend_o = blend_o / np.clip(blend_o.sum(1, keepdims=True), 1e-9, None)
                pred_o = predict_anchor(blend_o)
                bal = balanced_accuracy_score(y, pred_o)
                pcr = per_class_recall(y, pred_o)
                if not all(pcr[c] >= pcr_anc[c] - GUARDRAIL for c in range(3)):
                    continue
                # test version
                blend_t = anchor_t.copy()
                blend_t[:, col] = (1 - a) * anchor_t[:, col] + a * cand_t_iso[:, col]
                blend_t = blend_t / np.clip(blend_t.sum(1, keepdims=True), 1e-9, None)
                pred_t = predict_anchor(blend_t)
                test_flips = int((pred_t != predict_anchor(anchor_t)).sum())
                candidates.append(dict(
                    name=f"{cand_name.replace('recipe_full_te_','')}_col{cls}_a{int(a*100):03d}",
                    pred_test=pred_t,
                    oof_bal=bal, pcr=pcr.tolist(),
                    test_flips=test_flips,
                ))
    log(f"  {len(candidates)} candidates passed OOF guardrail")
    candidates.sort(key=lambda c: c["test_flips"], reverse=True)
    for c in candidates[:4]:
        name = f"opt3_col_{c['name']}"
        sub = write_sub(name, ids, c["pred_test"], summary)
        summary[name].update(
            oof_bal=float(c["oof_bal"]),
            oof_delta_vs_anchor=float(c["oof_bal"] - bal_anc),
            pcr=c["pcr"],
            test_flips_vs_anchor=c["test_flips"],
            test_diff_vs_primary=diff_vs_primary(sub, primary_path),
        )
        log(f"    {name}: OOF Δ={c['oof_bal']-bal_anc:+.5f} "
            f"test_flips={c['test_flips']}")


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

    bal_anc = balanced_accuracy_score(y, predict_anchor(anchor_o))
    pcr_anc = per_class_recall(y, predict_anchor(anchor_o))
    log(f"  anchor (LB-best 4-stack) OOF = {bal_anc:.5f}  "
        f"PCR L={pcr_anc[0]:.4f} M={pcr_anc[1]:.4f} H={pcr_anc[2]:.4f}")

    ids = pd.read_csv("submissions/submission_recipe_full_te.csv")["id"].values
    primary_path = "submissions/submission_tier1b_greedy_meta.csv"

    summary = {"anchor_oof_bal": float(bal_anc),
               "anchor_pcr": pcr_anc.tolist(),
               "primary": primary_path}
    opt1_meta_3wnn(y, anchor_o, anchor_t, ids, primary_path, summary)
    opt2_nn_route(y, anchor_o, anchor_t, ids, primary_path, summary)
    opt3_col_blend(y, anchor_o, anchor_t, ids, primary_path, summary)

    out = ART / "lb_ready_emit_results.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {out}")
    log(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
