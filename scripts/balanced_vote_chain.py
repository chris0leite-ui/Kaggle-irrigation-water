"""V3-precision + V4 LB-weighted soft-vote + per-anchor right-on-PRIMARY-wrong.

Three diagnostic chains on the cleaned 3-anchor bank from balanced_vote_explore.

Step 1: V3 OOF rows where V3=H and PRIMARY=M. Precision vs macro-recall break-even.
Step 2: V4 = soft-vote with weights ∝ exp((LB - LB_min) / T). T-sweep.
Step 3: per-non-PRIMARY anchor measurement of correct-on-PRIMARY-wrong rows.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402
from tier1b_helpers import ART, BIAS, load_y, log  # noqa: E402
from balanced_vote_explore import reconstruct_bank  # noqa: E402
from balanced_vote_helpers import (  # noqa: E402
    soft_vote_class_weighted, score_predictions, gate_check, PI,
)

OUT = ART / "balanced_vote_chain_results.json"
LB_SCORES = {"PRIMARY": 0.98094, "recipe": 0.97939, "catboost": 0.97935}


def step1_v3_precision(bank: dict, y: np.ndarray, anchor_pred: np.ndarray) -> dict:
    cleaned = {n: bank[n] for n in ["PRIMARY", "recipe", "catboost"]}
    probs = [cleaned[k][0] for k in cleaned]
    v3_oof = soft_vote_class_weighted(probs).argmax(1)

    mask = (v3_oof == 2) & (anchor_pred == 1)  # V3=H, PRIMARY=M
    n_override = int(mask.sum())
    correct_H = int((y[mask] == 2).sum())
    primary_right_M = int((y[mask] == 1).sum())
    neither_L = int((y[mask] == 0).sum())
    precision = correct_H / max(n_override, 1)

    cc = np.bincount(y, minlength=3)
    breakeven = cc[2] / (cc[1] + cc[2])

    override = anchor_pred.copy()
    override[mask] = 2
    delta_oof = fast_bal_acc(y, override, class_counts=cc) - fast_bal_acc(y, anchor_pred, class_counts=cc)
    return {
        "n_override": n_override, "correct_H": correct_H,
        "primary_right_M": primary_right_M, "neither_right_L": neither_L,
        "precision": float(precision), "breakeven": float(breakeven),
        "passes_breakeven": bool(precision > breakeven),
        "delta_macro_recall_OOF": float(delta_oof),
        "verdict": "DEPLOY" if delta_oof > 2e-4 else "NULL",
    }


def step2_v4_lb_weighted(bank: dict, y: np.ndarray, anchor_pred: np.ndarray) -> dict:
    cleaned = {n: bank[n] for n in ["PRIMARY", "recipe", "catboost"]}
    probs_oof = [cleaned[k][0] for k in cleaned]
    probs_test = [cleaned[k][1] for k in cleaned]
    lb_min = min(LB_SCORES.values())
    lb_diffs = np.array([LB_SCORES[n] - lb_min for n in cleaned])

    sweep = []
    for T in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]:
        weights = np.exp(lb_diffs / T); weights = weights / weights.sum()
        avg_o = sum(w * p for w, p in zip(weights, probs_oof))
        avg_t = sum(w * p for w, p in zip(weights, probs_test))
        # 1/π_c re-weight + renorm
        bal_o = avg_o / PI[None, :]; bal_o = bal_o / bal_o.sum(axis=1, keepdims=True)
        bal_t = avg_t / PI[None, :]; bal_t = bal_t / bal_t.sum(axis=1, keepdims=True)
        s_raw = score_predictions(bal_o.argmax(1), y, anchor_pred, f"V4_T{T}_raw")
        bias, _ = tune_log_bias(bal_o, y, prior=PI)
        s_tun = score_predictions(
            (np.log(np.clip(bal_o, 1e-12, 1)) + bias).argmax(1),
            y, anchor_pred, f"V4_T{T}_tuned")
        sweep.append({"T": T, "weights": weights.tolist(),
                      "raw": s_raw, "tuned": s_tun, "bias": bias.tolist()})
    return {"sweep": sweep}


def step3_per_anchor(bank: dict, y: np.ndarray, anchor_pred: np.ndarray) -> dict:
    out = {}
    for name in bank:
        if name == "PRIMARY":
            continue
        pred = (np.log(np.clip(bank[name][0], 1e-12, 1)) + BIAS).argmax(1)
        anchor_right = (anchor_pred == y); cand_right = (pred == y)
        disagree = (pred != anchor_pred)
        cand_wins = int((cand_right & disagree).sum())
        primary_wins = int((anchor_right & disagree).sum())
        n_dis = int(disagree.sum())

        per_class = {}
        for k, cn in enumerate(["Low", "Medium", "High"]):
            in_k = (y == k)
            n_cls = int(in_k.sum())
            n_right_pr_wrong = int((cand_right & ~anchor_right & in_k).sum())
            n_wrong_pr_right = int((~cand_right & anchor_right & in_k).sum())
            per_class[cn] = {
                "cand_right_pr_wrong": n_right_pr_wrong,
                "cand_wrong_pr_right": n_wrong_pr_right,
                "net_lift_if_swap": n_right_pr_wrong - n_wrong_pr_right,
                "macro_recall_contribution": (n_right_pr_wrong - n_wrong_pr_right) / max(n_cls, 1),
            }
        # Net macro-recall contribution if we surgically replaced PRIMARY by candidate
        # on the disagreement rows (oracle case — assumes we know true labels).
        oracle_macro_lift = sum(per_class[c]["macro_recall_contribution"] / 3
                                 for c in per_class)
        out[name] = {
            "disagreements": n_dis, "cand_wins": cand_wins,
            "primary_wins": primary_wins,
            "cand_win_rate": cand_wins / max(n_dis, 1),
            "per_class": per_class,
            "oracle_macro_lift": float(oracle_macro_lift),
        }
    return out


def main():
    log("=== V3-precision + V4 LB-weighted + per-anchor diagnostic chain ===")
    y = load_y()
    bank = reconstruct_bank(y)
    anchor_pred = (np.log(np.clip(bank["PRIMARY"][0], 1e-12, 1)) + BIAS).argmax(1)

    log("\n=== STEP 1: V3-precision probe (V3=H, PRIMARY=M) ===")
    s1 = step1_v3_precision(bank, y, anchor_pred)
    log(f"  override n             = {s1['n_override']}")
    log(f"  correct (truly H)      = {s1['correct_H']}")
    log(f"  primary right (M)      = {s1['primary_right_M']}")
    log(f"  neither right (L)      = {s1['neither_right_L']}")
    log(f"  precision              = {s1['precision']*100:.2f}%")
    log(f"  break-even             = {s1['breakeven']*100:.2f}%")
    log(f"  passes break-even      = {s1['passes_breakeven']}")
    log(f"  Δ macro OOF (override) = {s1['delta_macro_recall_OOF']:+.5f}")
    log(f"  verdict                = {s1['verdict']}")

    log("\n=== STEP 2: V4 LB-weighted soft-vote T-sweep ===")
    s2 = step2_v4_lb_weighted(bank, y, anchor_pred)
    print(f"  {'T':>8s}  {'wPRI':>6s} {'wREC':>6s} {'wCAT':>6s}  "
          f"{'rawΔ':>9s} {'tunΔ':>9s} {'tun_errs':>8s} "
          f"{'tun_recH':>9s} {'tun_netH':>9s} {'tun_asy':>7s}  gates")
    for r in s2["sweep"]:
        g = gate_check(r["tuned"])
        gs = "".join("✓" if g[k] else "✗" for k in ("G1","G2","G3","G4"))
        w = r["weights"]
        t = r["tuned"]
        print(f"  {r['T']:>8.4f}  {w[0]:>6.3f} {w[1]:>6.3f} {w[2]:>6.3f}  "
              f"{r['raw']['delta_bal']:+9.5f} {t['delta_bal']:+9.5f} "
              f"{t['errs']:>8d} {t['rec_H']:>9.5f} {t['net_H']:>+9d} "
              f"{t['asym']:>+7.2f}  {gs}")

    log("\n=== STEP 3: per-anchor right-on-PRIMARY-wrong (disagreement geometry) ===")
    s3 = step3_per_anchor(bank, y, anchor_pred)
    print(f"  {'anchor':>14s}  {'n_dis':>6s}  {'cand_W':>6s}  {'pri_W':>6s}  "
          f"{'cand%':>6s}  {'L':>20s}  {'M':>20s}  {'H':>20s}  oracle_macro_lift")
    for n, r in s3.items():
        L = r["per_class"]["Low"]; M = r["per_class"]["Medium"]; H = r["per_class"]["High"]
        Ls = f"+{L['cand_right_pr_wrong']}/-{L['cand_wrong_pr_right']}"
        Ms = f"+{M['cand_right_pr_wrong']}/-{M['cand_wrong_pr_right']}"
        Hs = f"+{H['cand_right_pr_wrong']}/-{H['cand_wrong_pr_right']}"
        print(f"  {n:>14s}  {r['disagreements']:>6d}  {r['cand_wins']:>6d}  "
              f"{r['primary_wins']:>6d}  {r['cand_win_rate']*100:>5.2f}%  "
              f"{Ls:>20s}  {Ms:>20s}  {Hs:>20s}  {r['oracle_macro_lift']:+.5f}")

    log("\n=== persist ===")
    with open(OUT, "w") as f:
        json.dump({"step1": s1, "step2": s2, "step3": s3}, f, indent=2)
    log(f"  saved {OUT}")


if __name__ == "__main__":
    main()
