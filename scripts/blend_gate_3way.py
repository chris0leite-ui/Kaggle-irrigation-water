"""Blend gate for the 3 'reopen and dig deeper' candidates.

Loads the 3 candidate OOFs:
  1. multitask_xgb         — multi-task base XGB with aux heads in objective
  2. kan                   — KAN with recipe FE (loaded if Kaggle kernel done)
  3. leakfree_distill      — proper leak-eliminated distillation student

For each: standalone OOF/test diagnostics + Jaccard vs LB-best 4-stack +
fixed-bias α-sweep into LB-best 4-stack and prior LB-best 3-stack.

Decision gate (per CLAUDE.md): emit blend candidate if BOTH:
  - α-sweep peak Δ ≥ +0.0002 vs LB-best 4-stack OOF (= 0.98084)
  - per-class recall guardrail PASS (each class ≥ anchor − 5e-4)

Anchor data: recipe bias [1.4324, 1.4689, 3.4008]; LB-best primary OOF
0.98084 (LB 0.98094); prior LB-best 3-stack OOF 0.98061 (LB 0.98008).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    bal_at_bias, build_lbbest_stack, iso_cal, load_y, normed, BIAS,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
EPS = 1e-9

CANDIDATES = [
    ("multitask_xgb",       "Multi-task XGB at base level (aux heads in objective)"),
    ("kan",                  "KAN with recipe FE [256,128,64], grid=5, order=3"),
    ("leakfree_distill",     "Soft-distill with leak-free per-outer-fold teacher"),
]


def per_class_recall(p: np.ndarray, y: np.ndarray, bias=BIAS) -> np.ndarray:
    pred = (np.log(np.clip(p, EPS, 1)) + bias).argmax(1)
    out = np.zeros(3)
    for k in range(3):
        m = (y == k)
        out[k] = ((pred == k) & m).sum() / max(m.sum(), 1)
    return out


def jaccard_errs(p1: np.ndarray, p2: np.ndarray, y: np.ndarray,
                 bias=BIAS) -> float:
    e1 = ((np.log(np.clip(p1, EPS, 1)) + bias).argmax(1) != y)
    e2 = ((np.log(np.clip(p2, EPS, 1)) + bias).argmax(1) != y)
    inter = (e1 & e2).sum()
    union = (e1 | e2).sum()
    return float(inter / max(union, 1))


def alpha_sweep(anchor: np.ndarray, candidate: np.ndarray, y: np.ndarray,
                grid=(0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)) -> dict:
    out = {}
    for a in grid:
        if a == 0:
            blend = anchor.copy()
        else:
            blend = log_blend([anchor, candidate], np.array([1.0 - a, a]))
        ba = bal_at_bias(blend, y)
        pcr = per_class_recall(blend, y)
        out[f"a{a:.3f}"] = dict(alpha=a, bal=float(ba),
                                 recL=float(pcr[0]), recM=float(pcr[1]),
                                 recH=float(pcr[2]))
    return out


def analyze_one(name: str, descr: str, y: np.ndarray,
                anchor3: np.ndarray, anchor4: np.ndarray) -> dict:
    oof_p = ART / f"oof_{name}.npy"
    test_p = ART / f"test_{name}.npy"
    if not oof_p.exists():
        return dict(name=name, status="missing_oof", path=str(oof_p))
    oof = normed(np.load(oof_p).astype(np.float32))
    test = normed(np.load(test_p).astype(np.float32)) if test_p.exists() else None

    # Standalone diagnostics at recipe bias.
    standalone_bal = bal_at_bias(oof, y)
    pcr_std = per_class_recall(oof, y)
    errs_std = ((np.log(np.clip(oof, EPS, 1)) + BIAS).argmax(1) != y).sum()

    # Iso-cal (same as LB-best primary's xgb_metastack).
    oof_iso, test_iso = (None, None)
    if test is not None:
        oof_iso, test_iso = iso_cal(oof, test, y)

    # Vs anchors.
    j3 = jaccard_errs(oof, anchor3, y)
    j4 = jaccard_errs(oof, anchor4, y)
    sweep_a3 = alpha_sweep(anchor3, oof, y)
    sweep_a4 = alpha_sweep(anchor4, oof, y)

    # Iso variant sweep too.
    sweep_iso_a4 = None
    if oof_iso is not None:
        sweep_iso_a4 = alpha_sweep(anchor4, oof_iso, y)

    # Best blend Δ vs each anchor.
    def best_delta(sweep: dict, anchor_bal: float) -> tuple:
        best_a, best_d, best_pcr = None, -1, None
        for k, v in sweep.items():
            d = v["bal"] - anchor_bal
            if d > best_d:
                best_a, best_d = v["alpha"], d
                best_pcr = (v["recL"], v["recM"], v["recH"])
        return best_a, best_d, best_pcr

    a3_bal = bal_at_bias(anchor3, y)
    a4_bal = bal_at_bias(anchor4, y)
    a3_best = best_delta(sweep_a3, a3_bal)
    a4_best = best_delta(sweep_a4, a4_bal)
    a4_iso_best = best_delta(sweep_iso_a4, a4_bal) if sweep_iso_a4 else None

    # Decision gates: emit if Δ vs LB-4 ≥ +2e-4 AND PCR within -5e-4 each class.
    a4_pcr = per_class_recall(anchor4, y)
    a4_pcr_floor = a4_pcr - 5e-4

    def gate_pass(best_a, best_d, best_pcr) -> bool:
        if best_d < 2e-4:
            return False
        return all(p >= floor for p, floor in zip(best_pcr, a4_pcr_floor))

    raw_pass = gate_pass(*a4_best)
    iso_pass = gate_pass(*a4_iso_best) if a4_iso_best else False

    return dict(
        name=name, descr=descr, status="ok",
        standalone=dict(bal=float(standalone_bal),
                        errs=int(errs_std),
                        recL=float(pcr_std[0]),
                        recM=float(pcr_std[1]),
                        recH=float(pcr_std[2])),
        jaccard_vs_3stack=float(j3),
        jaccard_vs_4stack=float(j4),
        sweep_vs_3stack=sweep_a3,
        sweep_vs_4stack=sweep_a4,
        sweep_iso_vs_4stack=sweep_iso_a4,
        peak_vs_3stack=dict(alpha=a3_best[0], delta=a3_best[1],
                            recL=a3_best[2][0], recM=a3_best[2][1],
                            recH=a3_best[2][2]),
        peak_vs_4stack=dict(alpha=a4_best[0], delta=a4_best[1],
                            recL=a4_best[2][0], recM=a4_best[2][1],
                            recH=a4_best[2][2]),
        peak_iso_vs_4stack=(dict(alpha=a4_iso_best[0],
                                  delta=a4_iso_best[1],
                                  recL=a4_iso_best[2][0],
                                  recM=a4_iso_best[2][1],
                                  recH=a4_iso_best[2][2])
                             if a4_iso_best else None),
        gate_pass_raw=bool(raw_pass),
        gate_pass_iso=bool(iso_pass),
    )


def main() -> None:
    print("[blend_gate_3way] loading anchors", flush=True)
    y = load_y()

    # Reconstruct LB-best 3-stack and 4-stack.
    s3_o, _ = build_lbbest_stack(y)              # 3-stack (lb3 + RealMLP + nr_iso)
    # 4-stack = 3-stack + xgb_metastack_iso α=0.30 (the LB 0.98094 winner).
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_o_iso, _ = iso_cal(meta_o, meta_t, y)
    s4_o = log_blend([s3_o, meta_o_iso], np.array([0.7, 0.3]))

    print(f"[anchors] 3-stack OOF bal = {bal_at_bias(s3_o, y):.5f}", flush=True)
    print(f"[anchors] 4-stack OOF bal = {bal_at_bias(s4_o, y):.5f}", flush=True)

    results = {}
    for name, descr in CANDIDATES:
        print(f"[gate] analyzing {name}: {descr}", flush=True)
        try:
            results[name] = analyze_one(name, descr, y, s3_o, s4_o)
        except Exception as e:
            results[name] = dict(name=name, status="error", error=str(e))
        if results[name].get("status") == "ok":
            r = results[name]
            print(f"  standalone: bal={r['standalone']['bal']:.5f} "
                  f"errs={r['standalone']['errs']} "
                  f"PCR=[{r['standalone']['recL']:.4f}, "
                  f"{r['standalone']['recM']:.4f}, "
                  f"{r['standalone']['recH']:.4f}]", flush=True)
            print(f"  Jaccard vs 3-stack = {r['jaccard_vs_3stack']:.4f}, "
                  f"vs 4-stack = {r['jaccard_vs_4stack']:.4f}", flush=True)
            print(f"  peak vs 3-stack: α={r['peak_vs_3stack']['alpha']:.3f} "
                  f"Δ={r['peak_vs_3stack']['delta']:+.5f}", flush=True)
            print(f"  peak vs 4-stack: α={r['peak_vs_4stack']['alpha']:.3f} "
                  f"Δ={r['peak_vs_4stack']['delta']:+.5f}", flush=True)
            if r['peak_iso_vs_4stack']:
                print(f"  peak iso vs 4-stack: "
                      f"α={r['peak_iso_vs_4stack']['alpha']:.3f} "
                      f"Δ={r['peak_iso_vs_4stack']['delta']:+.5f}", flush=True)
            print(f"  GATE: raw={'PASS' if r['gate_pass_raw'] else 'FAIL'}, "
                  f"iso={'PASS' if r['gate_pass_iso'] else 'FAIL'}", flush=True)

    out_path = ART / "blend_gate_3way_results.json"
    with open(out_path, "w") as f:
        json.dump(dict(
            anchor_3stack_bal=float(bal_at_bias(s3_o, y)),
            anchor_4stack_bal=float(bal_at_bias(s4_o, y)),
            candidates=results,
        ), f, indent=2)
    print(f"\n[blend_gate_3way] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
