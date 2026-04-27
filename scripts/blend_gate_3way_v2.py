"""Blend gate v2 for the 3 'reopen and dig deeper' candidates.

v2 differences from v1:
  - Apply iso-cal to candidates whose bias profile differs materially
    from recipe bias [1.4324, 1.4689, 3.4008]. KAN's tuned bias is
    [3.93, 2.47, 3.40] — Low offset 2.5x recipe; iso-cal aligns scales.
  - Sweep both raw and iso forms.
  - Report explicit per-class recall guardrail status against LB-best
    4-stack (each class >= anchor - 5e-4).

Decision gate: emit submission only if BOTH:
  - alpha-sweep peak Delta >= +2e-4 vs LB-best 4-stack OOF (= 0.98084)
  - per-class recall guardrail PASS at peak alpha
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

# (oof_basename, descr, default_use_iso)
CANDIDATES = [
    ("multitask_xgb",     "Multi-task XGB (3 main + 3 aux heads, joint obj)",  False),
    ("kan",                "KAN [157->256->128->64->3] on recipe FE",            True),
    ("leakfree_distill",   "Soft-distill with leak-free per-outer-fold teacher", False),
]


def per_class_recall(p: np.ndarray, y: np.ndarray, bias=BIAS) -> np.ndarray:
    pred = (np.log(np.clip(p, EPS, 1)) + bias).argmax(1)
    out = np.zeros(3)
    for k in range(3):
        m = (y == k)
        out[k] = ((pred == k) & m).sum() / max(m.sum(), 1)
    return out


def n_errs(p: np.ndarray, y: np.ndarray, bias=BIAS) -> int:
    pred = (np.log(np.clip(p, EPS, 1)) + bias).argmax(1)
    return int((pred != y).sum())


def jaccard_errs(p1: np.ndarray, p2: np.ndarray, y: np.ndarray,
                 bias=BIAS) -> float:
    e1 = ((np.log(np.clip(p1, EPS, 1)) + bias).argmax(1) != y)
    e2 = ((np.log(np.clip(p2, EPS, 1)) + bias).argmax(1) != y)
    inter = (e1 & e2).sum()
    union = (e1 | e2).sum()
    return float(inter / max(union, 1))


def alpha_sweep(anchor: np.ndarray, candidate: np.ndarray, y: np.ndarray,
                grid=(0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
                ) -> dict:
    out = {}
    a_ref = bal_at_bias(anchor, y)
    a_pcr = per_class_recall(anchor, y)
    for a in grid:
        if a == 0:
            blend = anchor.copy()
        else:
            blend = log_blend([anchor, candidate], np.array([1.0 - a, a]))
        ba = bal_at_bias(blend, y)
        pcr = per_class_recall(blend, y)
        out[f"a{a:.3f}"] = dict(
            alpha=a, bal=float(ba), delta=float(ba - a_ref),
            recL=float(pcr[0]), recM=float(pcr[1]), recH=float(pcr[2]),
            pcr_pass=bool(np.all(pcr >= a_pcr - 5e-4)),
        )
    return out


def best_in_sweep(sweep: dict) -> tuple:
    best_a, best_d, best_pcr_pass, best_pcr = None, -1, False, None
    for k, v in sweep.items():
        if v["delta"] > best_d:
            best_a = v["alpha"]
            best_d = v["delta"]
            best_pcr_pass = v["pcr_pass"]
            best_pcr = (v["recL"], v["recM"], v["recH"])
    return best_a, best_d, best_pcr_pass, best_pcr


def analyze_one(name: str, descr: str, default_iso: bool, y: np.ndarray,
                anchor4: np.ndarray) -> dict:
    oof_p = ART / f"oof_{name}.npy"
    test_p = ART / f"test_{name}.npy"
    if not oof_p.exists():
        return dict(name=name, status="missing_oof", path=str(oof_p))
    oof = normed(np.load(oof_p).astype(np.float32))
    test = normed(np.load(test_p).astype(np.float32)) if test_p.exists() else None

    # Iso-cal both raw and iso variants.
    oof_iso, test_iso = (None, None)
    if test is not None:
        oof_iso, test_iso = iso_cal(oof, test, y)

    res = dict(name=name, descr=descr, status="ok",
                default_iso=default_iso)
    res["raw"] = dict(
        bal_at_recipe_bias=float(bal_at_bias(oof, y)),
        errs_at_recipe_bias=n_errs(oof, y),
        jaccard_vs_4stack=float(jaccard_errs(oof, anchor4, y)),
        sweep=alpha_sweep(anchor4, oof, y),
    )
    if oof_iso is not None:
        res["iso"] = dict(
            bal_at_recipe_bias=float(bal_at_bias(oof_iso, y)),
            errs_at_recipe_bias=n_errs(oof_iso, y),
            jaccard_vs_4stack=float(jaccard_errs(oof_iso, anchor4, y)),
            sweep=alpha_sweep(anchor4, oof_iso, y),
        )
    return res


def print_block(name: str, info: dict, anchor_bal: float) -> None:
    if info.get("status") != "ok":
        print(f"\n[{name}] STATUS: {info.get('status')}", flush=True)
        return
    print(f"\n[{name}] {info['descr']}", flush=True)
    for tag in ("raw", "iso"):
        if tag not in info:
            continue
        d = info[tag]
        a, dd, pp, pcr = best_in_sweep(d["sweep"])
        gate = (dd >= 2e-4) and pp
        print(f"  {tag.upper():4s}  bal={d['bal_at_recipe_bias']:.5f}  "
              f"errs={d['errs_at_recipe_bias']}  "
              f"Jaccard={d['jaccard_vs_4stack']:.4f}  "
              f"peak: alpha={a:.3f} delta={dd:+.5f}  "
              f"PCR=[{pcr[0]:.4f}, {pcr[1]:.4f}, {pcr[2]:.4f}]  "
              f"GATE={'PASS' if gate else 'FAIL'}",
              flush=True)


def main() -> None:
    print(f"[gate v2] loading anchors", flush=True)
    y = load_y()

    s3_o, _ = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_o_iso, _ = iso_cal(meta_o, meta_t, y)
    s4_o = log_blend([s3_o, meta_o_iso], np.array([0.7, 0.3]))
    a4 = bal_at_bias(s4_o, y)
    print(f"[anchors] LB-best 3-stack OOF = {bal_at_bias(s3_o, y):.5f}",
          flush=True)
    print(f"[anchors] LB-best 4-stack OOF = {a4:.5f}", flush=True)

    results = {}
    for name, descr, default_iso in CANDIDATES:
        try:
            info = analyze_one(name, descr, default_iso, y, s4_o)
        except Exception as e:
            info = dict(name=name, status="error", error=str(e))
        print_block(name, info, a4)
        results[name] = info

    out_path = ART / "blend_gate_3way_results.json"
    with open(out_path, "w") as f:
        json.dump(dict(
            anchor_3stack_bal=float(bal_at_bias(s3_o, y)),
            anchor_4stack_bal=float(a4),
            candidates=results,
        ), f, indent=2)
    print(f"\n[gate v2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
