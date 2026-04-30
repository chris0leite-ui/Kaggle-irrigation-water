"""T1-v2 — score LLM responses against critical-rows ground truth.

Generic for haiku / sonnet / opus. Wilson 95% CI. Outputs:
  - precision at CONF thresholds
  - intersection scoring across multiple LLMs
  - 3-tier verdict: CLEAR / MARGINAL / NULL

Usage:
  python scripts/T1_llm_v2_score.py haiku       # scores response_haiku_*.txt
  python scripts/T1_llm_v2_score.py sonnet      # scores response_sonnet_*.txt
  python scripts/T1_llm_v2_score.py haiku sonnet # intersection scoring
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ART = Path("scripts/artifacts")
OUT = ART / "T1_v2"

ROW_RX = re.compile(
    r"ROW\s+(\d+)\s*\n"
    r"\s*RULE_SCORE:\s*(\d+)\s*\n"
    r"\s*RULE_PRED:\s*(\w+)\s*\n"
    r"\s*FINAL:\s*(\w+)\s*\n"
    r"\s*CONF:\s*([\d.]+)\s*\n"
    r"\s*REASON:\s*([^\n]*)",
    flags=re.IGNORECASE,
)


def parse(text: str) -> pd.DataFrame:
    rows = []
    for m in ROW_RX.finditer(text):
        rows.append({
            "pseudo_id": int(m.group(1)),
            "rule_score": int(m.group(2)),
            "rule_pred": m.group(3).strip().title(),
            "final": m.group(4).strip().title(),
            "conf": float(m.group(5)),
            "reason": m.group(6).strip(),
        })
    return pd.DataFrame(rows)


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson 95% CI for binomial proportion."""
    if n == 0:
        return (None, None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, centre - half, centre + half)


def load_responses(model: str) -> pd.DataFrame:
    frames = []
    for p in sorted(OUT.glob(f"response_{model}_*.txt")):
        text = p.read_text()
        df = parse(text)
        df["batch"] = p.stem
        frames.append(df)
    if not frames:
        raise SystemExit(f"no response_{model}_*.txt files in {OUT}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("pseudo_id", keep="first")


def score_one(model: str, keys: pd.DataFrame) -> dict:
    parsed = load_responses(model)
    df = parsed.merge(keys, on="pseudo_id", how="inner")
    print(f"\n=== {model} ===")
    print(f"matched: {len(df)} rows / {len(keys)} expected")
    print(f"FINAL distribution: {df['final'].value_counts().to_dict()}")
    print(f"CONF mean={df['conf'].mean():.3f}, median={df['conf'].median():.3f}")

    bank_only_acc = (df["true_label"] == "Medium").mean()
    correct = (df["final"] == df["true_label"]).sum()
    print(f"\nBANK-ONLY (always M) acc: {bank_only_acc:.4f} ({int((df['true_label'] == 'Medium').sum())}/{len(df)})")
    print(f"{model.upper()} acc: {correct / len(df):.4f} ({correct}/{len(df)})")

    print(f"\n{model.upper()} M-verdict precision (Wilson 95% CI):")
    pred_M_stats = {}
    for thr in (0.0, 0.7, 0.8, 0.9):
        sub = df[(df["final"] == "Medium") & (df["conf"] >= thr)]
        if len(sub) == 0:
            print(f"  CONF>={thr}: 0 rows")
            continue
        n_correct = int((sub["true_label"] == "Medium").sum())
        n = len(sub)
        p, lo, hi = wilson_ci(n_correct, n)
        print(f"  CONF>={thr}: {n} rows  P(true=M)={p:.4f}  CI=[{lo:.4f}, {hi:.4f}]  ({n_correct}/{n})")
        pred_M_stats[f"thr_{thr}"] = {
            "n": n, "n_correct": n_correct, "precision": p, "ci_lo": lo, "ci_hi": hi
        }

    print(f"\n{model.upper()} H-verdict precision:")
    for thr in (0.0, 0.7, 0.8):
        sub = df[(df["final"] == "High") & (df["conf"] >= thr)]
        if len(sub) == 0:
            continue
        n_correct = int((sub["true_label"] == "High").sum())
        n = len(sub)
        p, lo, hi = wilson_ci(n_correct, n)
        print(f"  CONF>={thr}: {n} rows  P(true=H)={p:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

    # Verdict on M @ CONF>=0.7
    main_stats = pred_M_stats.get("thr_0.7", {})
    if main_stats:
        p, lo, hi = main_stats["precision"], main_stats["ci_lo"], main_stats["ci_hi"]
        print(f"\n=== VERDICT for {model} ===")
        print(f"M @ CONF>=0.7 precision: {p:.4f}  Wilson 95% CI [{lo:.4f}, {hi:.4f}]  on n={main_stats['n']}")
        print(f"Bank-only baseline:      {bank_only_acc:.4f}")
        print(f"Required to clear:       0.92")
        if lo >= 0.90 and p >= 0.92:
            verdict = "CLEAR"
        elif p >= 0.88:
            verdict = "MARGINAL"
        else:
            verdict = "NULL"
        print(f"Verdict: {verdict}")
        return {
            "model": model,
            "n_matched": len(df),
            "bank_only_acc": float(bank_only_acc),
            "model_acc": float(correct / len(df)),
            "M_conf07_precision": float(p),
            "M_conf07_ci_lo": float(lo),
            "M_conf07_ci_hi": float(hi),
            "M_conf07_n": int(main_stats["n"]),
            "verdict": verdict,
            "scored_df": df,
        }


def intersection_score(stats_list: list[dict]):
    """Compare LLM models on the SAME pseudo_ids; score only rows where ALL agree on M @ CONF>=0.7."""
    if len(stats_list) < 2:
        return None
    base = stats_list[0]["scored_df"]
    base_M = base[(base["final"] == "Medium") & (base["conf"] >= 0.7)][["pseudo_id", "true_label"]]
    inter_ids = set(base_M["pseudo_id"])
    for s in stats_list[1:]:
        df = s["scored_df"]
        m = df[(df["final"] == "Medium") & (df["conf"] >= 0.7)]["pseudo_id"]
        inter_ids &= set(m)
    keys = base_M[base_M["pseudo_id"].isin(inter_ids)]
    n = len(keys)
    if n == 0:
        return None
    n_correct = int((keys["true_label"] == "Medium").sum())
    p, lo, hi = wilson_ci(n_correct, n)
    print(f"\n=== INTERSECTION (all unanimous M @ CONF>=0.7) ===")
    print(f"n_unanimous: {n}  n_correct: {n_correct}")
    print(f"precision: {p:.4f}  CI [{lo:.4f}, {hi:.4f}]")
    return {"n": n, "n_correct": n_correct, "precision": float(p), "ci_lo": float(lo), "ci_hi": float(hi)}


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["haiku"]
    keys = pd.read_csv(OUT / "eval_keys.csv")
    print(f"loaded {len(keys)} eval keys")
    print(f"baseline true-M on critical rows: {(keys['true_label'] == 'Medium').mean():.4f}")

    stats_list = []
    for model in models:
        try:
            s = score_one(model, keys)
            stats_list.append(s)
        except SystemExit as e:
            print(f"WARN: {e}")

    inter = None
    if len(stats_list) > 1:
        inter = intersection_score(stats_list)

    summary = {
        "n_critical_rows": len(keys),
        "baseline_true_M": float((keys["true_label"] == "Medium").mean()),
        "models": [{k: v for k, v in s.items() if k != "scored_df"} for s in stats_list],
        "intersection": inter,
    }
    out = OUT / f"results_{'_'.join(models)}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
