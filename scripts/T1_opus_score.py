"""T1-opus — score opus's responses against TRAIN OOF ground truth.

Compares opus's FINAL labels and CONF to the true labels in eval_keys.csv.
Reports:
  - Overall accuracy
  - Precision of high-CONF M verdicts (the metric that matters for the
    H->M override decision rule)
  - Comparison vs bank-only baseline (which would say M for every row).

Usage:
  python scripts/T1_opus_score.py <opus_response_text_file>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ART = Path("scripts/artifacts")
T1_OPUS = ART / "T1_opus"

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
            "opus_final": m.group(4).strip().title(),
            "opus_conf": float(m.group(5)),
            "opus_reason": m.group(6).strip(),
        })
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        # default location
        path = T1_OPUS / "response.txt"
    else:
        path = Path(sys.argv[1])
    text = path.read_text()
    parsed = parse(text)
    print(f"parsed {len(parsed)} ROW blocks from {path}")

    keys = pd.read_csv(T1_OPUS / "eval_keys.csv")
    print(f"eval_keys: {len(keys)} rows")

    df = parsed.merge(keys, on="pseudo_id", how="inner")
    print(f"matched: {len(df)} rows\n")

    # Overall stats
    print(f"opus FINAL distribution: {df['opus_final'].value_counts().to_dict()}")
    print(f"opus CONF mean={df['opus_conf'].mean():.3f} median={df['opus_conf'].median():.3f}")
    print(f"true label distribution: {df['true_label'].value_counts().to_dict()}\n")

    # Bank-only baseline: would say M for every row
    bank_only_acc = (df["true_label"] == "Medium").mean()
    print(f"BANK-ONLY (always M) accuracy: {bank_only_acc:.4f} ({int((df['true_label'] == 'Medium').sum())}/{len(df)})")

    # Opus accuracy
    correct = (df["opus_final"] == df["true_label"]).sum()
    print(f"OPUS accuracy: {correct / len(df):.4f} ({correct}/{len(df)})")
    print()

    # The decision-rule metric: of opus's M verdicts (the ones that would TRIGGER an
    # H->M flip on the test side), what fraction are actually true-M?
    for thr in [0.0, 0.7, 0.8, 0.9]:
        sub = df[(df["opus_final"] == "Medium") & (df["opus_conf"] >= thr)]
        if len(sub):
            p_M = (sub["true_label"] == "Medium").mean()
            n_correct_M = int((sub["true_label"] == "Medium").sum())
            print(f"opus says M, CONF>={thr}: {len(sub)} rows -> P(true=M)={p_M:.4f} ({n_correct_M}/{len(sub)})")
        else:
            print(f"opus says M, CONF>={thr}: 0 rows")
    print()

    # Conversely, if opus says H, how often is true=H?
    for thr in [0.0, 0.7, 0.8]:
        sub = df[(df["opus_final"] == "High") & (df["opus_conf"] >= thr)]
        if len(sub):
            p_H = (sub["true_label"] == "High").mean()
            n_correct_H = int((sub["true_label"] == "High").sum())
            print(f"opus says H, CONF>={thr}: {len(sub)} rows -> P(true=H)={p_H:.4f} ({n_correct_H}/{len(sub)})")

    # Verdict
    sub_m = df[(df["opus_final"] == "Medium") & (df["opus_conf"] >= 0.7)]
    if len(sub_m):
        p_M_07 = (sub_m["true_label"] == "Medium").mean()
        print(f"\n=== VERDICT ===")
        print(f"opus M @ CONF>=0.7 precision: {p_M_07:.4f} on {len(sub_m)} rows")
        print(f"bank-only baseline:           {bank_only_acc:.4f}")
        print(f"required to clear test-side break-even: 0.92")
        delta = p_M_07 - bank_only_acc
        print(f"opus delta over bank-only: {delta:+.4f}")
        if p_M_07 >= 0.92:
            print("--> OPUS CLEARS test-side break-even. Mechanism may have signal.")
        elif p_M_07 >= 0.88:
            print("--> OPUS in marginal zone. Could clear after T6 haircut, risky.")
        elif p_M_07 > bank_only_acc + 0.02:
            print("--> OPUS adds modest signal but still below break-even.")
        else:
            print("--> OPUS does NOT meaningfully exceed bank-only floor. T1 closes for opus too.")

    # Save scored df
    df.to_csv(T1_OPUS / "scored.csv", index=False)
    print(f"\nsaved scored: {T1_OPUS / 'scored.csv'}")

    # JSON summary
    summary = {
        "n": len(df),
        "opus_acc": float(correct / len(df)),
        "bank_only_acc": float(bank_only_acc),
        "opus_M_conf07_precision": float(p_M_07) if len(sub_m) else None,
        "n_opus_says_M_conf07": len(sub_m),
        "verdict": (
            "OPUS_CLEARS" if (len(sub_m) and p_M_07 >= 0.92)
            else "OPUS_MARGINAL" if (len(sub_m) and p_M_07 >= 0.88)
            else "OPUS_MODEST_LIFT" if (len(sub_m) and p_M_07 > bank_only_acc + 0.02)
            else "OPUS_NO_LIFT"
        ),
    }
    (T1_OPUS / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"results: {T1_OPUS / 'results.json'}")


if __name__ == "__main__":
    main()
