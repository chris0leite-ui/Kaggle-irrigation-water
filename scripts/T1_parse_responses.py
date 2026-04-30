"""T1 — parse subagent ROW-block responses into structured labels.

Expected response format per row:
  ROW <test_id>
  RULE_SCORE: <int>
  RULE_PRED: <Low|Medium|High>
  FINAL: <Low|Medium|High>
  CONF: <0.0-1.0>
  REASON: <<=15 words>

Tolerates: extra whitespace, surrounding chatter, missing fields
(invalid rows are dropped with a warning), label-case variations.

Usage:
  python scripts/T1_parse_responses.py <response_text_file>
or import:
  from T1_parse_responses import parse_response_text
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

CLASS_NORM = {
    "low": "Low", "Low": "Low", "LOW": "Low",
    "medium": "Medium", "Medium": "Medium", "MEDIUM": "Medium",
    "med": "Medium", "Med": "Medium",
    "high": "High", "High": "High", "HIGH": "High",
}

ROW_RX = re.compile(
    r"ROW\s+(\d+)\s*\n"
    r"\s*RULE_SCORE:\s*(\d+)\s*\n"
    r"\s*RULE_PRED:\s*(\w+)\s*\n"
    r"\s*FINAL:\s*(\w+)\s*\n"
    r"\s*CONF:\s*([\d.]+)\s*\n"
    r"\s*REASON:\s*([^\n]*)",
    flags=re.IGNORECASE,
)


def parse_response_text(text: str) -> pd.DataFrame:
    rows = []
    for m in ROW_RX.finditer(text):
        tid = int(m.group(1))
        rule_score = int(m.group(2))
        rule_pred = CLASS_NORM.get(m.group(3).strip())
        final = CLASS_NORM.get(m.group(4).strip())
        try:
            conf = float(m.group(5))
        except ValueError:
            conf = float("nan")
        reason = m.group(6).strip()
        if rule_pred is None or final is None:
            continue
        rows.append({
            "test_id": tid,
            "rule_score": rule_score,
            "rule_pred": rule_pred,
            "llm_final": final,
            "llm_conf": conf,
            "llm_reason": reason,
        })
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/T1_parse_responses.py <text_file>")
        sys.exit(1)
    text = Path(sys.argv[1]).read_text()
    df = parse_response_text(text)
    print(f"parsed {len(df)} rows")
    print(df.head())
    print("\nllm_conf stats:")
    print(df["llm_conf"].describe())
    print("\nllm_final distribution:")
    print(df["llm_final"].value_counts())


if __name__ == "__main__":
    main()
