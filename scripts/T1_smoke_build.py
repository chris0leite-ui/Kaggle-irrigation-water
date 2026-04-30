"""Emit a 1-row and 5-row smoke prompt to disk for the Agent tool to use.

The orchestrator (main Claude session) reads these files and pastes
them to the haiku Agent. Output goes to scripts/artifacts/T1_smoke/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T1_format_batch import build_batch_prompt  # noqa: E402


def main():
    ART = Path("scripts/artifacts")
    DATA = Path("data")
    out_dir = ART / "T1_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    bord = pd.read_csv(ART / "T1_borderline_top500.csv")
    test = pd.read_csv(DATA / "test.csv")

    for n in (1, 5, 10, 50):
        sub = bord.head(n)
        p = build_batch_prompt(sub, test)
        out_path = out_dir / f"prompt_n{n}.txt"
        out_path.write_text(p)
        print(f"wrote {out_path}  chars={len(p)}  rows={n}")


if __name__ == "__main__":
    main()
