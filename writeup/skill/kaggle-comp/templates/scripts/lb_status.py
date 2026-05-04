"""Kaggle LB submissions status — authoritative source of truth.

Run this BEFORE recommending any submission CSV as an "unprobed" or
"highest-EV next probe" candidate.

Usage:
  python scripts/lb_status.py                    # full table sorted by date
  python scripts/lb_status.py <filename>          # check specific filename
  python scripts/lb_status.py --best              # show LB-best filename and score
  python scripts/lb_status.py --top N             # top N submissions by LB

Exit codes:
  0   filename present (warns if you intended to "probe" it as new)
  1   filename absent (safe to LB-probe with user approval)
  2   kaggle CLI error / network failure

CLAUDE.md rule: every LB-probe recommendation must verify the
candidate is NOT already in the submissions list. This script is the
single-line check.
"""
from __future__ import annotations

import re
import subprocess
import sys

COMP = "{{COMP_SLUG}}"


def fetch():
    try:
        out = subprocess.check_output(
            ["kaggle", "competitions", "submissions", COMP],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as e:
        print(f"kaggle CLI error: {e.output}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print("kaggle CLI not installed; pip install kaggle", file=sys.stderr)
        sys.exit(2)
    return out


def parse(out):
    """Parse the table.  Columns: fileName, date, description, status, publicScore.
    Returns list of dicts."""
    lines = out.strip().split("\n")
    rows = []
    # Find header and separator
    header_idx = None
    for i, line in enumerate(lines):
        if "fileName" in line and "publicScore" in line:
            header_idx = i
            break
    if header_idx is None:
        return rows
    # Skip header + separator (------)
    for line in lines[header_idx + 2:]:
        # Match score like 0.98140 anywhere; fileName is first token; status is "SubmissionStatus.X"
        if not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            continue
        fname = parts[0]
        # public score is the floating-point that ends the line (or just before privateScore)
        score_match = re.search(r"\b(0\.\d{4,6})\b\s*$", line) or re.search(
            r"\b(0\.\d{4,6})\b", parts[-1] if len(parts) >= 1 else "")
        score = float(score_match.group(1)) if score_match else None
        status = next((p for p in parts if "SubmissionStatus" in p), "?")
        rows.append({"file": fname, "score": score, "status": status,
                     "raw": line.strip()})
    return rows


def main():
    out = fetch()
    rows = parse(out)
    if not rows:
        print("No submissions found (or parse failed).")
        sys.exit(2)

    args = sys.argv[1:]
    if not args:
        # Full table sorted by score desc
        with_score = [r for r in rows if r["score"] is not None]
        with_score.sort(key=lambda r: r["score"], reverse=True)
        print(f"{'LB':<10} {'fileName'}")
        print("-" * 80)
        for r in with_score:
            print(f"{r['score']:<10.5f} {r['file']}")
        sys.exit(0)

    if args[0] == "--best":
        with_score = [r for r in rows if r["score"] is not None]
        if not with_score:
            print("No scored submissions.")
            sys.exit(2)
        best = max(with_score, key=lambda r: r["score"])
        print(f"LB-best: {best['score']:.5f}  {best['file']}")
        sys.exit(0)

    if args[0] == "--top":
        n = int(args[1]) if len(args) > 1 else 10
        with_score = [r for r in rows if r["score"] is not None]
        with_score.sort(key=lambda r: r["score"], reverse=True)
        for r in with_score[:n]:
            print(f"{r['score']:<10.5f} {r['file']}")
        sys.exit(0)

    # filename match
    target = args[0]
    # Strip directory and trailing whitespace
    target_base = target.split("/")[-1].strip()
    matches = [r for r in rows if r["file"] == target_base]
    if matches:
        for m in matches:
            score = f"{m['score']:.5f}" if m['score'] is not None else "?"
            print(f"FOUND: {m['file']}  LB={score}  {m['status']}")
        # Exit 0 — found means already probed
        sys.exit(0)
    else:
        print(f"NOT FOUND: {target_base}  (safe to recommend as LB-probe candidate)")
        sys.exit(1)


if __name__ == "__main__":
    main()
