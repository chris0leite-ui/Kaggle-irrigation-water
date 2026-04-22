"""Snapshot scripts/artifacts/catboost_optuna.log into a JSON checkpoint.

Used to persist trial history mid-run so recovery is possible if the
process is killed or the container recycles. Parses the trial params
+ tuned_bal lines emitted by catboost_optuna.py.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path


LOG = Path("logs/catboost_optuna.log")
OUT = Path("scripts/artifacts/catboost_optuna_progress.json")


def main() -> None:
    if not LOG.exists():
        print(f"no log at {LOG}")
        sys.exit(1)

    trials: dict[int, dict] = {}
    phase = "phase1"
    best_params = None
    started_at = None
    last_event_at = None

    # trial-start: "[HH:MM:SS]   trial N: {...}"
    rx_start = re.compile(
        r"^\[(\d\d:\d\d:\d\d)\]\s+trial\s+(\d+):\s+(\{.*\})\s*$"
    )
    # trial-end: "[HH:MM:SS]   trial N: tuned_bal=0.9xxx (iters [...], fold_std=...)"
    rx_end = re.compile(
        r"^\[(\d\d:\d\d:\d\d)\]\s+trial\s+(\d+):\s+tuned_bal=([0-9.]+)\s+"
        r"\(iters\s+(\[[^\]]*\]),\s+fold_std=([0-9.]+)\)\s*$"
    )
    rx_best = re.compile(
        r"^\[(\d\d:\d\d:\d\d)\]\s+best_trial=(\d+)\s+tuned=([0-9.]+)"
    )

    with LOG.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if started_at is None and line.startswith("["):
                started_at = line[1:9]
            if line.startswith("["):
                last_event_at = line[1:9]

            if "phase 1 done" in line:
                phase = "between_phases"
            if "phase2:" in line:
                phase = "phase2"
            if "done" == line.strip().split("] ")[-1].strip():
                phase = "complete"

            m = rx_start.match(line)
            if m:
                t = int(m.group(2))
                params_str = m.group(3)
                # Python dict repr -> parseable JSON
                params_json = params_str.replace("'", '"')
                try:
                    params = json.loads(params_json)
                except json.JSONDecodeError:
                    params = {"_raw": params_str}
                trials.setdefault(t, {})["params"] = params
                trials[t]["started_at"] = m.group(1)
                continue

            m = rx_end.match(line)
            if m:
                t = int(m.group(2))
                trials.setdefault(t, {})
                trials[t]["tuned_bal"] = float(m.group(3))
                iters = json.loads(m.group(4))
                trials[t]["best_iters"] = iters
                trials[t]["fold_std"] = float(m.group(5))
                trials[t]["ended_at"] = m.group(1)
                continue

            m = rx_best.match(line)
            if m:
                best_trial = int(m.group(2))
                best_val = float(m.group(3))
                # we'll attach best_params from this trial if present
                if best_trial in trials:
                    best_params = trials[best_trial].get("params")

    completed = [t for t, v in trials.items() if "tuned_bal" in v]
    if completed:
        ranked = sorted(completed, key=lambda t: trials[t]["tuned_bal"],
                        reverse=True)
        best_trial_num = ranked[0]
        best_tuned = trials[best_trial_num]["tuned_bal"]
        best_params = trials[best_trial_num].get("params")
    else:
        best_trial_num = None
        best_tuned = None

    out = {
        "snapshot_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": str(LOG),
        "started_at": started_at,
        "last_event_at": last_event_at,
        "phase": phase,
        "n_trials_started": len(trials),
        "n_trials_completed": len(completed),
        "trials": [
            {
                "number": t, **v,
            }
            for t, v in sorted(trials.items())
        ],
        "best_trial_so_far": best_trial_num,
        "best_tuned_bal_so_far": best_tuned,
        "best_params_so_far": best_params,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")
    print(f"trials started={len(trials)}  completed={len(completed)}  "
          f"phase={phase}")
    if best_tuned is not None:
        print(f"best-so-far: trial {best_trial_num}  "
              f"tuned_bal={best_tuned:.5f}")


if __name__ == "__main__":
    main()
