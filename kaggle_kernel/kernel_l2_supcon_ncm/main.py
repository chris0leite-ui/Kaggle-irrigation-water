"""L2 SupCon-NCM kernel entry orchestrator."""
from __future__ import annotations

import time


def main() -> None:
    start_t = time.time()
    run_pipeline(start_t)


if __name__ == "__main__":
    main()
