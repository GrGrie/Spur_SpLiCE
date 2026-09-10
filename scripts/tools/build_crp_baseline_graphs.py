"""Backward-compatible entry point; use build_cospro_baseline_graphs."""

from scripts.tools.build_cospro_baseline_graphs import *  # noqa: F401,F403
from scripts.tools.build_cospro_baseline_graphs import main


if __name__ == "__main__":
    main()
