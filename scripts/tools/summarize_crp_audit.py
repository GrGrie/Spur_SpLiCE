"""Backward-compatible entry point; use summarize_cospro_audit."""

from scripts.tools.summarize_cospro_audit import *  # noqa: F401,F403
from scripts.tools.summarize_cospro_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
