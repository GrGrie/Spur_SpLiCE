"""Backward-compatible entry point; use generate_cospro_concept_groups."""

from scripts.tools.generate_cospro_concept_groups import *  # noqa: F401,F403
from scripts.tools.generate_cospro_concept_groups import main


if __name__ == "__main__":
    main()
