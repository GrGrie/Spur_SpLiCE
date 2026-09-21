"""Machine-specific paths and identities, each overridable through an environment variable.

This module is the single Python source for cluster locations and the W&B identity. The bash
counterpart is scripts/load_splice_cluster_env.sh, which exports the same variables on the cluster.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

# Scratch location on the university cluster. Historical artifact URIs record this literal path,
# so provenance matching uses it even when SPUR_SPLICE_SCRATCH_ROOT points elsewhere.
CLUSTER_SCRATCH_ROOT = PurePosixPath("/scratch/xar68reb/CoSpRo")
CLUSTER_DATA_FOLDER = PurePosixPath("/home/xar68reb/Datasets")

DATA_FOLDER_ENV = "DATA_FOLDER"
WANDB_ENTITY_ENV = "WANDB_ENTITY"
DEFAULT_WANDB_ENTITY = "gsgrechkin-rptu"


def wandb_entity() -> str:
    return os.environ.get(WANDB_ENTITY_ENV) or DEFAULT_WANDB_ENTITY


def data_folder(default: str | Path | None = None) -> str | None:
    """Dataset root from DATA_FOLDER, else ``default``."""

    value = os.environ.get(DATA_FOLDER_ENV)
    if value:
        return value
    return None if default is None else str(default)
