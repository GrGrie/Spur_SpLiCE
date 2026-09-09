#!/usr/bin/env bash

# Shared cluster environment bootstrap for the active SpLiCE launchers.
# The important detail is that all Python calls use CONDA_PREFIX/bin/python,
# rather than whichever `python` happens to be first on PATH in a Slurm shell.

: "${PROJECT_DIR:?PROJECT_DIR must be set before loading the SpLiCE environment}"
SPLICE_CONDA_ENV="${SPLICE_CONDA_ENV:-grgrie-train}"
SPUR_SPLICE_ARTIFACT_ROOT="${SPUR_SPLICE_ARTIFACT_ROOT:-/scratch/xar68reb/CoSpRo}"

mkdir -p "${SPUR_SPLICE_ARTIFACT_ROOT}"
if [[ ! -w "${SPUR_SPLICE_ARTIFACT_ROOT}" ]]; then
    echo "ERROR: artifact root is not writable: ${SPUR_SPLICE_ARTIFACT_ROOT}" >&2
    exit 2
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${SPLICE_CONDA_ENV}"

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: expected Python interpreter does not exist: ${PYTHON_BIN}" >&2
    exit 2
fi

export SPLICE_CONDA_ENV PYTHON_BIN SPUR_SPLICE_ARTIFACT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-5}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

print(f"[env] python={sys.executable}")
print(f"[env] CONDA_PREFIX={os.environ.get('CONDA_PREFIX', '')}")
if importlib.util.find_spec("sklearn") is None:
    raise SystemExit(
        "ERROR: sklearn is unavailable in the Python environment used by this Slurm job. "
        f"Install it into {sys.executable}, not another Python installation."
    )
import sklearn
print(f"[env] sklearn={sklearn.__version__} ({sklearn.__file__})")
PY
