#!/usr/bin/env bash
#SBATCH --job-name=cospro-freeze-env
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/SLURM/freeze-env-%j.out
#SBATCH --error=outputs/SLURM/freeze-env-%j.err

# Records the exact package versions of the cluster training environment in environment/.
# The lock files synchronize through Git and document the environment behind every result.
#
#   sbatch scripts/freeze_environment.sbatch

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM environment
source scripts/load_splice_cluster_env.sh

conda env export --no-builds --name "${SPLICE_CONDA_ENV}" | grep -v '^prefix:' > environment/cluster-conda.lock.yml
"${PYTHON_BIN}" -m pip freeze --exclude-editable > environment/cluster-pip.lock.txt
"${PYTHON_BIN}" - <<'PY' > environment/cluster-runtime.json
import json, platform, torch
print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
}, indent=2))
PY
echo "[env] wrote environment/cluster-conda.lock.yml, environment/cluster-pip.lock.txt and environment/cluster-runtime.json"
