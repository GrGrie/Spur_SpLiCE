#!/usr/bin/env bash
#SBATCH --time=00:05:00
#SBATCH -J Splice_Env_Check
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuidle
#SBATCH --gpu_cmode=shared # Set the GPU into shared mode, so that multiple processes can run on it
#SBATCH -o ./output/%x_%j.out
#SBATCH -e ./output/%x_%j.err

set -euo pipefail

echo "== Host and job =="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not_in_slurm}"
echo "PWD=$PWD"


module purge
module load miniforge3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh


#echo "== Activate conda env =="
#if [ -f "$HOME/.bashrc" ]; then
#  # shellcheck disable=SC1090
#  source "$HOME/.bashrc"
#fi

#if command -v conda >/dev/null 2>&1; then
#  eval "$(conda shell.bash hook)"
#else
#  echo "ERROR: conda command not found after sourcing ~/.bashrc"
#  exit 1
#fi

conda activate grgrie-train

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"
echo "PATH=$PATH"
which python
which pip || true
python --version
python -m pip --version

echo
echo "== Installed package metadata =="
python -m pip show huggingface-hub || true
python -m pip show open-clip-torch || true
python -m pip show torch || true
python -m pip show torchvision || true
python -m pip show pandas || true
python -m pip show wandb || true

echo
echo "== Python import check =="
python - <<'PY'
import importlib
import sys

print("python executable:", sys.executable)
print("python version:", sys.version.replace("\n", " "))

packages = [
    "torch",
    "torchvision",
    "open_clip",
    "huggingface_hub",
    "pandas",
    "wandb",
]

for name in packages:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        location = getattr(module, "__file__", "built-in")
        print(f"OK {name}: version={version} file={location}")
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        raise
PY

echo
echo "== CUDA check =="
python - <<'PY'
import torch

print("torch cuda available:", torch.cuda.is_available())
print("torch cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch current device:", torch.cuda.current_device())
    print("torch device name:", torch.cuda.get_device_name(0))
PY

echo
echo "== open_clip Hugging Face checkpoint check =="
python - <<'PY'
import open_clip

print("Creating open_clip ViT-B-32 with pretrained laion2b_s34b_b79k...")
model = open_clip.create_model("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
print("OK open_clip checkpoint loaded")
print("model class:", type(model).__name__)
PY

echo
echo "== SpLiCE package check =="
python - <<'PY'
import splice

print("splice module:", splice.__file__)
print("available models:", splice.available_models())
vocab = splice.get_vocabulary("laion")
print("laion vocab length:", len(vocab))
print("laion vocab tail sample:", vocab[-5:])
PY

echo
echo "All checks completed."
