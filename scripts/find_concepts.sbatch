#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -J find_concepts_celebA
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared # Set the GPU into shared mode, so that multiple processes can run on it
#SBATCH -o ./output/%x_%j.out
#SBATCH -e ./output/%x_%j.err

# ——— hyperparameters ———
DATASET="celebA"
DATA_FOLDER="/home/xar68reb/Datasets"
SPLIT="train"
OUT_PATH="outputs/celebA_splice_concepts.json"
TOP_K=10
# ——— End of hyperparameters ———


echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================="

# Load environment
module purge
module load miniforge3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate grgrie-train

cd /home/xar68reb/Spur_SpLiCE

echo "========== SYSTEM INFO =========="
nvidia-smi || echo "nvidia-smi failed"
echo "================================="

echo "========== PYTORCH INFO =========="
python - <<EOF
import os, torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("cudnn enabled:", torch.backends.cudnn.enabled)
print("cudnn version:", torch.backends.cudnn.version())
print("Device:", torch.cuda.get_device_name(0))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
EOF
echo "================================="



echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Running training with arguments:
	--DATASET = 	$DATASET
	--DATA_FOLDER = $DATA_FOLDER
	--SPLIT = 		$SPLIT
	--OUT_PATH = 	$OUT_PATH
	--TOP_K = 		$TOP_K
  "
 
python tools/discover_splice_spurious_concepts.py \
	--dataset "$DATASET" \
	--data_folder "$DATA_FOLDER" \
	--split "$SPLIT" \
	--out_path "$OUT_PATH" \
	--top_k "$TOP_K"

conda deactivate
