#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --job-name=SpLiCE_oracle_upper
#SBATCH --array=0-1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/%x_%A_%a.out
#SBATCH --error=./output/%x_%A_%a.err

set -euo pipefail

# Privileged upper-bound battery:
# task 0 = matched SimCLR baseline
# task 1 = SimCLR + same-target/opposite-spurious oracle relations
# Metadata is used only by task 1 and only as an intentional upper bound.

PROJECT_DIR="/home/xar68reb/Spur_SpLiCE"
CONDA_ENV="grgrie-train"
DATA_FOLDER="./datasets"
DATASET="waterbirds"
MODEL="resnet18_large"
SEED="0"
EPOCHS="500"
BATCH_SIZE="256"
NUM_WORKERS="5"
LEARNING_RATE="0.01"
WEIGHT_DECAY="1e-4"
TEMP="0.05"
ORACLE_WEIGHT="0.1"
PRINT_FREQ="25"
LINEAR_PROBE_FREQ="25"
LINEAR_PROBE_EPOCHS="100"
CHECKPOINT_DIR="outputs/oracle_upper_bound"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
    0)
        RUN_LABEL="Baseline"
        SPLICE_ARGS=(--splice_mode none)
        ;;
    1)
        RUN_LABEL="OracleRelational"
        SPLICE_ARGS=(--splice_mode oracle_relational --splice_weight "${ORACLE_WEIGHT}")
        ;;
    *)
        echo "Unsupported array task ${TASK_ID}" >&2
        exit 2
        ;;
esac

cd "${PROJECT_DIR}"
mkdir -p ./output "${CHECKPOINT_DIR}"

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "=== SpLiCE oracle relational upper bound ==="
echo "task=${TASK_ID} label=${RUN_LABEL} seed=${SEED} epochs=${EPOCHS} weight=${ORACLE_WEIGHT}"

python -u spur_splice.py \
    --dataset "${DATASET}" \
    --data_folder "${DATA_FOLDER}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_decay_epochs auto \
    --weight_decay "${WEIGHT_DECAY}" \
    --temp "${TEMP}" \
    --print_freq "${PRINT_FREQ}" \
    --rank_eval_freq 0 \
    --train_set_linear_layer ds_train \
    --linear_eval_split val \
    --linear_probe_mode periodic \
    --linear_probe_freq "${LINEAR_PROBE_FREQ}" \
    --linear_probe_epochs "${LINEAR_PROBE_EPOCHS}" \
    --linear_lr_decay_epochs auto \
    --linear_spurious_probe true \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --keep_checkpoints \
    --save_freq 50 \
    "${SPLICE_ARGS[@]}"

echo "Finished task=${TASK_ID} label=${RUN_LABEL}; $(date --iso-8601=seconds)"
conda deactivate
