#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --job-name=SpLiCE_CRP_v2_train
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./slurm-%x_%j.out
#SBATCH --error=./slurm-%x_%j.err

set -euo pipefail

# This file is intentionally self-contained. Edit values here if the cluster
# account, dataset location, or training budget changes; no shell exports are required.
PROJECT_DIR="/home/xar68reb/Spur_SpLiCE"
CONDA_ENV="grgrie-train"
DATA_FOLDER="./datasets"
DATASET="waterbirds"
TEACHER_GRAPH="outputs/crp/waterbirds/teacher_graph_seed0.pt"
CHECKPOINT_DIR="outputs/crp_v2_training"

MODEL="resnet18_large"
SEED="0"
EPOCHS="500"
BATCH_SIZE="256"
NUM_WORKERS="5"
LEARNING_RATE="0.01"
WEIGHT_DECAY="1e-4"
SIMCLR_TEMPERATURE="0.05"
CRP_WEIGHT="0.1"
CRP_TEMPERATURE="0.1"
CRP_START_EPOCH="10"
CRP_WARMUP_EPOCHS="10"
PRINT_FREQ="25"
LINEAR_PROBE_FREQ="25"
LINEAR_PROBE_EPOCHS="100"
WANDB_PROJECT="Spur_SpLiCE"
WANDB_ENTITY="gsgrechkin-rptu"
WANDB_RUN_NAME="${DATASET}_S${SEED}_CRPv2_w${CRP_WEIGHT}_e${EPOCHS}"
WANDB_GROUP="${DATASET}_crp_v2_e${EPOCHS}_seed${SEED}"
WANDB_TAGS="dataset_${DATASET},seed_${SEED},family_crp_v2,label_free,machine_cluster,protocol_e${EPOCHS}"

cd "${PROJECT_DIR}"
mkdir -p ./output "${CHECKPOINT_DIR}"

if [[ ! -f "${TEACHER_GRAPH}" ]]; then
    echo "CRP teacher graph not found: ${TEACHER_GRAPH}" >&2
    echo "Run: bash scripts/Submit-SpLiCE_CRP_v2_pipeline.sh" >&2
    exit 2
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== SpLiCE-CRP v2 relational student training ==="
echo "dataset=${DATASET} graph=${TEACHER_GRAPH} seed=${SEED} epochs=${EPOCHS}"
echo "crp_weight=${CRP_WEIGHT} crp_temperature=${CRP_TEMPERATURE} start=${CRP_START_EPOCH} warmup=${CRP_WARMUP_EPOCHS}"

python -u spur_splice.py \
    --use_wandb \
    --wandb_name "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --entity "${WANDB_ENTITY}" \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_tags "${WANDB_TAGS}" \
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
    --temp "${SIMCLR_TEMPERATURE}" \
    --print_freq "${PRINT_FREQ}" \
    --rank_eval_freq 0 \
    --splice_mode crp_relational \
    --splice_weight "${CRP_WEIGHT}" \
    --crp_teacher_graph "${TEACHER_GRAPH}" \
    --crp_temperature "${CRP_TEMPERATURE}" \
    --crp_start_epoch "${CRP_START_EPOCH}" \
    --crp_warmup_epochs "${CRP_WARMUP_EPOCHS}" \
    --train_set_linear_layer ds_train \
    --linear_eval_split val \
    --linear_probe_mode periodic \
    --linear_probe_freq "${LINEAR_PROBE_FREQ}" \
    --linear_probe_epochs "${LINEAR_PROBE_EPOCHS}" \
    --linear_lr_decay_epochs auto \
    --linear_spurious_probe true \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --keep_checkpoints \
    --save_freq 50

echo "Finished CRP v2 student training; $(date --iso-8601=seconds)"
conda deactivate
