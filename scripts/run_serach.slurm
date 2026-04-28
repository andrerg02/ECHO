#!/bin/bash
#SBATCH --job-name=flatnsd-diam-search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:10
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p results

source /home/parentd1/.bashrc
conda activate ECHO

export WANDB__SERVICE_WAIT=300
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

OUTDIR="./results"

TASK=diam
LOGGER=wandb
ENTITY_NAME="***"

MODELS="--models flatnsd_gp"
SCHEDULER="--scheduler none"
N_SAMPLES="--n_samples 100"
OUTPUT_DIR="--output_dir ${OUTDIR}"
EPOCHS="--max_epochs 50"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

echo "Detected ${NUM_GPUS} GPU(s)."

COMMON_ARGS=" ${MODELS} ${SCHEDULER} ${N_SAMPLES} --num_gpus=${NUM_GPUS} ${OUTPUT_DIR} ${EPOCHS}"

python download-all.py --root ./ --task "${TASK}"

python search.py ${COMMON_ARGS} \
  --tasks "${TASK}" \
  --logger "${LOGGER}" \
  --entity_name "${ENTITY_NAME}"