#!/bin/bash
#SBATCH --job-name=test_linfc
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -x

source /ibex/user/wux0b/miniforge/etc/profile.d/conda.sh
conda activate onn

nvidia-smi

# ==========================================
# Scout run: CNN uses 10x10 MZI mesh, FC uses electronic nn.Linear
#   - Isolates FC layer choice vs OpticalSharedLinear
#   - Same alpha=0, beta=0 (pure CE) for comparability with
#     run_test_alpha0_beta0.sh
#   - Same final_div_factor=10 (experiment B)
# ==========================================
K_VALUE=15
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA=0.0
KD_BETA=0.0
EPOCHS=30

export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
export POWER_BUDGET_MW=50.0
export ONECYCLE_PCT_START=0.1
export ONECYCLE_FINAL_DIV_FACTOR=10
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA
export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}_linfc_flr10"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Scout: Linear FC, CNN MZI mesh preserved"
echo "alpha=$KD_ALPHA, beta=$KD_BETA, K=$K_VALUE (unused for FC), ch=$HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "Sigma: $FIXED_SIGMA"
echo "CNN Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "FC: Electronic nn.Linear(576, 10)"
echo "=========================================="

if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

mkdir -p logs

MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node=$NUM_GPUS \
      --master_port=$MASTER_PORT \
      train_distill.py \
      --num-shared-weights $K_VALUE \
      --hidden-channels $HIDDEN_CHANNELS \
      --input-noise-sigma $FIXED_SIGMA \
      --epochs $EPOCHS \
      --batch-size 200 \
      --lr 0.01 \
      --lossy-mzi \
      --detection-mode power \
      --no-optical-fc \
      --save-model

echo ">>> Linear-FC scout run finished."
