#!/bin/bash
#SBATCH --job-name=no_distill_k15
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -x

# Source conda environment
source /ibex/user/wux0b/miniforge/etc/profile.d/conda.sh
conda activate onn

# Check GPU status
nvidia-smi

# ==========================================
# Configuration: No Distillation (Standard Training)
# Input Noise + Weight Noise + NO KD
# ==========================================
K_VAL=30
SIGMA_INPUT=0.15
SIGMA_WEIGHT=0.001
KD_ALPHA=0.0  # Disable Output Distillation
KD_BETA=0.0   # Disable Feature Distillation

# Voltage optimizer settings for lossy hardware-aware model
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
EPOCHS=20
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Job: No Distillation / Standard Training (K=$K_VAL)"
echo "Input Noise: $SIGMA_INPUT"
echo "Weight Noise: $SIGMA_WEIGHT (Voltage Error)"
echo "Distillation: DISABLED (Alpha=$KD_ALPHA, KD_Beta=$KD_BETA)"
echo "=========================================="

# Setup data link if needed
if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

# Set distillation parameters via environment variables
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA
export SAVE_SUFFIX="_no_distill"

# Find a free port for distributed training
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Run Training
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_distill.py \
    --num-shared-weights $K_VAL \
    --input-noise-sigma $SIGMA_INPUT \
    --weight-noise-sigma $SIGMA_WEIGHT \
    --epochs $EPOCHS \
    --batch-size 200 \
    --lr 0.01 \
    --lossy-mzi \
    --detection-mode power \
    --fc-activation-mode linear

echo ">>> No Distillation Job Finished"
