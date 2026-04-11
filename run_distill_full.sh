#!/bin/bash
#SBATCH --job-name=distill_full_k15
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
# Configuration: Hardware-Aware Distillation
#
# Key findings from hyperparameter search:
# - KD_BETA=0.0 (no feature MSE) dramatically outperforms KD_BETA>0
#   (63.95% vs ~35%) — feature_adapter's 295k params interfere with
#   voltage gradient signal
# - KD_ALPHA=0.3: loss = 0.7*CE + 0.3*KD (no feature MSE)
# - pct_start=0.1: shorter warmup, longer effective learning window
# - EPOCHS=30: previous 20ep runs showed lr anneal killing training
#   while acc was still climbing at epoch 18
# ==========================================
K_VAL=15
HIDDEN_CHANNELS=4
SIGMA_INPUT=0.15
SIGMA_WEIGHT=0.001
KD_ALPHA=0.7        # alpha sweep best: 0.7 > 0.5 > 0.3 > 0.1 > 0.9
KD_BETA=0.0         # no feature MSE loss — critical for hardware-aware training
EPOCHS=30
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# Hardware-aware training settings
export VOLTAGE_LR_MULT=10.0          # voltage lr = args.lr * 10 = 0.1
export VOLTAGE_CLAMP_V=6.0           # per-MZI voltage clamp (V)
export POWER_BUDGET_MW=50.0          # per-processor thermal power limit (mW)
export ONECYCLE_PCT_START=0.1        # shorter warmup (3ep), longer learning window (27ep)

echo "=========================================="
echo "Job: Hardware-Aware Distillation (K=$K_VAL, Channels=$HIDDEN_CHANNELS)"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START)"
echo "Input Noise: $SIGMA_INPUT"
echo "Weight Noise: $SIGMA_WEIGHT (Voltage Error)"
echo "Distillation: KD_Alpha=$KD_ALPHA, KD_Beta=$KD_BETA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "=========================================="

# Setup data link if needed
if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

# Create logs directory
mkdir -p logs

# Set distillation parameters via environment variables
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA
export SAVE_SUFFIX="_hw_aware"

# Find a free port for distributed training
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Run Training
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_distill.py \
    --num-shared-weights $K_VAL \
    --hidden-channels $HIDDEN_CHANNELS \
    --input-noise-sigma $SIGMA_INPUT \
    --weight-noise-sigma $SIGMA_WEIGHT \
    --epochs $EPOCHS \
    --batch-size 200 \
    --lr 0.01 \
    --lossy-mzi \
    --detection-mode power \
    --fc-activation-mode linear \
    --save-model

echo ">>> Hardware-Aware Distillation Finished"
