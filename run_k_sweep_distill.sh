#!/bin/bash
#SBATCH --job-name=k_sweep_distill
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -x

# Source conda
source /ibex/user/wux0b/miniforge/etc/profile.d/conda.sh
conda activate onn

# Check environment
nvidia-smi

# ==========================================
# K-sweep for Hardware-Aware Distillation
#
# Sweeps num_shared_weights (K) to measure how
# FC capacity affects accuracy under the full
# multi-MZI calibration with thermal crosstalk
# and per-processor power constraints.
# ==========================================

# === Hardware-aware training settings ===
export VOLTAGE_LR_MULT=10.0    # voltage param lr = args.lr * 10
export VOLTAGE_CLAMP_V=6.0     # per-MZI voltage clamp (V)
export POWER_BUDGET_MW=50.0    # per-processor thermal power limit (mW)

# === Sweep configuration ===
K_LIST="5 10 15 20 30 45 58"
HIDDEN_CHANNELS=4
EPOCHS=15
LR=0.01
BATCH_SIZE=200

# === Distillation settings ===
KD_ALPHA=0.5
KD_BETA=1.0
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "K-sweep Hardware-Aware Distillation"
echo "K values: $K_LIST"
echo "Channels: $HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS"
echo "LR: $LR (voltage: x${VOLTAGE_LR_MULT})"
echo "Power budget: ${POWER_BUDGET_MW} mW"
echo "Voltage clamp: +/-${VOLTAGE_CLAMP_V} V"
echo "KD: alpha=$KD_ALPHA, beta=$KD_BETA"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Setup data link if needed
if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

# Create logs directory
mkdir -p logs

for K in $K_LIST; do
    echo ""
    echo ">>> Starting K=$K"

    # Unique save suffix for this K value
    export SAVE_SUFFIX="_hw_aware_K${K}"

    # Find a free port for distributed training
    MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train_distill.py \
        --num-shared-weights $K \
        --hidden-channels $HIDDEN_CHANNELS \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --lossy-mzi \
        --detection-mode power \
        --fc-activation-mode linear \
        --save-model

    echo ">>> Finished K=$K"
done

echo ""
echo "=========================================="
echo "K-sweep completed."
echo "Results saved as distilled_shared_K{K}_ch${HIDDEN_CHANNELS}${SAVE_SUFFIX}_best.pt"
echo "=========================================="
