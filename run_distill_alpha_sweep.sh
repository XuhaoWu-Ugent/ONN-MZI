#!/bin/bash
#SBATCH --job-name=alpha_sweep
#SBATCH --time=2-00:00:00
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

# Constants
K_VALUE=30
HIDDEN_CHANNELS=4  # Number of CNN filters (was 12, now 4 for debugging)
FIXED_SIGMA=0.15  # Choose a challenging noise level
KD_ALPHA_LIST="0.1 0.3 0.5 0.7 0.9"

# Voltage optimizer settings for lossy hardware-aware model
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Distillation Alpha Sweep (K=$K_VALUE, Channels=$HIDDEN_CHANNELS, Sigma=$FIXED_SIGMA)"
echo "Alphas: $KD_ALPHA_LIST"
echo "=========================================="

# Setup data link if needed
if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

for KD_ALPHA in $KD_ALPHA_LIST; do
  echo ""
  echo ">>> Starting Distillation with Alpha=$KD_ALPHA (Sigma=$FIXED_SIGMA)"

  # Find a free port using Python
  MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

  # Pass params via environment variables
  export DISTILL_ALPHA=$KD_ALPHA
  export SAVE_SUFFIX="_alpha${KD_ALPHA}_sigma${FIXED_SIGMA}"

  torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train_distill.py \
        --num-shared-weights $K_VALUE \
        --hidden-channels $HIDDEN_CHANNELS \
        --input-noise-sigma $FIXED_SIGMA \
        --epochs 20 \
        --batch-size 200 \
        --lr 0.01 \
        --save-model

  echo ">>> Finished Alpha=$KD_ALPHA. Model saved with suffix $SAVE_SUFFIX"
done

echo "Alpha sweep completed."
