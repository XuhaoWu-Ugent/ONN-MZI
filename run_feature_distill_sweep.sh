#!/bin/bash
#SBATCH --job-name=feature_distill_sweep
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

# Configuration
K_VAL=30
HIDDEN_CHANNELS=4  # Number of CNN filters (was 12, now 4 for debugging)
SIGMA=0.15
KD_ALPHA_FIXED=0.3

# Voltage optimizer settings for lossy hardware-aware model
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
KD_BETAS=(0.0 0.1 0.3 0.5 1.0 5.0)
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Feature Distillation Sweep (K=$K_VAL, Channels=$HIDDEN_CHANNELS, Alpha=$KD_ALPHA_FIXED, Sigma=$SIGMA)"
echo "KD_Betas: ${KD_BETAS[*]}"
echo "=========================================="

# Setup data link if needed
if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

for KD_BETA in "${KD_BETAS[@]}"; do
    echo ""
    echo ">>> Starting Distillation with KD_Beta=$KD_BETA (Alpha=$KD_ALPHA_FIXED)"

    # Set environment variables for the python script
    export DISTILL_ALPHA=$KD_ALPHA_FIXED
    export DISTILL_BETA=$KD_BETA

    # Add a suffix to save separate model files
    export SAVE_SUFFIX="_alpha${KD_ALPHA_FIXED}_beta${KD_BETA}"
    
    # Find a free port using Python
    MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train_distill.py \
        --num-shared-weights $K_VAL \
        --hidden-channels $HIDDEN_CHANNELS \
        --input-noise-sigma $SIGMA \
        --epochs 20 \
        --batch-size 200 \
        --lr 0.01 \
        --lossy-mzi \
        --detection-mode power \
        --fc-activation-mode linear
        
    echo ">>> Finished KD_Beta=$KD_BETA"
done

echo ""
echo "=========================================="
echo "Feature Sweep Completed."
echo "=========================================="