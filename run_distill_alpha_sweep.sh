#!/bin/bash
#SBATCH --job-name=alpha_sweep_b0
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

# ==========================================
# Alpha sweep with KD_BETA=0.0
#
# Previous alpha sweep was done at KD_BETA=1.0 which suppressed
# all configs to ~35-38%. Re-sweep at KD_BETA=0.0 (the optimal
# beta) to find the true best alpha.
# ==========================================
K_VALUE=15
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA_LIST="0.1 0.3 0.5 0.7 0.9"
KD_BETA=0.0
EPOCHS=30

# Hardware-aware training settings
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
export POWER_BUDGET_MW=50.0
export ONECYCLE_PCT_START=0.1
export DISTILL_BETA=$KD_BETA

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Alpha Sweep (KD_BETA=$KD_BETA, K=$K_VALUE, ch=$HIDDEN_CHANNELS)"
echo "Alphas: $KD_ALPHA_LIST"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START)"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "=========================================="

# Setup data link if needed
if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

# Create logs directory
mkdir -p logs

for KD_ALPHA in $KD_ALPHA_LIST; do
  echo ""
  echo ">>> Starting Distillation with Alpha=$KD_ALPHA (Beta=$KD_BETA, Sigma=$FIXED_SIGMA)"

  # Find a free port
  MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

  # Pass params via environment variables
  export DISTILL_ALPHA=$KD_ALPHA
  export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}"

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
        --fc-activation-mode linear \
        --save-model

  echo ">>> Finished Alpha=$KD_ALPHA"
done

echo "Alpha sweep (beta=0) completed."
