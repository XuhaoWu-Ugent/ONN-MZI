#!/bin/bash
#SBATCH --job-name=clean_beta
#SBATCH --time=2-00:00:00
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
# Beta sweep with Clean Optical FC, alpha fixed at 0.3.
# After alpha sweep identifies the best alpha, re-run this with the
# winning alpha by editing KD_ALPHA_FIXED below.
# Default alpha=0.3 is the historical best for the original (legacy)
# architecture — treat as a reasonable starting point.
# ==========================================
K_VALUE=15
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA_FIXED=0
KD_BETA_LIST="0.0 0.01 0.05 0.1 0.2 0.3 0.5 0.75 1.0"
EPOCHS=30

export OPTICAL_FC_CLEAN=1
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
export POWER_BUDGET_MW=50.0
export ONECYCLE_PCT_START=0.1
export ONECYCLE_FINAL_DIV_FACTOR=10
export DISTILL_ALPHA=$KD_ALPHA_FIXED

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Clean FC Beta Sweep (alpha=$KD_ALPHA_FIXED, K=$K_VALUE, ch=$HIDDEN_CHANNELS)"
echo "Betas: $KD_BETA_LIST"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "=========================================="

if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

mkdir -p logs

for KD_BETA in $KD_BETA_LIST; do
    echo ""
    echo ">>> Starting Clean-FC Distillation beta=$KD_BETA (alpha=$KD_ALPHA_FIXED)"

    MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    export DISTILL_BETA=$KD_BETA
    export SAVE_SUFFIX="_alpha${KD_ALPHA_FIXED}_beta${KD_BETA}_fcclean_flr10"

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

    echo ">>> Finished beta=$KD_BETA"
done

echo ""
echo "Clean-FC beta sweep completed."
