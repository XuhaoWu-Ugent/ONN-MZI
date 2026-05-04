#!/bin/bash
#SBATCH --job-name=slot1_coherent
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
# SLOT 1: Coherent FC diagnostic
#
# Hypothesis under test:
#   The 6.5 pt gap from CNN+nn.Linear (95%) to Clean FC K=58 ceiling
#   (88.46%) is dominated by the power-mode non-negative weight
#   constraint. Switching detection_mode 'power' -> 'coherent' lets a
#   single optical path carry signed weights via complex amplitude
#   interference, removing the dual-path differential bandaid.
#
# Pass/fail criterion:
#   acc >= 91%   => non-negativity is the dominant bottleneck;
#                  next step is coherent + larger K / mesh.
#   acc 88~90%   => non-negativity not main cause; check slot 2 result
#                  (slice aggregation) and consider mesh scaling (slot 3).
#
# All other knobs identical to the 86.95% baseline (run_test_fc_clean.sh)
# so any acc shift is attributable to the detection mode.
# ==========================================
K_VALUE=15
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA=0.0
KD_BETA=0.0
EPOCHS=30

export OPTICAL_FC_CLEAN=1
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
export POWER_BUDGET_MW=50.0
export ONECYCLE_PCT_START=0.1
export ONECYCLE_FINAL_DIV_FACTOR=10
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA
export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}_fcclean_coherent"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "SLOT 1: Coherent FC (test non-negativity hypothesis)"
echo "alpha=$KD_ALPHA, beta=$KD_BETA, K=$K_VALUE, ch=$HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "Detection: coherent (only diff vs run_test_fc_clean.sh)"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "Baseline to beat (power mode, same config): 86.95%"
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
      --detection-mode coherent \
      --fc-activation-mode linear \
      --save-model

echo ">>> SLOT 1 (coherent FC) finished."
