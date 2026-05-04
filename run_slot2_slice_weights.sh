#!/bin/bash
#SBATCH --job-name=slot2_slicewts
#SBATCH --time=0-08:00:00
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
# SLOT 2: Learnable slice aggregation diagnostic
#
# Hypothesis under test:
#   The fixed `Σ y_i / √N` aggregation across N=58 slices smooths out
#   discriminative signal. Replacing it with `Σ s_i · y_i` (58 trainable
#   scalars, init at 1/√N to reproduce baseline at step 0) lets the
#   model learn per-slice importance.
#
# Configuration:
#   K=58 (no weight sharing — every slice has its own physical processor)
#     so the aggregation is the only shared component being tested.
#   OPTICAL_FC_SLICE_WEIGHTS=1 enables the new learnable aggregation
#     in module/optical_linear_shared.py.
#
# Pass/fail criterion:
#   acc >= 90%   => slice aggregation is a real bottleneck; next step
#                  is to upgrade to (N, 10) per-slice-per-port weights
#                  (580 params, equivalent to a hidden electronic layer).
#   acc ≈ 88.5%  => aggregation is NOT the bottleneck; rule out and move
#                  attention to slot 1 result and mesh scaling.
#
# Baseline to beat (K=58, fixed aggregation, otherwise identical):
#   88.46% (from clean_k-46750580.out K=58 run)
# ==========================================
K_VALUE=58
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA=0.0
KD_BETA=0.0
EPOCHS=30

export OPTICAL_FC_CLEAN=1
export OPTICAL_FC_SLICE_WEIGHTS=1
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
export POWER_BUDGET_MW=50.0
export ONECYCLE_PCT_START=0.1
export ONECYCLE_FINAL_DIV_FACTOR=10
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA
export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}_K${K_VALUE}_fcclean_slicewts"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "SLOT 2: Learnable slice aggregation (K=$K_VALUE)"
echo "alpha=$KD_ALPHA, beta=$KD_BETA, K=$K_VALUE, ch=$HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "OPTICAL_FC_SLICE_WEIGHTS=$OPTICAL_FC_SLICE_WEIGHTS  <-- new env hook"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "Baseline to beat (K=58, fixed aggregation): 88.46%"
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
      --fc-activation-mode linear \
      --save-model

echo ">>> SLOT 2 (learnable slice aggregation) finished."
