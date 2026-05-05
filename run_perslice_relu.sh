#!/bin/bash
#SBATCH --job-name=perslice_relu
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
# Per-slice ReLU diagnostic (cheapest nonlinearity test)
#
# Hypothesis under test:
#   The 88.5% power-mode ceiling is hit because the entire FC is linear:
#       hidden = Σ_i (T_pos_i - T_neg_i) · x_i   (signed but linear in x)
#   Replacing the post-aggregation diff with PER-SLICE rectification:
#       hidden = Σ_i ReLU((T_pos_i - T_neg_i) · x_i)
#   makes the single FC layer piecewise-linear (a real nonlinear function
#   of x) without adding a second optical layer or coherent detection.
#
#   In hardware terms, this requires per-slice differential PD readout
#   + small-signal electronic ReLU (one diode/comparator per slice),
#   then electrical summation. No optical-electrical-optical conversion
#   needed — output remains compatible with "all-optical encoder + PD
#   readout boundary" framing.
#
# Pass/fail criterion (vs K=15 fixed-agg baseline = 86.95%):
#   acc >= 90%   => nonlinearity IS the dominant remaining bottleneck;
#                  pursue this in paper (per-slice ReLU adds ~0 hardware,
#                  big accuracy boost).
#   acc 88~89%   => nonlinearity is meaningful but not the only thing;
#                  combine with slice_weights or mesh scaling.
#   acc ≈ 87%    => nonlinearity at this position doesn't help; the
#                  bottleneck is structural elsewhere (mesh capacity,
#                  sharing).
#
# Configuration: same as 86.95% baseline (K=15, σ=0.15, 30 ep, ch=4)
#   except OPTICAL_FC_PER_SLICE_RELU=1.
# ==========================================
K_VALUE=15
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA=0.0
KD_BETA=0.0
EPOCHS=30

export OPTICAL_FC_CLEAN=1
export OPTICAL_FC_PER_SLICE_RELU=1
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
export POWER_BUDGET_MW=50.0
export ONECYCLE_PCT_START=0.1
export ONECYCLE_FINAL_DIV_FACTOR=10
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA
export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}_fcclean_persliceReLU"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Per-slice ReLU diagnostic (nonlinearity ablation)"
echo "alpha=$KD_ALPHA, beta=$KD_BETA, K=$K_VALUE, ch=$HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "OPTICAL_FC_PER_SLICE_RELU=$OPTICAL_FC_PER_SLICE_RELU  <-- new env hook"
echo "Forward: hidden = Σ_i ReLU((T_pos_i − T_neg_i) · x_i)"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "Baseline to beat (K=15, no per-slice ReLU): 86.95%"
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

echo ">>> Per-slice ReLU diagnostic finished."
