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
# SLOT 2: Learnable slice aggregation diagnostic (CORRECTED, K=15)
#
# Hypothesis under test:
#   With K=15 weight sharing, slices i, i+15, i+30, i+45 are forced
#   through the SAME physical T matrix. The fixed equal-weight sum
#   `Σ y_i / √N` then gives every slice the same importance even when
#   they carry different amounts of discriminative signal. Replacing
#   it with `Σ s_i · y_i` (58 trainable scalars, init 1/√N) lets each
#   slice be weighted independently — something the shared T cannot
#   express on its own.
#
#   PRIOR ATTEMPT (incorrect): ran this at K=58. There each slice has
#   its own independent T_i, so s_i is functionally absorbed by T_i in
#   most of the dynamic range. Result was +0.3 pt (within noise) —
#   that experiment tested the mechanism in the regime where it has
#   no leverage. The correct test is at K=15.
#
# Pass/fail criterion (vs K=15 fixed-agg baseline = 86.95%):
#   acc >= 88%   => slice aggregation IS a bottleneck under sharing;
#                  upgrade to (N, 10) per-slice-per-port weights worth
#                  trying as a follow-up.
#   acc ≈ 87%    => aggregation is not a bottleneck even when T is
#                  shared; rule out and stop pursuing this direction.
#
# Baseline to beat (K=15, fixed aggregation, otherwise identical):
#   86.95% (from test_fcclean-46632148.out)
# ==========================================
K_VALUE=15
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
echo "SLOT 2 (CORRECTED): Learnable slice aggregation under K=$K_VALUE sharing"
echo "alpha=$KD_ALPHA, beta=$KD_BETA, K=$K_VALUE, ch=$HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "OPTICAL_FC_SLICE_WEIGHTS=$OPTICAL_FC_SLICE_WEIGHTS  <-- new env hook"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "Baseline to beat (K=15, fixed aggregation): 86.95%"
echo "Mechanism leverage: at K=15 each T is shared by 4 slices, so per-slice"
echo "  scaling cannot be absorbed by T (unlike K=58)."
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
