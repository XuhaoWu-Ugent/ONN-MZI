#!/bin/bash
#SBATCH --job-name=mesh_r20
#SBATCH --time=0-12:00:00
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
# Mesh-size scaling outlook experiment: r=20 FC mesh
#
# Hypothesis under test:
#   Scaling the FC mesh from r=10 to r=20 reduces the number of input
#   slices from N=58 to N=29 (2x fewer aggregation steps) and gives
#   each slice 4x richer cross-input mixing capacity (Clements MZI
#   count grows ~r²/2: r=10 needs 45, r=20 needs 190). This tests
#   whether the per-slice receptive-field bottleneck is meaningful
#   in power mode.
#
# Configuration:
#   CNN: unchanged (r=10, mzi_row_num=5, mzi_column_num=4, repeat=5)
#     — CNN is constrained by 3x3 patch -> 10 ports; cannot scale.
#   FC:  r=20, fc_mzi_row_num=10, fc_mzi_column_num=9, fc_repeat_num=10
#     — Clements-equivalent expressivity: 200 MZIs per processor
#       (vs 50 at r=10). Total FC MZI = 2 paths × K × 200.
#   K=15 sharing kept (matches production config).
#   Detection: power (matches paper's hardware target).
#
# Outlook framing (NOT for hardware deployment):
#   - Calibration recycles same 50 physical MZI entries via index%50;
#     no new fab data needed.
#   - Crosstalk silently disabled (loader only attaches when
#     channel.total_mzis matches num_physical=50; r=20 processors are
#     200 MZIs each so won't receive crosstalk). Ideal-MZI behaviour.
#   - Output projection: nn.Linear(20, 10, bias=False) auto-created
#     by OpticalSharedLinear, adds 200 electronic params.
#   - Net effect: this experiment tells us the IDEAL ceiling of
#     scaling mesh r — useful for paper's "outlook" / "future work"
#     section, not for current hardware deployment claim.
#
# Pass/fail criterion (vs r=10 K=15 baseline = 86.95%):
#   acc >= 89%   => mesh size IS a meaningful axis; continue scaling
#                  (r=40 next, or pursue mesh in paper's outlook).
#   acc 87~88%   => mesh scaling gives marginal returns; the per-slice
#                  receptive field is not a major bottleneck.
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
export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}_fcclean_meshr20"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Mesh r=20 outlook (CNN r=10 + FC r=20, K=$K_VALUE)"
echo "alpha=$KD_ALPHA, beta=$KD_BETA, K=$K_VALUE, ch=$HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "FC mesh: row=10, col=9, repeat=10 -> 200 MZIs/proc, r=20"
echo "Slices: N=29 (vs 58 at r=10)"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "Baseline to beat (r=10, K=15, fixed agg): 86.95%"
echo "Note: outlook experiment, not hardware-aware (crosstalk off, calibration cycles 50 MZIs)"
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
      --fc-mzi-row-num 10 \
      --fc-mzi-column-num 9 \
      --fc-mzi-repeat-num 10 \
      --save-model

echo ">>> Mesh r=20 outlook run finished."
