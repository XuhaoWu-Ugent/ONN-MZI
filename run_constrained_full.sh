#!/bin/bash
#SBATCH --job-name=cl_full
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
# ConstrainedLinearFC ablation — FULL constraints (mimics optical FC math)
#
# Hypothesis under test:
#   The 86.95% (Power CNN + Optical FC K=15) → 95.34% (Power CNN +
#   nn.Linear) gap = 8.39 pt. This is the cost of switching the FC,
#   under the same CNN. Decomposition:
#     (a) Math constraints: non-neg weights, col-sum (energy
#         conservation), K=15 sharing, block-structure slicing
#     (b) Mesh-specific: voltage→T parametrization, α^11 insertion
#         loss, thermal crosstalk
#
#   ConstrainedLinearFC implements (a) directly via softmax-bounded
#   raw nn.Parameters, skipping the mesh entirely. With the same
#   CNN, training, and dataset:
#     - acc ≈ 87% → math constraints alone cause the gap; mesh-
#                   specific factors don't matter. Paper story:
#                   "the 7pt gap is structural, not engineering."
#     - acc ≈ 92-95% → math constraints are loose; the mesh-specific
#                      factors carry most of the cost. Paper story:
#                      "theoretical ceiling is high; engineering
#                      mismatch is fixable."
#     - acc somewhere in between → mixed; need finer ablation
#
# Configuration:
#   CNN: identical to 86.95% baseline (power, K=15 by default,
#        but FC's K is what matters here. CNN is calibrated.)
#   FC:  ConstrainedLinearFC with all 3 math constraints on
#        (matches optical FC's reachable W_eff space, minus
#        mesh-specific factors).
#
# Ablation variants (rerun with these env vars to isolate which
# constraint matters):
#   - Drop col-sum:    CONSTRAINED_FC_COLSUM=0  (just non-neg + blocks + K=15)
#   - Drop non-neg:    CONSTRAINED_FC_NONNEG=0  (sliced + dual-path linear, ≈ block-Toeplitz nn.Linear with K=15)
#   - Drop blocks:     CONSTRAINED_FC_BLOCKS=0  (single 10×580 W with non-neg + col-sum)
#   - Drop sharing:    --num-shared-weights 58  (K=58, no sharing)
# ==========================================
K_VALUE=15
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA=0.0
KD_BETA=0.0
EPOCHS=30

export OPTICAL_FC_CLEAN=1
export USE_CONSTRAINED_FC=1
export CONSTRAINED_FC_NONNEG=1
export CONSTRAINED_FC_COLSUM=1
export CONSTRAINED_FC_BLOCKS=1
export VOLTAGE_LR_MULT=10.0
export VOLTAGE_CLAMP_V=6.0
export POWER_BUDGET_MW=50.0
export ONECYCLE_PCT_START=0.1
export ONECYCLE_FINAL_DIV_FACTOR=10
export DISTILL_ALPHA=$KD_ALPHA
export DISTILL_BETA=$KD_BETA
export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}_clfull"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "ConstrainedLinearFC ablation: ALL constraints (no MZI mesh)"
echo "alpha=$KD_ALPHA, beta=$KD_BETA, K=$K_VALUE, ch=$HIDDEN_CHANNELS"
echo "Epochs: $EPOCHS"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "USE_CONSTRAINED_FC=$USE_CONSTRAINED_FC"
echo "Constraints: nonneg=$CONSTRAINED_FC_NONNEG, colsum=$CONSTRAINED_FC_COLSUM, blocks=$CONSTRAINED_FC_BLOCKS"
echo "Reference points:"
echo "  Power CNN + Optical FC (K=15)  : 86.95%   ← same math, mesh-implemented"
echo "  Power CNN + nn.Linear FC       : 95.34%   ← no constraints"
echo "Sigma: $FIXED_SIGMA"
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

echo ">>> ConstrainedLinearFC full-constraints run finished."
