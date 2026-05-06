#!/bin/bash
#SBATCH --job-name=cl_r_sweep
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
# ConstrainedLinearFC mesh-size sweep (r = 10 → 580)
#
# What this experiment measures:
#   The LOSSLESS MATHEMATICAL CEILING of scaling the FC's per-slice
#   mesh size r from 10 (current production) up to r=576/580
#   (single-shot, processing the entire CNN output in one pass).
#
#   We use ConstrainedLinearFC — direct nn.Parameter + softmax-bounded
#   T_pos / T_neg WITHOUT any MZI-mesh forward — so the result is
#   independent of mesh topology. It only enforces the two physical
#   constraints any power-mode optical FC must satisfy:
#     (a) non-negativity of T_pos and T_neg          (power mode)
#     (b) column-sum = 1 of each T  (lossless limit; real meshes have
#         col-sum ≤ 1 due to insertion loss, leaks to non-read ports,
#         and back-reflection)
#
#   These constraints are r-INDEPENDENT (per-input properties), so
#   mesh-size scaling alone cannot remove them.
#
# What this experiment does NOT measure:
#   The ACTUAL accuracy a real photonic chip would achieve at r=580.
#   Our row-column + Redheffer recirculating mesh has linear-in-r MZI
#   count (~5.5r per processor at repeat=5), whereas full Clements
#   needs r²/2. At large r the row-col topology cannot realise every
#   doubly-substochastic 10×r matrix. So a real implementation falls
#   below this ceiling by an unknown topology-dependent margin.
#
#   Treat ConstrainedLinear's accuracy here as an UPPER BOUND on what
#   mesh-size scaling can achieve, regardless of topology choice.
#
# Predicted shape of the curve:
#   - r=10:    87-89%  (K=15 sharing applies, 4 slices/block)
#   - r=20:    88-90%  (K=15 sharing, 2 slices/block)
#   - r=40:    K=15 caps to n_slices ⇒ no sharing, knee around here
#   - r ≥ 80:  plateau ~90-91%
#   - r=580:   single-shot; ~91% (ceiling)
#   - vs nn.Linear: ~95.34% (gap of ~3-4 pt is physical, not topological)
#
# Implication for paper outlook:
#   No mesh-size scaling can close the residual ~3-4 pt gap to
#   nn.Linear. That gap is structural — non-negativity + column-sum —
#   and requires either a different detection mode (coherent), optical
#   amplification (breaks col-sum), or an electronic decoder layer.
#
# Configuration:
#   K=15 fixed (caps to n_slices when r ≥ 40). The sweep therefore
#   shows TWO effects fused into one curve: (i) reduced K-sharing
#   penalty as r increases, (ii) larger receptive field per slice.
#
#   Epochs: 15 (compressed from 30; ConstrainedLinear converges by
#   epoch ~10 in our local r=10 run that hit 89.09%).
#
# Reference points (already measured, all on Power CNN + ch=4):
#   86.95% — Optical FC K=15 (mesh, sliced r=10 baseline)
#   88.46% — Optical FC K=58 (mesh, no sharing)
#   89.09% — ConstrainedLinear K=15 r=10 (math limit, local 30 ep)
#   95.34% — nn.Linear (unconstrained electronic FC)
# ==========================================

R_VALUES=(10 20 40 80 160 290 580)
K_VALUE=15
HIDDEN_CHANNELS=4
FIXED_SIGMA=0.15
KD_ALPHA=0.0
KD_BETA=0.0
EPOCHS=15

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

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI-MZIparameters/data ../data
fi

mkdir -p logs

for R in "${R_VALUES[@]}"; do
    MESH_ROW=$((R / 2))
    MESH_COL=$((MESH_ROW - 1))
    if [ "$MESH_COL" -lt 1 ]; then MESH_COL=1; fi

    export SAVE_SUFFIX="_clrsweep_r${R}_K${K_VALUE}"

    MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    echo "=========================================="
    echo ">>> Starting ConstrainedLinear sweep r=$R"
    echo "    mesh row=$MESH_ROW, col=$MESH_COL (gives FC r=$R)"
    echo "    K_VALUE=$K_VALUE (caps to n_slices=ceil(576/$R))"
    echo "    Epochs=$EPOCHS, sigma=$FIXED_SIGMA"
    echo "=========================================="

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
          --fc-mzi-row-num $MESH_ROW \
          --fc-mzi-column-num $MESH_COL \
          --fc-mzi-repeat-num 5 \
          --save-model

    echo ">>> Finished r=$R"
done

echo ""
echo "=========================================="
echo ">>> Full ConstrainedLinear r-sweep finished."
echo "    Reference points (for analysis):"
echo "      86.95% — Optical FC K=15 (mesh, sliced r=10)"
echo "      88.46% — Optical FC K=58 (no sharing)"
echo "      89.09% — ConstrainedLinear K=15 r=10 (math limit, local 30 epochs)"
echo "      95.34% — nn.Linear FC (unconstrained)"
echo "    See logs for each r=$(echo ${R_VALUES[@]}) run's best acc."
echo "=========================================="
