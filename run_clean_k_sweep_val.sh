#!/bin/bash
#SBATCH --job-name=clean_k_val
#SBATCH --time=3-00:00:00
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
# K sweep with Clean Optical FC and a VALIDATION SPLIT (re-run of
# run_clean_k_sweep.sh for SI Fig. S2c, revision 2026-08).
#
# Differences from run_clean_k_sweep.sh:
#   * --val-split 5000: 5000 images are held out from the 60k MNIST training
#     set (fixed permutation, seed 42). The best checkpoint of each run is
#     selected on VALIDATION accuracy; the 10k test set is only reported.
#   * Per-K summary line in the log:
#       [Final] K=.. val_split=5000 best_val_acc=..% at epoch .. |
#               test_acc_at_best_val=..% | last_epoch_test_acc=..%
#     -> parse with parse_k_sweep_val_log.py to build the Fig. S2c table.
#   * Checkpoints get the suffix _val5000 so the original ones are untouched.
# Everything else (K list, epochs, lr, sigma schedule, env vars) is identical.
# ==========================================

K_LIST="1 3 5 8 10 12 15 20 25 30 40 58"
VAL_SPLIT=5000   # images held out from the 60k MNIST training set for model selection
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

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "=========================================="
echo "Clean FC K Sweep WITH VALIDATION SPLIT (alpha=beta=0, ch=$HIDDEN_CHANNELS, val=$VAL_SPLIT)"
echo "K values: $K_LIST  (N=58 for ch=4)"
echo "Epochs: $EPOCHS (pct_start=$ONECYCLE_PCT_START, final_div=$ONECYCLE_FINAL_DIV_FACTOR)"
echo "OPTICAL_FC_CLEAN=$OPTICAL_FC_CLEAN"
echo "Sigma: $FIXED_SIGMA"
echo "Voltage: lr_mult=$VOLTAGE_LR_MULT, clamp=+/-${VOLTAGE_CLAMP_V}V, power=${POWER_BUDGET_MW}mW"
echo "=========================================="

if [ ! -d "../data" ]; then
    ln -s /home/wux0b/ONN-MZI/data ../data
fi

mkdir -p logs

for K in $K_LIST; do
    echo ""
    echo ">>> Starting Clean-FC K=$K (val-split $VAL_SPLIT)"

    MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    export SAVE_SUFFIX="_alpha${KD_ALPHA}_beta${KD_BETA}_K${K}_fcclean_flr10_val${VAL_SPLIT}"

    torchrun --nproc_per_node=$NUM_GPUS \
          --master_port=$MASTER_PORT \
          train_distill.py \
          --num-shared-weights $K \
          --hidden-channels $HIDDEN_CHANNELS \
          --input-noise-sigma $FIXED_SIGMA \
          --epochs $EPOCHS \
          --batch-size 200 \
          --lr 0.01 \
          --lossy-mzi \
          --detection-mode power \
          --fc-activation-mode linear \
          --val-split $VAL_SPLIT \
          --save-model

    echo ">>> Finished K=$K"
done

echo ""
echo "Clean-FC K sweep (validation split) completed."
