#!/bin/bash
#SBATCH --job-name=v5e_fromscratch
#SBATCH --array=1-20%4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# 20 from-scratch multi-MZI fits (one per random seed) for the Fig.2
# training-curve band (mean +/- std over seeds).
#
# SLURM array: 20 tasks, "%4" caps it at 4 RUNNING AT ONCE -> matches the
# 4-GPU allocation. Each task gets its OWN GPU (--gres=gpu:v100:1), so there is
# NO GPU contention (running 4 fits on a single shared GPU was >5x slower).
# 5 waves x ~2.2h/fit ~= ~11h wall. Each fit is ~2.2-3h (80 epochs x ~95s on a
# free V100); --time padded to 24h so even a much slower node never hits the limit.
#
# Reproduces results/v5e_fromscratch_training_log.json, varying only --seed.
# Per-run output: results/v5e_run<seed>_log.json (per-epoch train_loss/val_loss).
# Pull those 20 JSONs back to the workstation to build the figure band.
#
# PREREQUISITE (NOT in the git repo, copy it over first, ~47 MB):
#   scp <workstation>/Mesh_measurements_multi/parsed_dataset_grouped.pt \
#       <ibex>:<repo>/Mesh_measurements_multi/parsed_dataset_grouped.pt
# Submit from the repo root:  sbatch run_v5e_fromscratch_ibex.sh

set -e
set -x

# --- environment (edit to match your ibex setup; copied from run_distill_full.sh)
source /ibex/user/wux0b/miniforge/etc/profile.d/conda.sh
conda activate onn
nvidia-smi

mkdir -p logs results

DATA=Mesh_measurements_multi                  # dir holding the dataset cache
CACHE=$DATA/parsed_dataset_grouped.pt
if [ ! -f "$CACHE" ]; then
    echo "ERROR: dataset cache not found at $CACHE" >&2
    echo "       scp parsed_dataset_grouped.pt (~47 MB) from the workstation first." >&2
    exit 1
fi

SEED=$SLURM_ARRAY_TASK_ID
echo ">>> v5e from-scratch multi-MZI fit, seed=$SEED"

python -u train_multi.py \
    --mesh-dir "$DATA" \
    --cache-path "$CACHE" \
    --init-params results/mzi_parameters.json \
    --measured-physics results/mzi_measured_physics.json \
    --enable-crosstalk \
    --lambda-phi0 0 \
    --epochs 80 \
    --lr 0.005 \
    --seed "$SEED" \
    --val-jsons 18,19 \
    --log-history "results/v5e_run${SEED}_log.json" \
    --output-file "results/_tmp_v5e_run${SEED}.json"

echo ">>> seed=$SEED done -> results/v5e_run${SEED}_log.json"
