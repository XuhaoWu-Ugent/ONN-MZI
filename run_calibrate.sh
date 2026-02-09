#!/bin/bash
#SBATCH --job-name=onn_calibrate
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -x

# 1. Source conda environment
source /ibex/user/wux0b/miniforge/etc/profile.d/conda.sh
conda activate onn

# 2. Check GPU status
nvidia-smi

# 3. Ensure directories exist
mkdir -p logs
mkdir -p results

# 4. Setup data link (Optional: adjust according to Ibex path)
# [ ! -d "data" ] && ln -s /home/wux0b/ONN-MZI/data data

# 5. Run Calibration Training
python train.py 
    --data-dir data 
    --epochs 50 
    --batch-size 128 
    --learning-rate 0.01 
    --output-file results/mzi_parameters.json

echo ">>> Calibration Job Finished"
