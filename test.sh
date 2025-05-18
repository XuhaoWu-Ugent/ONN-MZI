#!/bin/bash

# experiment.sh
# Make sure this script has execution permission: chmod +x experiment.sh

# Create log directory
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# Get current timestamp as experiment identifier
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define parameter combinations to test
HIDDEN_CHANNELS=(16 32 64 128 256)
NUM_LAYERS=(3 4 5 6 7)
LEARNING_RATES=(0.0001 0.0005 0.001 0.005 0.01)
BATCH_SIZES=(256 512 1024 2048)
MZI_REPEAT_NUM=(3 4 5 6 7)
MZI_ROW_NUM=(3 4 5 6 7)
MZI_COLUMN_NUM=(3 4 5 6)
KERNEL_SIZES=(3 5 7)
INPUT_SIZES=(14 28 32)
GRAD_CLIPS=(0.5 1.0 1.5 2.0)

# Record experiment configuration
echo "Starting experiments at $TIMESTAMP" > "${LOG_DIR}/experiment_summary_${TIMESTAMP}.txt"

# Calculate total number of experiments
TOTAL_EXPERIMENTS=$((${#HIDDEN_CHANNELS[@]} * ${#NUM_LAYERS[@]} * ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#MZI_REPEAT_NUM[@]} * ${#MZI_ROW_NUM[@]} * ${#MZI_COLUMN_NUM[@]} * ${#KERNEL_SIZES[@]} * ${#INPUT_SIZES[@]} * ${#GRAD_CLIPS[@]}))
CURRENT_EXPERIMENT=0

# Error handling
set -e
trap 'echo "Error occurred. Exiting..."; exit 1' ERR

# Loop through different parameter combinations
for hidden in "${HIDDEN_CHANNELS[@]}"; do
    for layers in "${NUM_LAYERS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for batch in "${BATCH_SIZES[@]}"; do
                for mzi_repeat in "${MZI_REPEAT_NUM[@]}"; do
                    for mzi_row in "${MZI_ROW_NUM[@]}"; do
                        for mzi_col in "${MZI_COLUMN_NUM[@]}"; do
                            for kernel in "${KERNEL_SIZES[@]}"; do
                                for input_size in "${INPUT_SIZES[@]}"; do
                                    for grad_clip in "${GRAD_CLIPS[@]}"; do
                                        # Update and display progress
                                        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
                                        echo "Running experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS"
                                        
                                        # Create experiment identifier
                                        EXP_ID="h${hidden}_l${layers}_lr${lr}_b${batch}_mzir${mzi_repeat}_mzin${mzi_row}_mzic${mzi_col}_k${kernel}_i${input_size}_g${grad_clip}"
                                        
                                        # Log GPU status before experiment
                                        echo "GPU Status before experiment:" >> "${LOG_DIR}/${TIMESTAMP}_${EXP_ID}.log"
                                        nvidia-smi >> "${LOG_DIR}/${TIMESTAMP}_${EXP_ID}.log"
                                        
                                        echo "Running experiment with parameters:"
                                        echo "Hidden Channels: $hidden"
                                        echo "Num Layers: $layers"
                                        echo "Learning Rate: $lr"
                                        echo "Batch Size: $batch"
                                        echo "MZI Repeat Num: $mzi_repeat"
                                        echo "MZI Row Num: $mzi_row"
                                        echo "MZI Column Num: $mzi_col"
                                        echo "Kernel Size: $kernel"
                                        echo "Input Size: $input_size"
                                        echo "Gradient Clip: $grad_clip"
                                        
                                        # Run experiment and save output to log file
                                        python main.py \
                                            --hidden-channels $hidden \
                                            --num-layers $layers \
                                            --lr $lr \
                                            --batch-size $batch \
                                            --mzi-repeat-num $mzi_repeat \
                                            --mzi-row-num $mzi_row \
                                            --mzi-column-num $mzi_col \
                                            --kernel-size $kernel \
                                            --input-size $input_size \
                                            --grad-clip $grad_clip \
                                            --epochs 50 \
                                            --wandb \
                                            --seed 42 \
                                            2>&1 | tee -a "${LOG_DIR}/${TIMESTAMP}_${EXP_ID}.log"
                                        
                                        # Record experiment configuration to summary file
                                        echo "${EXP_ID}: hidden=${hidden}, layers=${layers}, lr=${lr}, batch=${batch}, mzi_repeat=${mzi_repeat}, mzi_row=${mzi_row}, mzi_col=${mzi_col}, kernel=${kernel}, input_size=${input_size}, grad_clip=${grad_clip}" >> "${LOG_DIR}/experiment_summary_${TIMESTAMP}.txt"
                                        
                                        # Log GPU status after experiment
                                        echo "GPU Status after experiment:" >> "${LOG_DIR}/${TIMESTAMP}_${EXP_ID}.log"
                                        nvidia-smi >> "${LOG_DIR}/${TIMESTAMP}_${EXP_ID}.log"
                                        
                                        # Wait for GPU to cool down
                                        sleep 10
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All experiments completed!"
