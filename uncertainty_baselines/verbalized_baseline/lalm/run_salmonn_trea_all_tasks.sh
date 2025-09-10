#!/bin/bash

echo "Starting SALMONN TREA experiments..."
echo "=================================="


echo "Setting up environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate salmonn

export CUDA_VISIBLE_DEVICES=6
# Run all tasks sequentially
tasks=("duration" "order")

for task in "${tasks[@]}"; do
    echo ""
    echo "Running SALMONN on $task task..."
    echo "Time: $(date)"
    echo "--------------------------------"
    
    python salmonn_trea.py \
        --task $task \
        --output_dir lalm_results \
        --dataset_path /home/debarpanb/VLM_project/TREA_dataset
    
    if [ $? -eq 0 ]; then
        echo "✓ $task task completed successfully"
    else
        echo "✗ $task task failed"
        exit 1
    fi
    echo "--------------------------------"
done

echo ""
echo "All SALMONN TREA tasks completed!"
echo "Results saved in lalm_results/ directory"
echo "=================================="
