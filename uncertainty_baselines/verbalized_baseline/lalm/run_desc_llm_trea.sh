#!/bin/bash

echo "Starting DESC_LLM TREA experiments..."
echo "=================================="

echo "Setting up environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate irl_torch2

# Run all tasks sequentially
tasks=("count" "duration" "order")

for task in "${tasks[@]}"; do
    echo ""
    echo "Running DESC_LLM on $task task..."
    echo "Time: $(date)"
    echo "--------------------------------"
    
    python desc_llm_trea.py \
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
echo "All DESC_LLM TREA tasks completed!"
echo "Results saved in lalm_results/ directory"
echo "=================================="
