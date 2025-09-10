#!/bin/bash

# Script to run Qwen Audio model on all TREA tasks and calculate AUROC

echo "=== Running Qwen Audio on TREA Dataset ==="
echo "Working directory: $(pwd)"

# Tasks to run
TASKS=("count" "duration" "order")

# Run inference for each task
for TASK in "${TASKS[@]}"; do
    echo ""
    # echo "=== Processing $TASK task ==="
    # python qwen_audio_trea.py --task $TASK
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed $TASK task inference"
        
        # Calculate AUROC
        echo "Calculating AUROC for $TASK task..."
        python get_auroc_trea.py --task $TASK --model qwen
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully calculated AUROC for $TASK task"
        else
            echo "✗ Failed to calculate AUROC for $TASK task"
        fi
    else
        echo "✗ Failed to complete $TASK task inference"
    fi
    
    echo "----------------------------------------"
done

echo ""
echo "=== All tasks completed ==="
echo "Results saved in lalm_results/ directory:"
ls -la lalm_results/qwen_guess_prob_trea_*.csv 2>/dev/null || echo "No TREA result files found"
