#!/bin/bash

# This script calculates the AUROC of the FESTA-derived confidence score 
# for multiple LALM models and tasks.

MODELS=("desc_llm" "qwen" "salmonn")
TASKS=("count" "duration" "order")

# Define a single output file
OUTPUT_FILE="auroc_kl_fusion_results_lalm.txt"
> "$OUTPUT_FILE"

echo "Starting LALM AUROC calculation run..."

# Loop through all model and task combinations
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running AUROC for $model on $task"
        
        python kl_div_fusion.py --model "$model" --task "$task" >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $task" >> "$OUTPUT_FILE"
        fi
    done
done

echo "LALM AUROC calculation finished. Results are in $OUTPUT_FILE"
