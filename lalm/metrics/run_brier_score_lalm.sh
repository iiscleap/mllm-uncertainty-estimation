#!/bin/bash

# This script calculates the Brier Score of the FESTA-derived confidence score 
# for multiple LALM models and tasks.

MODELS=("desc_llm" "qwen" "salmonn")
TASKS=("count" "duration" "order")

# Define and clear output file at the start of the run
OUTPUT_FILE="brier_results_lalm.txt"
> "$OUTPUT_FILE"

echo "Starting LALM Brier Score calculation run..."

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running Brier Score for $model on $task"
        
        python brier_score.py \
            --model "$model" \
            --dataset "$task" >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $task" >> "$OUTPUT_FILE"
        fi
    done
done

echo "LALM Brier Score calculation finished. Results are in $OUTPUT_FILE"
