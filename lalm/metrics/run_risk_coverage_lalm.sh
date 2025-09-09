#!/bin/bash

# This script calculates the Risk-Coverage of the FESTA-derived confidence score 
# for multiple LALM models and tasks.

MODELS=("desc_llm" "qwen" "salmonn")
TASKS=("count" "duration" "order")

# Define and clear output file at the start of the run
OUTPUT_FILE="risk_coverage_results_lalm.txt"
> "$OUTPUT_FILE"

echo "Starting LALM Risk–Coverage calculation run..."

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running Risk–Coverage for $model on $task"
        
        python risk_coverage.py \
            --model "$model" \
            --dataset "$task" \
            --abstention_step 0.1 >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $task" >> "$OUTPUT_FILE"
        fi
    done
done

echo "LALM Risk–Coverage calculation finished. Results are in $OUTPUT_FILE"
