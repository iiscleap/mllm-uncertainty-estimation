#!/bin/bash

# This script calculates the ECE of the FESTA-derived confidence score 
# for multiple LVLM models, datasets, and normalization methods.

MODELS=("gemma3" "llava" "qwenvl" "phi4")
DATASETS=("blink" "vsr")

# Define and clear output file at the start of the run
OUTPUT_FILE="risk_coverage_results_final.txt"
> "$OUTPUT_FILE"

echo "Starting final Risk–Coverage calculation run..."

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running Risk–Coverage for $model on $dataset"
        
        python risk_coverage.py \
            --model "$model" \
            --dataset "$dataset" \
            --abstention_step 0.1 >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $dataset" >> "$OUTPUT_FILE"
        fi
    done
done

echo "Final Risk–Coverage calculation finished. Results are in $OUTPUT_FILE"
