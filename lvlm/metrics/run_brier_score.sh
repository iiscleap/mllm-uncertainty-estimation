#!/bin/bash

# This script calculates the ECE of the FESTA-derived confidence score 
# for multiple LVLM models, datasets, and normalization methods.

MODELS=("gemma3" "llava" "qwenvl" "phi4")
DATASETS=("blink" "vsr")

# Define and clear output file at the start of the run
OUTPUT_FILE="brier_results_final.txt"
> "$OUTPUT_FILE"

echo "Starting final Brier Score calculation run..."

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running Brier Score for $model on $dataset"
        
        python brier_score.py \
            --model "$model" \
            --dataset "$dataset" >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $dataset" >> "$OUTPUT_FILE"
        fi
    done
done

echo "Final Brier Score calculation finished. Results are in $OUTPUT_FILE"
