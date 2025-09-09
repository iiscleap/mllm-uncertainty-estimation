#!/bin/bash

# This script calculates the AUCPR of the FESTA-derived confidence score 
# for multiple LVLM models, datasets, and normalization methods.

MODELS=("gemma3" "llava" "qwenvl" "phi4")
DATASETS=("blink" "vsr")

# Define a single output file
OUTPUT_FILE="aucpr_results_final.txt"
> "$OUTPUT_FILE"

echo "Starting final AUCPR calculation run..."

# Loop through all model and dataset combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running AUCPR for $model on $dataset"
        
        python kl_div_aucpr.py --model "$model" --dataset "$dataset" >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $dataset" >> "$OUTPUT_FILE"
        fi
    done
done

echo "Final AUCPR calculation finished. Results are in $OUTPUT_FILE"
