#!/bin/bash

# This script calculates the AUC of the FESTA-derived confidence score 
# for multiple LVLM models, datasets, and normalization methods.

MODELS=("gemma3" "llava" "qwenvl" "phi4")
DATASETS=("blink" "vsr")

# Define and clear output file at the start of the run
OUTPUT_FILE="auc_results_final.txt"
> "$OUTPUT_FILE"

echo "Starting final AUC calculation run..."

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running AUC for $model on $dataset"
        
        python get_auroc_universal.py \
            --model "$model" \
            --dataset "$dataset" >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $dataset" >> "$OUTPUT_FILE"
        fi
    done
done

echo "Final AUC calculation finished. Results are in $OUTPUT_FILE"
