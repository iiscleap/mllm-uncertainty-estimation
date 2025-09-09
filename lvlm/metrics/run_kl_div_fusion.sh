#!/bin/bash

# This script calculates the AUROC of the FESTA-derived confidence score 
# for multiple LVLM models, datasets, and normalization methods.

MODELS=("gemma3" "llava" "qwenvl" "phi4")
DATASETS=("blink" "vsr")

# Define a single output file
OUTPUT_FILE="auroc_kl_fusion_results_lvlm.txt"
> "$OUTPUT_FILE"

echo "Starting final AUROC calculation run..."

# Loop through all model and dataset combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running AUROC for $model on $dataset"
        
        python kl_div_fusion.py --model "$model" --dataset "$dataset" >> "$OUTPUT_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error running script for $model on $dataset" >> "$OUTPUT_FILE"
        fi
    done
done

echo "Final AUROC calculation finished. Results are in $OUTPUT_FILE"
