#!/bin/bash

# This script calculates toggle-based uncertainty for multiple models and datasets.

MODELS=("gemma3" "llava" "phi4" "qwenvl" "pixtral")
DATASETS=("blink" "vsr")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running uncertainty calculation for Model: $model, Dataset: $dataset"
        python calculate_toggle.py --model "$model" --dataset "$dataset"
        if [ $? -ne 0 ]; then
            echo "Error running script for Model: $model, Dataset: $dataset"
            exit 1
        fi
    done
done

echo "All uncertainty calculations are complete. Results are in rephrase_uncertainty_results_lvlm.txt"
