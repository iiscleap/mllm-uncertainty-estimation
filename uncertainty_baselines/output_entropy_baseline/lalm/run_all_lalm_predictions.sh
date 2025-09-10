#!/bin/bash

# Batch script to run all LALM model predictions for output entropy baseline
# Usage: ./run_all_lalm_predictions.sh

set -e  # Exit on any error

echo "Starting LALM prediction generation for output entropy baseline..."
echo "=================================================="

# Define arrays for iteration
models=("qwen" "desc_llm" "salmonn")
tasks=("count" "duration" "order")
exp_types=("orig" "neg")
methods=("vanilla" "sampling" "perturb" "perturb_sampling")

# Function to run a single prediction script
run_prediction() {
    local model=$1
    local task=$2
    local exp_type=$3
    local method=$4
    
    echo "Running: ${model} - ${task} - ${exp_type} - ${method}"
    
    cd "${model}"
    
    # Check if script exists
    script_name="${model}_${method}.py"
    if [ ! -f "$script_name" ]; then
        echo "Warning: Script $script_name not found, skipping..."
        cd ..
        return
    fi
    
    # Run the script with error handling
    if python "$script_name" --task "$task" --exp_type "$exp_type"; then
        echo "✓ Completed: ${model} - ${task} - ${exp_type} - ${method}"
    else
        echo "✗ Failed: ${model} - ${task} - ${exp_type} - ${method}"
    fi
    
    cd ..
    echo "---"
}

# Main execution loop
for model in "${models[@]}"; do
    echo "Processing model: $model"
    
    if [ ! -d "$model" ]; then
        echo "Warning: Directory $model not found, skipping..."
        continue
    fi
    
    for task in "${tasks[@]}"; do
        for exp_type in "${exp_types[@]}"; do
            for method in "${methods[@]}"; do
                run_prediction "$model" "$task" "$exp_type" "$method"
                
                # Add small delay to prevent overwhelming the system
                sleep 2
            done
        done
    done
    
    echo "Completed all predictions for $model"
    echo "=================================================="
done

echo "All LALM prediction generation completed!"
echo "Check individual model directories for output files."
