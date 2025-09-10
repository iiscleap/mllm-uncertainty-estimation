#!/bin/bash

# VLM Verbalized Confidence Runner with Conda Environment Management
# Runs all models on BLINK and VSR datasets sequentially and saves AUC results
# Each model uses its specific conda environment

set -e  # Exit on any error

# Configuration
OUTPUT_DIR="vlm_results"
ALL_MODELS=("gemma3" "llava" "phi4" "qwenvl")
ALL_DATASETS=("blink" "vsr")

# Parse command-line arguments to determine which models/datasets to run
MODELS_TO_RUN=("${ALL_MODELS[@]}")
DATASETS_TO_RUN=("${ALL_DATASETS[@]}")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)
            if [[ -n "$2" && "$2" != --* ]]; then
                MODELS_TO_RUN=("$2")
                shift
            else
                echo "Error: Argument for --model is missing" >&2
                exit 1
            fi
            ;;
        --dataset)
            if [[ -n "$2" && "$2" != --* ]]; then
                DATASETS_TO_RUN=("$2")
                shift
            else
                echo "Error: Argument for --dataset is missing" >&2
                exit 1
            fi
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Model-specific conda environments
declare -A MODEL_ENVS
MODEL_ENVS["gemma3"]="gemma3"
MODEL_ENVS["llava"]="irl_torch2"  
MODEL_ENVS["phi4"]="phi4"
MODEL_ENVS["qwenvl"]="irl_torch2"

# Function to print status messages
print_status() {
    local message=$2
    echo "$message"
}

print_header() {
    echo
    echo "=================================================="
    echo "$1"
    echo "=================================================="
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to activate conda environment for a specific model
activate_model_env() {
    local model=$1
    local env_name="${MODEL_ENVS[$model]}"
    
    echo "Setting up environment for ${model^^}: $env_name"
    
    # Source conda
    source ~/anaconda3/etc/profile.d/conda.sh
    
    # Activate the appropriate environment
    if conda activate "$env_name"; then
        echo "✓ Successfully activated conda environment: $env_name"
        return 0
    else
        echo "✗ Failed to activate conda environment: $env_name"
        return 1
    fi
}

# Function to run a single model on a single dataset
run_single_model() {
    local model=$1
    local dataset=$2
    local script_name="${model}_universal.py"
    
    echo "Running ${model^^} on ${dataset^^} dataset..."
    
    if [[ ! -f "$script_name" ]]; then
        echo "Error: Script $script_name not found!"
        return 1
    fi
    
    # Activate the correct conda environment for this model
    if ! activate_model_env "$model"; then
        echo "Failed to set up environment for $model"
        return 1
    fi
    
    # Run the model
    if python3 "$script_name" --dataset "$dataset" --output_dir "$OUTPUT_DIR"; then
        echo "✓ ${model^^} completed on ${dataset^^}"
        return 0
    else
        echo "✗ ${model^^} failed on ${dataset^^}"
        return 1
    fi
}

# Function to calculate and save AUC
calculate_and_save_auc() {
    local model=$1
    local dataset=$2
    local auc_file="${OUTPUT_DIR}/${model}_${dataset}_auc.txt"
    
    echo "Calculating AUC for ${model^^} on ${dataset^^}..."
    
    # Make sure we're in the right environment (use irl_torch2 for AUC calculation)
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate irl_torch2
    
    # Run AUROC calculation and capture output
    if python3 get_auroc_universal.py --dataset "$dataset" --model "$model" --results_dir "$OUTPUT_DIR" > "$auc_file" 2>&1; then
        echo "✓ AUC saved to $auc_file"
        
        # Extract and display the AUROC value
        local auroc_value=$(grep "AUROC:" "$auc_file" | cut -d' ' -f2)
        if [[ -n "$auroc_value" ]]; then
            echo "  AUROC: $auroc_value"
        fi
        return 0
    else
        echo "✗ Failed to calculate AUC for ${model^^} on ${dataset^^}"
        echo "Error calculating AUC for ${model} on ${dataset}" > "$auc_file"
        return 1
    fi
}

# Main execution
print_header "VLM Verbalized Confidence Experiments"
echo "Models to run: ${MODELS_TO_RUN[*]}"
echo "Datasets to run: ${DATASETS_TO_RUN[*]}"
echo "Output directory: $OUTPUT_DIR"

echo
echo "Model-Environment Mapping:"
for model in "${ALL_MODELS[@]}"; do
    echo "  ${model^^}: ${MODEL_ENVS[$model]}"
done

# Run each model on each dataset sequentially
for model in "${MODELS_TO_RUN[@]}"; do
    print_header "Processing Model: ${model^^}"
    
    for dataset in "${DATASETS_TO_RUN[@]}"; do
        echo
        echo "=== ${model^^} on ${dataset^^} ==="
        
        # Run the model
        if run_single_model "$model" "$dataset"; then
            # If model ran successfully, calculate AUC
            calculate_and_save_auc "$model" "$dataset"
        else
            # If model failed, create error file
            error_file="${OUTPUT_DIR}/${model}_${dataset}_auc.txt"
            echo "Model execution failed for ${model} on ${dataset}" > "$error_file"
            echo "Error file created: $error_file"
        fi
        
        echo "----------------------------------------"
    done
    
    echo
    echo "Completed all datasets for ${model^^}"
    echo
done

# Final summary
print_header "EXPERIMENT SUMMARY"

echo "Results saved in: $OUTPUT_DIR/"
echo
echo "Generated files:"
ls -la "$OUTPUT_DIR/" | grep -E "\.(csv|txt)$" || echo "No files found"

echo
echo "AUC Results Summary:"
echo "===================="

for model in "${MODELS_TO_RUN[@]}"; do
    echo
    echo "${model^^}:"
    for dataset in "${DATASETS_TO_RUN[@]}"; do
        auc_file="${OUTPUT_DIR}/${model}_${dataset}_auc.txt"
        if [[ -f "$auc_file" ]]; then
            auroc_line=$(grep "AUROC:" "$auc_file" 2>/dev/null || echo "No AUROC found")
            echo "  ${dataset^^}: $auroc_line"
        else
            echo "  ${dataset^^}: AUC file not found"
        fi
    done
done

print_header "ALL EXPERIMENTS COMPLETED"
echo "Check individual AUC files in $OUTPUT_DIR/ for detailed results"
