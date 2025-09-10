#!/bin/bash

# Usage: ./run_single_model_vlm.sh <model>
# model: llava, gemma3, qwenvl, pixtral, phi4

set -e # Exit on any error

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model>"
    echo "model: llava, gemma3, qwenvl, pixtral, phi4"
    exit 1
fi

MODEL=$1
DATASETS=("vsr")

# --- Conda Environment Configuration ---
declare -A MODEL_ENVS
MODEL_ENVS["gemma3"]="gemma3"
MODEL_ENVS["llava"]="irl_torch2"
MODEL_ENVS["phi4"]="phi4"
MODEL_ENVS["qwenvl"]="irl_torch2"
MODEL_ENVS["pixtral"]="pixtral" # Assuming pixtral has its own env

# --- Function to activate Conda environment ---
activate_env() {
    local model_name=$1
    local env_name=${MODEL_ENVS[$model_name]}

    if [ -z "$env_name" ]; then
        echo "✗ Error: No environment defined for model '$model_name'"
        exit 1
    fi

    echo "Activating Conda environment for ${model_name^^}: $env_name"
    source ~/anaconda3/etc/profile.d/conda.sh
    if ! conda activate "$env_name"; then
        echo "✗ Failed to activate conda environment: $env_name"
        exit 1
    fi
    echo "✓ Successfully activated environment: $env_name"
}

# --- Main Execution ---

# Activate the environment for the specified model
activate_env $MODEL

echo ""
echo "Running complete evaluation for $MODEL on all datasets"

# Create results directory if it doesn't exist
mkdir -p "${MODEL}_results"

# Loop over both datasets
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "============================================="
    echo "Starting evaluation for $MODEL on $DATASET dataset"
    echo "============================================="

    echo "Experiment types: image_only, text_only, text_image"
    echo ""

    # Run model inference for all three experiment types
    for EXP_TYPE in "text_image"; do
        echo "Running $MODEL inference for $DATASET - $EXP_TYPE..."
        
        case $MODEL in
            "llava")
                python llava_combined_modified.py $DATASET $EXP_TYPE
                ;;
            "gemma3")
                python gemma3_combined_modified.py $DATASET $EXP_TYPE
                ;;
            "qwenvl")
                python qwenvl_combined_modified.py $DATASET $EXP_TYPE
                ;;
            "pixtral")
                python pixtral_combined_modified.py $DATASET $EXP_TYPE
                ;;
            "phi4")
                python phi4_combined_modified.py $DATASET $EXP_TYPE
                ;;
            *)
                echo "Unknown model: $MODEL"
                exit 1
                ;;
        esac

        if [ $? -ne 0 ]; then
            echo "✗ Failed inference for $MODEL - $DATASET - $EXP_TYPE"
            continue # Skip to the next dataset if inference fails
        fi
        echo "✓ Completed inference for $MODEL - $DATASET - $EXP_TYPE"
        echo ""
    done

    # Calculate entropy and AUC for all experiment types
    echo "Calculating entropy and AUC scores for $DATASET..."
    for EXP_TYPE in "image_only" "text_only" "text_image"; do
        echo "Processing $EXP_TYPE..."
        python calculate_entropy_comprehensive_vlm.py $MODEL $DATASET $EXP_TYPE
        if [ $? -ne 0 ]; then
            echo "✗ Failed entropy calculation for $EXP_TYPE"
        else
        echo "✓ Completed entropy calculation for $EXP_TYPE"
        fi
    done

    echo ""
    echo "============================================="
    echo "Results for $MODEL - $DATASET saved in: entropy_results_vlm_${MODEL}_${DATASET}.txt"
    
    # Display final results for the current dataset
    if [ -f "entropy_results_vlm_${MODEL}_${DATASET}.txt" ]; then
        echo ""
        echo "Final AUC Results for $MODEL - $DATASET:"
        cat "entropy_results_vlm_${MODEL}_${DATASET}.txt"
    fi

    echo "Evaluation complete for $MODEL - $DATASET"
done

echo ""
echo "============================================="
echo "All evaluations for model $MODEL are complete."
echo "Model outputs saved in: ${MODEL}_results/"

