#!/bin/bash

# This script calculates entropy for all models and datasets, saving results to a single file.

# --- Configuration ---
MODELS=("llava" "gemma3" "qwenvl" "pixtral" "phi4")
DATASETS=("blink" "vsr")
EXP_TYPES=("image_only" "text_only" "text_image")
OUTPUT_FILE="entropy_results.txt"

# --- Main Execution ---

# Initialize/clear the output file
echo "Entropy Calculation Results" > $OUTPUT_FILE
echo "Generated on: $(date)" >> $OUTPUT_FILE
echo "=================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Get total number of experiments for progress tracking
total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#EXP_TYPES[@]}))
current_experiment=0

echo "Starting entropy calculation for all models and datasets..."

# Loop through each model, dataset, and experiment type
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for exp_type in "${EXP_TYPES[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "[$current_experiment/$total_experiments] Running: ${model} - ${dataset} - ${exp_type}"

            # Append a header for the current run to the main output file
            echo "--- Results for ${model} - ${dataset} - ${exp_type} ---" >> $OUTPUT_FILE

            # Run the Python script and append its output to the single results file
            python calculate_entropy_comprehensive_vlm.py "${model}" "${dataset}" "${exp_type}" >> $OUTPUT_FILE
            
            # Check the exit code of the python script
            if [ $? -ne 0 ]; then
                echo "    └─ ERROR: Python script failed for ${model} - ${dataset} - ${exp_type}" | tee -a $OUTPUT_FILE
            else
                echo "    └─ Success"
            fi
            echo "" >> $OUTPUT_FILE
        done
    done
done

echo "================================================="
echo "All entropy calculations complete."
echo "Results have been saved to ${OUTPUT_FILE}"
