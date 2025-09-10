#!/bin/bash

# This script calculates entropy for all LALM models and tasks, saving results to a single file.

# --- Configuration ---
MODELS=("desc_llm" "qwen" "salmonn")
TASKS=("count" "order" "duration")
EXP_TYPES=("audio_only" "text_only" "text_audio")
OUTPUT_FILE="entropy_results_lalm.txt"

# --- Main Execution ---

# Initialize/clear the output file
echo "LALM Entropy Calculation Results" > $OUTPUT_FILE
echo "Generated on: $(date)" >> $OUTPUT_FILE
echo "=================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Get total number of experiments for progress tracking
total_experiments=$((${#MODELS[@]} * ${#TASKS[@]} * ${#EXP_TYPES[@]}))
current_experiment=0

echo "Starting entropy calculation for all LALM models and tasks..."

# Loop through each model, task, and experiment type
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for exp_type in "${EXP_TYPES[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "[$current_experiment/$total_experiments] Running: ${model} - ${task} - ${exp_type}"

            # Append a header for the current run to the main output file
            echo "--- Results for ${model} - ${task} - ${exp_type} ---" >> $OUTPUT_FILE

            # Run the Python script and append its output to the single results file
            python calculate_entropy_comprehensive.py "${model}" "${task}" "${exp_type}" >> $OUTPUT_FILE
            
            # Check the exit code of the python script
            if [ $? -ne 0 ]; then
                echo "    └─ ERROR: Python script failed for ${model} - ${task} - ${exp_type}" | tee -a $OUTPUT_FILE
            else
                echo "    └─ Success"
            fi
            echo "" >> $OUTPUT_FILE
        done
    done
done

echo "================================================="
echo "All LALM entropy calculations complete."
echo "Results have been saved to ${OUTPUT_FILE}"
