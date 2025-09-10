#!/bin/bash

# This script calculates AUROC for all LALM models and tasks using get_auroc_trea.py

# --- Configuration ---
MODELS=("desc_llm" "qwen" "salmonn")
TASKS=("count" "duration" "order")
OUTPUT_FILE="verbalized_auroc_results_trea.txt"

# --- Main Execution ---

# Initialize/clear the output file
echo "TREA AUROC Calculation Results" > $OUTPUT_FILE
echo "Generated on: $(date)" >> $OUTPUT_FILE
echo "=================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

total_experiments=$((${#MODELS[@]} * ${#TASKS[@]}))
current_experiment=0

echo "Starting AUROC calculation for all models and tasks..."

# Loop through each model and task
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo "[$current_experiment/$total_experiments] Running: ${model} - ${task}"

        # Run the Python script and append its output to the single results file
        # The python script already prints a nice header, so we just append its output.
        python get_auroc_trea.py --model "${model}" --task "${task}" >> $OUTPUT_FILE
        
        # Add a separator for readability
        echo "" >> $OUTPUT_FILE
        echo "-------------------------------------------------" >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
    done
done

echo "================================================="
echo "All AUROC calculations complete."
echo "Results have been saved to ${OUTPUT_FILE}"
