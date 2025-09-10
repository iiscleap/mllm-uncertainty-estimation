#!/bin/bash

# This script calculates toggle-based uncertainty for LALM models and tasks.

MODELS=("qwen" "salmonn" "desc_llm")
TASKS=("order" "count" "duration")
EXP_TYPES=("text_only")

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for exp_type in "${EXP_TYPES[@]}"; do
            echo "Running uncertainty calculation for Model: $model, Task: $task, Exp: $exp_type"
            python calculate_toggle.py --model "$model" --task "$task" --exp_type "$exp_type"
            if [ $? -ne 0 ]; then
                echo "Error running script for Model: $model, Task: $task, Exp: $exp_type"
            fi
        done
    done
done

echo "All LALM uncertainty calculations are complete. Results are in rephrase_uncertainty_auroc_results_lalm.txt"
