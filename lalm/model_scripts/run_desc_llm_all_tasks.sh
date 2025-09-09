#!/bin/bash

# FESTA LALM Experiments - desc_llm All Tasks Runner
# This script runs all desc_llm experiments for count, duration, and order tasks
# with both vanilla and perturb sampling methods

set -e  # Exit on any error

function run_task_with_progress() {
    local task_name="$1"
    local csv_file="$2"
    local command=("${@:3}")
    
    # Check if CSV file exists
    if [[ ! -f "$csv_file" ]]; then
        echo "WARNING: CSV file not found: $csv_file"
        echo "Skipping task: $task_name"
        echo ""
        return
    fi
    
    local total_entries=$((($(wc -l <"$csv_file") - 1)))

    echo "========================================="
    echo "Running task: $task_name"
    echo "CSV File: $(basename "$csv_file")"
    echo "Total Entries: $total_entries"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    
    echo ""
    echo "Executing command: ${command[*]}"
    "${command[@]}"
    
    echo "Completed task: $task_name"
    echo "========================================="
    echo ""
}

function run_auc_calculation() {
    local task="$1"
    local csv_file="$2"
    local output_file="desc_llm_festa_auc_scores.txt"
    
    echo "=========================================" | tee -a "$output_file"
    echo "Calculating FESTA AUC for task: $task" | tee -a "$output_file"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$output_file"
    echo "=========================================" | tee -a "$output_file"
    
    echo "" | tee -a "$output_file"
    echo "Task: $task" | tee -a "$output_file"
    python kl_div_fusion.py --task "$task" --model desc_llm | tee -a "$output_file"
    echo "" | tee -a "$output_file"
    
    echo "Completed AUC calculation for: $task" | tee -a "$output_file"
    echo "=========================================" | tee -a "$output_file"
    echo "" | tee -a "$output_file"
}

echo "========================================="
echo "FESTA LALM Experiments - desc_llm Runner"
echo "========================================="
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Navigate to the experiment directory
cd /home/debarpanb/VLM_project/malay_works/FESTA-uncertainty-estimation/lalm_experiments

# Create output directories if they don't exist
# mkdir -p vanilla_output
# mkdir -p perturb_sampling_output

# Initialize AUC scores file
echo "FESTA AUC Scores - desc_llm Model" > desc_llm_festa_auc_scores.txt
echo "Generated on: $(date '+%Y-%m-%d %H:%M:%S')" >> desc_llm_festa_auc_scores.txt
echo "==========================================" >> desc_llm_festa_auc_scores.txt
echo "" >> desc_llm_festa_auc_scores.txt

echo "Starting desc_llm experiments..."
echo ""

# Define tasks
tasks=("count" "duration" "order")

# # ORIGINAL DATASET EXPERIMENTS
# echo "ORIGINAL DATASET EXPERIMENTS"
# echo "========================================="

# for task in "${tasks[@]}"; do
#     echo "Processing task: $task"
    
#     # Vanilla Inference - Original
#     run_task_with_progress "$task - Original (Vanilla)" \
#         "dataset/TREA_dataset/$task/$task.csv" \
#         python desc_llm_vanilla.py \
#         --task "$task" \
#         --csv_path "dataset/TREA_dataset/$task/$task.csv" \
#         --type "orig" \
#         --desc_folder "dataset/TREA_dataset/$task/audio_desc"

#     # Perturb Sampling - Original
#     run_task_with_progress "$task - Original (Perturb Sampling)" \
#         "dataset/TREA_dataset/$task/${task}_perturbed.csv" \
#         python desc_llm_perturb_sampling.py \
#         --task "$task" \
#         --csv_path "dataset/TREA_dataset/$task/${task}_perturbed.csv" \
#         --type "orig" \
#         --desc_folder "dataset/TREA_dataset/$task/perturbed_audio_desc"
# done

# # NEGATED DATASET EXPERIMENTS
# echo "NEGATED DATASET EXPERIMENTS"
# echo "========================================="

# for task in "${tasks[@]}"; do
#     echo "Processing negated task: $task"
    
#     # Vanilla Inference - Negated
#     run_task_with_progress "$task - Negated (Vanilla)" \
#         "dataset/TREA_dataset_negated/$task/${task}_negated.csv" \
#         python desc_llm_vanilla.py \
#         --task "$task" \
#         --csv_path "dataset/TREA_dataset_negated/$task/${task}_negated.csv" \
#         --type "neg" \
#         --desc_folder "dataset/TREA_dataset_negated/$task/audio_desc"

#     # Perturb Sampling - Negated
#     run_task_with_progress "$task - Negated (Perturb Sampling)" \
#         "dataset/TREA_dataset_negated/$task/${task}_negated_perturbed.csv" \
#         python desc_llm_perturb_sampling.py \
#         --task "$task" \
#         --csv_path "dataset/TREA_dataset_negated/$task/${task}_negated_perturbed.csv" \
#         --type "neg" \
#         --desc_folder "dataset/TREA_dataset_negated/$task/perturbed_audio_desc"
# done

# echo "All desc_llm experiments completed!"
# echo ""

# CALCULATE FESTA AUC SCORES
echo "CALCULATING FESTA AUC SCORES"
echo "========================================="

for task in "${tasks[@]}"; do
    run_auc_calculation "$task" "dataset/TREA_dataset/$task/$task.csv"
done

# Add summary to AUC scores file
echo "==========================================" >> desc_llm_festa_auc_scores.txt
echo "Summary completed on: $(date '+%Y-%m-%d %H:%M:%S')" >> desc_llm_festa_auc_scores.txt
echo "All tasks processed: ${tasks[*]}" >> desc_llm_festa_auc_scores.txt
echo "==========================================" >> desc_llm_festa_auc_scores.txt

echo "========================================="
echo "FESTA LALM Experiments - desc_llm Complete!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results saved in:"
echo "   - vanilla_output/"
echo "   - perturb_sampling_output/"
echo "   - desc_llm_festa_auc_scores.txt (AUC scores)"
echo "========================================="
