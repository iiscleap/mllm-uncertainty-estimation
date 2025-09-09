#!/bin/bash

# FESTA LVLM Experiments - Llava All Tasks Runner
# This script runs all Llava experiments for both BLINK and VSR datasets
# with detailed progress tracking and verbose output

set -e  # Exit on any error

function run_task_with_progress() {
    local task_name="$1"
    local csv_file="$2"
    local command=("${@:3}")
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
    local dataset="$1"
    local csv_file="$2"
    local output_file="llava_festa_auc_scores.txt"
    
    echo "=========================================" | tee -a "$output_file"
    echo "Calculating FESTA AUC for dataset: $dataset" | tee -a "$output_file"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$output_file"
    echo "=========================================" | tee -a "$output_file"
    
    echo "" | tee -a "$output_file"
    echo "Dataset: $dataset" | tee -a "$output_file"
    python kl_div_fusion.py --dataset "$dataset" --model llava | tee -a "$output_file"
    echo "" | tee -a "$output_file"
    
    echo "Completed AUC calculation for: $dataset" | tee -a "$output_file"
    echo "=========================================" | tee -a "$output_file"
    echo "" | tee -a "$output_file"
}

echo "========================================="
echo "FESTA LVLM Experiments - Llava Runner"
echo "========================================="
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Activate irl_torch2 conda environment
echo "Setting up environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate irl_torch2

# Create output directories if they don't exist
mkdir -p vanilla_output
mkdir -p perturb_sampling_output

# Initialize AUC scores file
echo "FESTA AUC Scores - LLaVA Model" > llava_festa_auc_scores.txt
echo "Generated on: $(date '+%Y-%m-%d %H:%M:%S')" >> llava_festa_auc_scores.txt
echo "==========================================" >> llava_festa_auc_scores.txt
echo "" >> llava_festa_auc_scores.txt

echo "Starting Llava experiments..."
echo ""

# Define datasets
datasets=("blink" "vsr")

# BLINK DATASET
echo "BLINK DATASET - VANILLA EXPERIMENTS"
echo "========================================="

run_task_with_progress "BLINK - Original Questions (Vanilla)" \
    "dataset/BLINK/questions.csv" \
    python llava_vanilla.py \
    --input_csv dataset/BLINK/questions.csv \
    --input_image_folder dataset/BLINK/orig_images \
    --dataset blink \
    --type orig

run_task_with_progress "BLINK - Negated Questions (Vanilla)" \
    "dataset/BLINK/negated_questions.csv" \
    python llava_vanilla.py \
    --input_csv dataset/BLINK/negated_questions.csv \
    --input_image_folder dataset/BLINK/orig_images \
    --dataset blink \
    --type neg

echo "BLINK DATASET - PERTURB SAMPLING"
echo "========================================="

run_task_with_progress "BLINK - Perturbed Questions (Perturb Sampling)" \
    "dataset/BLINK/perturbed_questions.csv" \
    python llava_perturb_sampling.py \
    --input_csv dataset/BLINK/perturbed_questions.csv \
    --input_image_folder dataset/BLINK/perturbed_images \
    --dataset blink \
    --type orig

run_task_with_progress "BLINK - Perturbed Negated Questions (Perturb Sampling)" \
    "dataset/BLINK/perturbed_negated_questions.csv" \
    python llava_perturb_sampling.py \
    --input_csv dataset/BLINK/perturbed_negated_questions.csv \
    --input_image_folder dataset/BLINK/perturbed_negated_images \
    --dataset blink \
    --type neg

# VSR DATASET
echo "VSR DATASET - VANILLA EXPERIMENTS"
echo "========================================="

run_task_with_progress "VSR - Original Questions (Vanilla)" \
    "dataset/VSR/questions.csv" \
    python llava_vanilla.py \
    --input_csv dataset/VSR/questions.csv \
    --input_image_folder dataset/VSR/orig_images \
    --dataset vsr \
    --type orig

run_task_with_progress "VSR - Negated Questions (Vanilla)" \
    "dataset/VSR/negated_questions.csv" \
    python llava_vanilla.py \
    --input_csv dataset/VSR/negated_questions.csv \
    --input_image_folder dataset/VSR/orig_images \
    --dataset vsr \
    --type neg

echo "VSR DATASET - PERTURB SAMPLING"
echo "========================================="

run_task_with_progress "VSR - Perturbed Questions (Perturb Sampling)" \
    "dataset/VSR/perturbed_questions.csv" \
    python llava_perturb_sampling.py \
    --input_csv dataset/VSR/perturbed_questions.csv \
    --input_image_folder dataset/VSR/perturbed_images \
    --dataset vsr \
    --type orig

run_task_with_progress "VSR - Perturbed Negated Questions (Perturb Sampling)" \
    "dataset/VSR/perturbed_negated_questions.csv" \
    python llava_perturb_sampling.py \
    --input_csv dataset/VSR/perturbed_negated_questions.csv \
    --input_image_folder dataset/VSR/perturbed_negated_images \
    --dataset vsr \
    --type neg

echo "All Llava experiments completed!"
echo ""

# CALCULATE FESTA AUC SCORES
echo "CALCULATING FESTA AUC SCORES"
echo "========================================="

for dataset in "${datasets[@]}"; do
    run_auc_calculation "$dataset" "dataset/${dataset^^}/questions.csv"
done

# Add summary to AUC scores file
echo "==========================================" >> llava_festa_auc_scores.txt
echo "Summary completed on: $(date '+%Y-%m-%d %H:%M:%S')" >> llava_festa_auc_scores.txt
echo "All datasets processed: ${datasets[*]}" >> llava_festa_auc_scores.txt
echo "==========================================" >> llava_festa_auc_scores.txt

echo "========================================="
echo "FESTA LVLM Experiments - Llava Complete!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results saved in:"
echo "   - vanilla_output/"
echo "   - perturb_sampling_output/"
echo "   - llava_festa_auc_scores.txt (AUC scores)"
echo "========================================="
