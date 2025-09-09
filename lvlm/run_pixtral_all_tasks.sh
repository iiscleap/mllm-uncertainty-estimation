#!/bin/bash

# FESTA LVLM Experiments - Pixtral All Tasks Runner
# This script runs all Pixtral experiments for both BLINK and VSR datasets
# with detailed progress tracking and verbose output
# NOTE: Pixtral needs CUDA 12 so it can only be run on GCP with vLLM

set -e  # Exit on any error

function run_task_with_progress() {
    local task_name="$1"
    local csv_file="$2"
    local command=("${@:3}")
    local total_entries=$(($(wc -l < "$csv_file") - 1))

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

echo "========================================="
echo "FESTA LVLM Experiments - Pixtral Runner"
echo "========================================="
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Setting up the vLLM environment (assuming it's set up for GCP)
echo "Setting up vLLM environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vllm_env  # Placeholder - update with actual vLLM environment name

# Create output directories if they don't exist
mkdir -p vanilla_output
mkdir -p perturb_sampling_output

echo "Starting Pixtral experiments..."
echo ""

# BLINK DATASET
echo "BLINK DATASET - VANILLA EXPERIMENTS"
echo "========================================="

run_task_with_progress "BLINK - Original Questions (Vanilla)" \
    "dataset/BLINK/questions.csv" \
    python pixtral_vanilla.py \
    --input_csv dataset/BLINK/questions.csv \
    --input_image_folder dataset/BLINK/orig_images \
    --dataset blink \
    --type orig

run_task_with_progress "BLINK - Negated Questions (Vanilla)" \
    "dataset/BLINK/negated_questions.csv" \
    python pixtral_vanilla.py \
    --input_csv dataset/BLINK/negated_questions.csv \
    --input_image_folder dataset/BLINK/orig_images \
    --dataset blink \
    --type neg

echo "BLINK DATASET - PERTURB SAMPLING"
echo "========================================="

run_task_with_progress "BLINK - Perturbed Questions (Perturb Sampling)" \
    "dataset/BLINK/perturbed_questions.csv" \
    python pixtral_perturb_sampling.py \
    --input_csv dataset/BLINK/perturbed_questions.csv \
    --input_image_folder dataset/BLINK/perturbed_images \
    --dataset blink \
    --type orig

run_task_with_progress "BLINK - Perturbed Negated Questions (Perturb Sampling)" \
    "dataset/BLINK/perturbed_negated_questions.csv" \
    python pixtral_perturb_sampling.py \
    --input_csv dataset/BLINK/perturbed_negated_questions.csv \
    --input_image_folder dataset/BLINK/perturbed_negated_images \
    --dataset blink \
    --type neg

# VSR DATASET
echo "VSR DATASET - VANILLA EXPERIMENTS"
echo "========================================="

run_task_with_progress "VSR - Original Questions (Vanilla)" \
    "dataset/VSR/questions.csv" \
    python pixtral_vanilla.py \
    --input_csv dataset/VSR/questions.csv \
    --input_image_folder dataset/VSR/orig_images \
    --dataset vsr \
    --type orig

run_task_with_progress "VSR - Negated Questions (Vanilla)" \
    "dataset/VSR/negated_questions.csv" \
    python pixtral_vanilla.py \
    --input_csv dataset/VSR/negated_questions.csv \
    --input_image_folder dataset/VSR/orig_images \
    --dataset vsr \
    --type neg

echo "VSR DATASET - PERTURB SAMPLING"
echo "========================================="

run_task_with_progress "VSR - Perturbed Questions (Perturb Sampling)" \
    "dataset/VSR/perturbed_questions.csv" \
    python pixtral_perturb_sampling.py \
    --input_csv dataset/VSR/perturbed_questions.csv \
    --input_image_folder dataset/VSR/perturbed_images \
    --dataset vsr \
    --type orig

run_task_with_progress "VSR - Perturbed Negated Questions (Perturb Sampling)" \
    "dataset/VSR/perturbed_negated_questions.csv" \
    python pixtral_perturb_sampling.py \
    --input_csv dataset/VSR/perturbed_negated_questions.csv \
    --input_image_folder dataset/VSR/perturbed_negated_images \
    --dataset vsr \
    --type neg

echo "All Pixtral experiments completed!"
echo ""

# CALCULATE FESTA AUC SCORES
echo "CALCULATING FESTA AUC SCORES"
echo "========================================="

run_task_with_progress "Calculating BLINK AUC" \
    "dataset/BLINK/questions.csv" \
    python kl_div_fusion.py --dataset blink --model pixtral

run_task_with_progress "Calculating VSR AUC" \
    "dataset/VSR/questions.csv" \
    python kl_div_fusion.py --dataset vsr --model pixtral

echo "========================================="
echo "FESTA LVLM Experiments - Pixtral Complete!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results saved in:"
echo "   - vanilla_output/"
echo "   - perturb_sampling_output/"
echo "========================================="
