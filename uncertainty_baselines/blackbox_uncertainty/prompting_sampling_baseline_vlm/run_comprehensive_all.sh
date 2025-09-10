#!/bin/bash
# Run comprehensive metrics and plots for all VLM baseline models and datasets
MODELS=("gemma3" "llava" "phi4" "pixtral" "qwenvl")
DATASETS=("blink" "vsr")
RESULTS_DIR="comprehensive_results"
mkdir -p "$RESULTS_DIR"
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Running comprehensive metrics for $model on $dataset"
    python comprehensive_metrics_with_plots.py --model "$model" --dataset "$dataset" --output_dir "$RESULTS_DIR"
  done
done
