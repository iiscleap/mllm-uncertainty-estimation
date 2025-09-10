#!/bin/bash
# Run comprehensive metrics script for all VLM models, datasets, and experiment types
MODELS=("llava" "gemma3" "qwenvl" "pixtral" "phi4")
DATASETS=("blink" "vsr")
EXPTYPES=("image_only" "text_only" "text_image")
OUTDIR="comprehensive_results"

mkdir -p "$OUTDIR"
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for exp_type in "${EXPTYPES[@]}"; do
      echo "Processing $model on $dataset ($exp_type)..."
      python3 comprehensive_metrics_with_plots.py \
        --model "$model" \
        --dataset "$dataset" \
        --exp_type "$exp_type" \
        --output_dir "$OUTDIR"
    done
  done
done

echo "All metrics and plots have been generated in the '$OUTDIR' directory."
