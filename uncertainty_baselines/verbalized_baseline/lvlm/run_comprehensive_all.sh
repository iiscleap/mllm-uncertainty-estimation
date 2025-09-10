#!/bin/bash
# Run comprehensive metrics script for all LVLM verbalized baseline models and datasets
MODELS=(gemma3 llava phi4 pixtral qwenvl)
DATASETS=(blink vsr)
OUTDIR="comprehensive_results"
RESDIR="vlm_results"

mkdir -p $OUTDIR
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Processing $model on $dataset..."
    python3 comprehensive_metrics_with_plots.py --model $model --dataset $dataset --results_dir $RESDIR --output_dir $OUTDIR
done
done
