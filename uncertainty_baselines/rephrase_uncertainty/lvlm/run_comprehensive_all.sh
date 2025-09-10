#!/bin/bash
# Run comprehensive metrics script for all LVLM rephrase uncertainty models and datasets
MODELS=(gemma3 llava phi4 qwenvl pixtral)
DATASETS=(blink vsr)
EXPTYPES=(text_only)
OUTDIR="comprehensive_results"

mkdir -p $OUTDIR
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for exp_type in "${EXPTYPES[@]}"; do
      echo "Processing $model on $dataset ($exp_type)..."
      python3 comprehensive_metrics_with_plots.py --model $model --dataset $dataset --exp_type $exp_type --output_dir $OUTDIR
    done
  done
done
