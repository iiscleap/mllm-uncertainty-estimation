#!/bin/bash
# Run comprehensive metrics script for all LVLM entropy baseline models, datasets, methods, and exp_types
MODELS=(gemma3 llava phi4 pixtral qwenvl)
DATASETS=(blink vsr)
METHODS=(sampling)
EXPTYPES=(orig)
OUTDIR="comprehensive_results"

mkdir -p $OUTDIR
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      for exptype in "${EXPTYPES[@]}"; do
        echo "Processing $model on $dataset with $method ($exptype)..."
        python3 comprehensive_metrics_with_plots.py --model $model --dataset $dataset --method $method --exp_type $exptype --output_dir $OUTDIR
      done
    done
  done
done
