#!/bin/bash
# Run comprehensive metrics script for all LALM entropy baseline models, tasks, methods, and exp_types
MODELS=(salmonn qwen desc_llm)
TASKS=(count duration order)
METHODS=(sampling)
EXPTYPES=(orig)
OUTDIR="comprehensive_results"

mkdir -p $OUTDIR
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for method in "${METHODS[@]}"; do
      for exptype in "${EXPTYPES[@]}"; do
        echo "Processing $model on $task with $method ($exptype)..."
        python3 comprehensive_metrics_with_plots.py --model $model --task $task --method $method --exp_type $exptype --output_dir $OUTDIR
      done
    done
  done
done
