#!/bin/bash
# Run comprehensive metrics script for all LALM rephrase uncertainty models and tasks
MODELS=(salmonn qwen desc_llm)
TASKS=(count order duration)
EXPTYPES=(text_only)
OUTDIR="comprehensive_results"

mkdir -p $OUTDIR
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for exp_type in "${EXPTYPES[@]}"; do
      echo "Processing $model on $task ($exp_type)..."
      python3 comprehensive_metrics_with_plots.py --model $model --task $task --exp_type $exp_type --output_dir $OUTDIR
    done
  done
done
