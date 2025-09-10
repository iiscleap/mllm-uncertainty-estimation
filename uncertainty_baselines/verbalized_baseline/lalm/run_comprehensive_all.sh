#!/bin/bash
# Run comprehensive metrics script for all LALM verbalized baseline models and tasks
MODELS=(desc_llm qwen salmonn)
TASKS=(count order duration)
OUTDIR="comprehensive_results"
RESDIR="lalm_results"

mkdir -p $OUTDIR
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "Processing $model on $task..."
    python3 comprehensive_metrics_with_plots.py --model $model --task $task --results_dir $RESDIR --output_dir $OUTDIR
done
done
