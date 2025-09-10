#!/bin/bash
# Run comprehensive metrics script for all bahat_baseline_lalm models and tasks/exp_types
MODELS=(desc_llm qwen salmonn)
TASKS=(count order duration)
EXPTYPES=(audio_only text_only text_audio)
OUTDIR="comprehensive_results"

mkdir -p $OUTDIR
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for exp_type in "${EXPTYPES[@]}"; do
      echo "Processing $model on $task ($exp_type)..."
      python3 comprehensive_metrics_with_plots.py --model $model --dataset $task --exp_type $exp_type --output_dir $OUTDIR
    done
  done
done
