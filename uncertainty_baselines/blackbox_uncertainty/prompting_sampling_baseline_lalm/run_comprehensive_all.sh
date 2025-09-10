#!/bin/bash
# Run comprehensive metrics and plots for all LALM baseline models and tasks
MODELS=("desc_llm" "qwen" "salmonn")
TASKS=("count" "order" "duration")
RESULTS_DIR="comprehensive_results"
mkdir -p "$RESULTS_DIR"
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "Running comprehensive metrics for $model on $task"
    python comprehensive_metrics_with_plots.py --model "$model" --task "$task" --output_dir "$RESULTS_DIR"
  done
done
