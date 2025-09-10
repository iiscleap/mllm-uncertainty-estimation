#!/bin/bash
# Run qwen on all LALM tasks and compute AUROC

set -e

# Configure these paths before running (no hardcoded placeholders)
# Count
COUNT_INPUT_FILE=${COUNT_INPUT_FILE:-""}
COUNT_AUDIO_DIR=${COUNT_AUDIO_DIR:-""}
COUNT_OUTPUT_FILE=${COUNT_OUTPUT_FILE:-"lalm_results/qwen_topk_sampling_count.csv"}

# Duration
DUR_INPUT_FILE=${DUR_INPUT_FILE:-""}
DUR_AUDIO_DIR=${DUR_AUDIO_DIR:-""}
DUR_OUTPUT_FILE=${DUR_OUTPUT_FILE:-"lalm_results/qwen_topk_sampling_duration.csv"}

# Order
ORD_INPUT_FILE=${ORD_INPUT_FILE:-""}
ORD_AUDIO_DIR=${ORD_AUDIO_DIR:-""}
ORD_OUTPUT_FILE=${ORD_OUTPUT_FILE:-"lalm_results/qwen_topk_sampling_order.csv"}

run_task() {
  local task=$1
  local input_file=$2
  local audio_dir=$3
  local output_file=$4
  if [[ -z "$input_file" || -z "$audio_dir" ]]; then
    echo "[skip] qwen: $task (missing INPUT_FILE or AUDIO_DIR)"
    return 0
  fi
  echo "[run] qwen: $task"
  python qwen.py --input_file "$input_file" --audio_dir "$audio_dir" --output_file "$output_file"
}

run_task "count"    "$COUNT_INPUT_FILE" "$COUNT_AUDIO_DIR" "$COUNT_OUTPUT_FILE"
run_task "duration" "$DUR_INPUT_FILE"   "$DUR_AUDIO_DIR"   "$DUR_OUTPUT_FILE"
run_task "order"    "$ORD_INPUT_FILE"   "$ORD_AUDIO_DIR"   "$ORD_OUTPUT_FILE"

python get_auroc_lalm.py --model qwen --task count
python get_auroc_lalm.py --model qwen --task duration
python get_auroc_lalm.py --model qwen --task order
