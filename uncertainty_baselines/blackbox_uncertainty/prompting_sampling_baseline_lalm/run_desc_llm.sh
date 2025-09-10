#!/bin/bash
# Run desc_llm on all LALM tasks and compute AUROC

set -e

# Configure these before running
# Count
COUNT_INPUT_FILE=${COUNT_INPUT_FILE:-""}
COUNT_DESC_DIR=${COUNT_DESC_DIR:-""}
COUNT_OUTPUT_FILE=${COUNT_OUTPUT_FILE:-"lalm_results/desc_llm_topk_sampling_count.csv"}

# Duration
DUR_INPUT_FILE=${DUR_INPUT_FILE:-""}
DUR_DESC_DIR=${DUR_DESC_DIR:-""}
DUR_OUTPUT_FILE=${DUR_OUTPUT_FILE:-"lalm_results/desc_llm_topk_sampling_duration.csv"}

# Order
ORD_INPUT_FILE=${ORD_INPUT_FILE:-""}
ORD_DESC_DIR=${ORD_DESC_DIR:-""}
ORD_OUTPUT_FILE=${ORD_OUTPUT_FILE:-"lalm_results/desc_llm_topk_sampling_order.csv"}

run_task() {
  local task=$1
  local input_file=$2
  local desc_dir=$3
  local output_file=$4
  if [[ -z "$input_file" || -z "$desc_dir" ]]; then
    echo "[skip] desc_llm: $task (missing INPUT_FILE or DESC_DIR)"
    return 0
  fi
  echo "[run] desc_llm: $task"
  python desc_llm.py --input_file "$input_file" --desc_folder "$desc_dir" --output_file "$output_file"
}

run_task "count"    "$COUNT_INPUT_FILE" "$COUNT_DESC_DIR" "$COUNT_OUTPUT_FILE"
run_task "duration" "$DUR_INPUT_FILE"   "$DUR_DESC_DIR"   "$DUR_OUTPUT_FILE"
run_task "order"    "$ORD_INPUT_FILE"   "$ORD_DESC_DIR"   "$ORD_OUTPUT_FILE"

python get_auroc_lalm.py --model desc_llm --task count
python get_auroc_lalm.py --model desc_llm --task duration
python get_auroc_lalm.py --model desc_llm --task order
