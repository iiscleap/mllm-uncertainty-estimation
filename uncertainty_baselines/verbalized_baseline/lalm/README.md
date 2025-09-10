# Verbalized Baseline — LALM (Audio)

Verbalized confidence from model outputs on TREA tasks (count, duration, order). Produces per-sample confidence and AUROC-style metrics.

## Models
- Qwen2-Audio: qwen_audio_trea.py
- DESC-LLM: desc_llm_trea.py
- SALMONN: salmonn_trea.py (requires external SALMONN setup)

## Quick start

Run all tasks for a model:
```bash
# Qwen2-Audio on all TREA tasks
./run_qwen_trea_all_tasks.sh

# DESC-LLM on all tasks
./run_desc_llm_trea.sh

# SALMONN on all tasks (after SALMONN setup)
./run_salmonn_trea_all_tasks.sh
```

Run a single script (edit dataset_path if needed):
```bash
python qwen_audio_trea.py \
  --task duration \
  --dataset_path /path/to/TREA_dataset \
  --output_dir lalm_results
```

## Metrics
```bash
# AUROC over verbalized confidence
python get_auroc_trea.py --model qwen --task duration --output lalm_auc.txt

# Comprehensive metrics + plots
python comprehensive_metrics_with_plots.py

# Batch comprehensive run
./run_comprehensive_all.sh
```

## Inputs/Outputs
- Inputs: TREA CSVs under {count|duration|order}/ and corresponding audio/ or descriptions/
- Outputs: CSVs under lalm_results/ with predicted answer and verbalized confidence

## Notes
- Use resample_audio_16khz.py if your audio needs 16kHz normalization
- SALMONN requires its repo and configs; see comments in salmonn_trea.py
