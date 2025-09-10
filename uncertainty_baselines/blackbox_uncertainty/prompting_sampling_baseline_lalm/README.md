# Prompting Sampling Baseline - Large Audio Language Models (LALM)

This folder implements prompting-based uncertainty estimation for audio-language models using top-k sampling across multiple runs.

## Models Supported
- **Qwen2-Audio**: Qwen's audio-language model
- **DESC-LLM**: Description-based LLM using audio captions
- **SALMONN**: Speech Audio Language Model with custom setup

## Tasks
- **count**: Number of distinct sound sources
- **duration**: Shortest/longest sound duration
- **order**: Temporal order of sound events

## Key Parameters
- `K = 4`: Number of answer choices
- `N = 5`: Number of sampling runs for uncertainty estimation

## Usage

### Individual Model Execution
```bash
# Qwen2-Audio (example: order task)
python qwen.py \
  --input_file /abs/path/.../order_task/order_subset_100samples.csv \
  --audio_dir  /abs/path/.../ESC_50_reasoning_order_dataset/audios \
  --output_file lalm_results/qwen_topk_sampling_order.csv

# DESC-LLM (uses precomputed audio descriptions)
python desc_llm.py \
  --input_file /abs/path/.../order_task/order_subset_100samples.csv \
  --desc_folder /abs/path/.../order_salmonn_mmau_desc \
  --output_file lalm_results/desc_llm_topk_sampling_order.csv

# SALMONN (requires SALMONN repo setup; export SALMONN_HOME)
export SALMONN_HOME=/abs/path/to/SALMONN
python salmonn.py \
  --input_file /abs/path/.../order_task/order_subset_100samples.csv \
  --audio_dir  /abs/path/.../ESC_50_reasoning_order_dataset/audios \
  --output_file lalm_results/salmonn_topk_sampling_order.csv
```

Each script now accepts CLI arguments:
- `--input_file`: subset CSV (per task)
- `--audio_dir`: audio WAV folder (for Qwen/SALMONN)
- `--desc_folder`: description folder (for DESC-LLM)
- `--output_file`: output CSV path

### Batch Execution
```bash
# Run all LALM models
./run_all_lalm.sh

# Run a specific model
./run_qwen.sh     # reads COUNT_*/DUR_*/ORD_* env vars for input/audio/output
./run_desc_llm.sh # reads COUNT_*/DUR_*/ORD_* env vars for input/desc/output
./run_salmonn.sh  # requires SALMONN_HOME and reads COUNT_*/DUR_*/ORD_* env vars
```

### Metrics Calculation
```bash
# AUROC for LALM tasks
python get_auroc_lalm.py --model qwen --task duration --output auroc_results.txt

# Comprehensive metrics with plots
python comprehensive_metrics_with_plots.py

# Run comprehensive analysis for all
./run_comprehensive_all.sh
```

## File Structure
- `{model}.py`: Individual model inference scripts
- `run_{model}.sh`: Run specific model on all tasks
- `run_all_lalm.sh`: Run all LALM models
- `get_auroc_lalm.py`: Calculate AUROC scores
- `comprehensive_metrics_with_plots.py`: Generate all metrics and visualizations

## Output
Results are saved in model-specific CSV files with uncertainty scores per sample based on answer consistency across sampling runs.

## Requirements
- PyTorch
- Transformers
- librosa (for Qwen2-Audio)
- scikit-learn
- SALMONN-specific dependencies (if using SALMONN)
