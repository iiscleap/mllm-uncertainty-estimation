# FESTA-uncertainty-estimation (LALM)
Uncertainty estimation using equivalent and complementary input sampling for Large Audio Language Models

## TREA Dataset Structure

There are two versions of the TREA dataset:
1. `TREA_dataset` : `/path/to/TREA_dataset`
2. `TREA_dataset_negated` : `/path/to/TREA_dataset_negated`

Copy the datasets from the paths to `/path/to/lalm/dataset/`

### Dataset Organization

Each dataset contains three task folders: `count/`, `duration/`, and `order/`

#### TREA_dataset/
- **count/**, **duration/**, **order/**:
  - `audio_desc/`: Textual audio descriptions
  - `audios/`: Original `.wav` audio files
  - `perturbed_audio_desc/`: Descriptions for perturbed audio
  - `perturbed_audios/`: Perturbed `.wav` audio files
  - `{task}.csv`: CSV file for original questions
  - `{task}_perturbed.csv`: CSV file for perturbed questions
  - `{task}_with_metadata.csv`: Combined metadata

#### TREA_dataset_negated/
- **count/**, **duration/**, **order/**:
  - `audio_desc/`: Negated descriptions
  - `audios/`: Negated `.wav` audio files
  - `perturbed_audio_desc/`: Descriptions for perturbed negated audio
  - `perturbed_audios/`: Perturbed negated audio files
  - `{task}_negated.csv`: CSV file for negated questions
  - `{task}_negated_perturbed.csv`: CSV file for perturbed negated questions

## Repository Structure

```
lalm/
├── model_scripts/          # Model inference scripts
│   ├── qwen_vanilla.py
│   ├── qwen_perturb_sampling.py
│   ├── desc_llm_vanilla.py
│   ├── desc_llm_perturb_sampling.py
│   ├── salmonn_vanilla.py
│   ├── salmonn_perturb_sampling.py
│   ├── salmonn.py                    # SALMONN model implementation
│   ├── run_qwen_all_tasks.sh         # Run all Qwen experiments
│   ├── run_desc_llm_all_tasks.sh     # Run all DESC-LLM experiments
│   └── run_salmonn_all_tasks.sh      # Run all SALMONN experiments
│
├── metrics/                # Metrics calculation scripts
│   ├── kl_div_fusion.py              # FESTA AUC calculation
│   ├── kl_div_aucpr.py               # KL divergence + AUCPR metrics
│   ├── brier_score.py                # Brier Score calculation
│   ├── ece_fusion.py                 # Expected Calibration Error
│   ├── risk_coverage.py              # Risk-Coverage analysis
│   ├── comprehensive_metrics_with_plots.py  # All metrics + visualizations
│   ├── create_summary_tables.py      # Summary tables and comparisons
│   ├── run_kl_div_fusion_lalm.sh     # Batch FESTA AUC calculation
│   ├── run_kl_div_aucpr_lalm.sh      # Batch KL + AUCPR calculation
│   ├── run_brier_score_lalm.sh       # Batch Brier Score calculation
│   ├── run_ece_fusion_lalm.sh        # Batch ECE calculation
│   ├── run_risk_coverage_lalm.sh     # Batch risk-coverage analysis
│   └── run_complete_analysis_final.sh # Complete analysis pipeline
│
├── resample_audio_16khz.py # Audio preprocessing utility
└── dataset/                # Datasets (user provided)
```

## Running Individual Scripts

### Vanilla Inference
Loads original audio and questions, generates single predictions per sample.

```bash
python model_scripts/qwen_vanilla.py \
  --task count \
  --csv_path /path/to/dataset/TREA_dataset/count/count.csv \
  --type orig \
  --wav_folder /path/to/dataset/TREA_dataset/count/audios
```

### Perturb Sampling
Loads perturbed questions and generates 10 varied outputs per sample using top-k sampling.

```bash
python model_scripts/qwen_perturb_sampling.py \
  --task count \
  --csv_path /path/to/dataset/TREA_dataset/count/count_perturbed.csv \
  --type orig \
  --wav_folder /path/to/dataset/TREA_dataset/count/perturbed_audios
```

### Parameters
- `--task`: `count`, `duration`, or `order`
- `--type`: `orig` (original dataset) or `neg` (negated dataset)
- Adjust `csv_path` and `wav_folder` according to task and type

### Perturb-sampling parameters
- `--k`: Number of high-temperature sampling runs per input (int). Controls how many perturbed outputs the script generates for each CSV row (default `10`).
- `--max_per_base`: Maximum number of perturbed samples to process per base id (int). When the dataset contains multiple perturbed variants that share the same base id, this caps how many of those perturbed variants are processed for that base (default `56`).

## Running All Tasks for a Model

```bash
# Run all experiments (vanilla + perturb) for specific model
cd model_scripts/
./run_qwen_all_tasks.sh
./run_desc_llm_all_tasks.sh
./run_salmonn_all_tasks.sh
```

## Model Setup

### 1. Qwen2-Audio
- **Environment**: Compatible with standard PyTorch environments
- **Requirements**: `transformers`, `librosa`, `torch`
- **Setup**: No additional setup required

### 2. DESC-LLM (Description-based LLM)
- **Environment**: Standard PyTorch environment
- **Requirements**: `transformers`, `torch`
- **Note**: Processes text descriptions instead of raw audio

### 3. SALMONN
- **Environment**: Requires specific setup
- **Requirements**: Custom SALMONN repository and dependencies
- **Complex Setup Required**:
  ```bash
  # Clone SALMONN repository
  git clone https://github.com/bytedance/SALMONN.git -b salmonn
  
  # Move configuration files to SALMONN/configs/
  # Replace SALMONN/models/salmonn.py with provided version
  # Move salmonn_vanilla.py and salmonn_perturb_sampling.py to SALMONN root
  ```

## Calculating Metrics

### Individual Metrics
```bash
# FESTA AUC Score
python metrics/kl_div_fusion.py --task count --model qwen

# Brier Score
python metrics/brier_score.py --task count --model qwen

# Expected Calibration Error
python metrics/ece_fusion.py --task count --model qwen

# Risk-Coverage Analysis
python metrics/risk_coverage.py --task count --model qwen

# KL Divergence + AUCPR
python metrics/kl_div_aucpr.py --task count --model qwen
```

### Batch Metric Calculation
```bash
cd metrics/
# Run specific metrics for all models/tasks
./run_kl_div_fusion_lalm.sh
./run_brier_score_lalm.sh
./run_ece_fusion_lalm.sh
./run_kl_div_aucpr_lalm.sh
./run_risk_coverage_lalm.sh
```

### Comprehensive Analysis
```bash
cd metrics/
# Run complete analysis with all metrics and visualizations
./run_complete_analysis_final.sh

# Results saved to: /path/to/results/lalm/
```

## Audio Preprocessing

### Resampling Audio
```bash
# Resample all audio files to 16kHz (needed for SALMONN) 
python resample_audio_16khz.py \
  --input_dir /path/to/audio/folder \
  --output_dir /path/to/resampled/folder
```

## Output Structure

```
vanilla_output/{model}_{task}_{type}_results.txt
perturb_sampling_output/{model}_{task}_{type}_results.txt
results/lalm/comprehensive_metrics_results.csv
results/lalm/{metric}_plots/
results/lalm/summary_analysis/
```

## Available Tasks

- **count**: Count distinct sound sources in audio
- **duration**: Determine shortest/longest duration sounds
- **order**: Identify temporal sequence of sound events

## Available Models

- **qwen**: Qwen2-Audio (audio-to-text model)
- **desc_llm**: Description-based LLM (text-only)
- **salmonn**: SALMONN (speech audio language model)

## Available Metrics

- **AUROC**: Area under ROC curve
- **AUPRC**: Area under Precision-Recall curve
- **Brier Score**: Calibration metric
- **ECE**: Expected Calibration Error
- **Risk-Coverage**: Risk vs coverage analysis
- **Comprehensive**: All metrics with visualizations

## Example Usage Workflows

### Complete Experiment for One Model
```bash
# 1. Run all tasks for Qwen
cd model_scripts/
./run_qwen_all_tasks.sh

# 2. Calculate all metrics
cd ../metrics/
python kl_div_fusion.py --task count --model qwen
python kl_div_fusion.py --task duration --model qwen
python kl_div_fusion.py --task order --model qwen
```

### Batch Analysis for All Models
```bash
# 1. Run experiments for all models (separately)
cd model_scripts/
./run_qwen_all_tasks.sh
./run_desc_llm_all_tasks.sh
./run_salmonn_all_tasks.sh

# 2. Run complete analysis
cd ../metrics/
./run_complete_analysis_final.sh
```

### Single Task Analysis
```bash
# For count task only
python model_scripts/qwen_vanilla.py --task count --csv_path /path/to/count.csv --type orig --wav_folder /path/to/audios
python model_scripts/qwen_perturb_sampling.py --task count --csv_path /path/to/count_perturbed.csv --type orig --wav_folder /path/to/perturbed_audios
python metrics/kl_div_fusion.py --task count --model qwen
```

