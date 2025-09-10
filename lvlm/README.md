# FESTA-uncertainty-estimation
Uncertainty estimation using equivalent and complementary input sampling

## Dataset Structure

There are two datasets: `BLINK` and `VSR`, each containing image and question files.
1. `BLINK` : `/path/to/BLINK`
2. `VSR` : `/path/to/VSR`

Copy the datasets from the given paths to `/path/to/FESTA-uncertainty-estimation/lvlm/dataset`

### Folder Overview

Each of `BLINK/` and `VSR/` contains:

- **Image Folders**:
  - `orig_images/`: Images for both original questions and negated questions
  - `perturbed_images/`: Perturbed images corresponding to original questions.
  - `perturbed_negated_images/`: Perturbed images corresponding to negated questions.

- **Text and CSV Files**:
  - `answer_list.txt`: Ground truth answers for the original questions.
  - `questions.csv`: Original set of questions
  - `negated_questions.csv`: Negated versions of the original questions.
  - `perturbed_questions.csv`: Perturbed questions for the original set
  - `perturbed_negated_questions.csv`: Perturbed questions for the negated set

## Repository Structure

```
lvlm/
├── model_scripts/          # Model inference scripts
│   ├── gemma3_vanilla.py
│   ├── gemma3_perturb_sampling.py
│   ├── llava_vanilla.py
│   ├── llava_perturb_sampling.py
│   ├── qwenvl_vanilla.py
│   ├── qwenvl_perturb_sampling.py
│   ├── phi4_vanilla.py
│   ├── phi4_perturb_sampling.py
│   ├── pixtral_vanilla.py
│   └── pixtral_perturb_sampling.py
│
├── metrics/                # Metrics calculation scripts
│   ├── kl_div_fusion.py              # FESTA AUC calculation
│   ├── kl_div_aucpr.py               # KL divergence + AUCPR metrics  
│   ├── brier_score.py                # Brier Score calculation
│   ├── ece_fusion.py                 # Expected Calibration Error
│   ├── risk_coverage.py              # Risk-Coverage analysis
│   ├── compute_neg_acc.py            # Negative accuracy computation
│   ├── comprehensive_metrics_with_plots.py  # All metrics + visualizations
│   ├── create_summary_tables.py      # Summary tables and comparisons
│   ├── run_brier_score.sh            # Batch Brier Score calculation
│   ├── run_ece_fusion.sh             # Batch ECE calculation  
│   ├── run_kl_div_aucpr.sh           # Batch KL divergence + AUCPR calculations
│   ├── run_kl_div_fusion.sh          # Batch KL divergence calculations
│   ├── run_risk_coverage.sh          # Batch risk-coverage analysis
│   └── run_complete_analysis_final.sh # Complete analysis pipeline
│
├── run_gemma3_all_tasks.sh  # Run all Gemma3 experiments
├── run_llava_all_tasks.sh   # Run all LLaVA experiments
├── run_qwenvl_all_tasks.sh  # Run all QwenVL experiments
├── run_phi4_all_tasks.sh    # Run all Phi4 experiments
├── run_pixtral_all_tasks.sh # Run all Pixtral experiments
└── dataset/                 # Datasets (user provided)
```

## Running the scripts

There are 2 different scripts - vanilla and perturb_sampling.

Vanilla is to be run on the `orig_images` and `questions.csv` OR `orig_images` and `negated_questions.csv`. Outputs are saved in `vanilla_output`

Perturb_sampling is to be run on the `perturbed_images` and `perturbed_questions.csv` OR `perturbed_negated_images` and `perturbed_negated_questions.csv`. Outputs are saved in `perturb_sampling_output`

```bash
python model_scripts/gemma3_vanilla.py \
  --input_csv /path/to/questions.csv \
  --input_image_folder /path/to/images \
  --dataset blink \
  --type neg
```

### Running All Tasks for a Model
```bash
# Run all experiments (vanilla + perturb) for specific model
./run_gemma3_all_tasks.sh
./run_llava_all_tasks.sh  
./run_qwenvl_all_tasks.sh
./run_phi4_all_tasks.sh
./run_pixtral_all_tasks.sh
```

### Running Complete Analysis
```bash
# Run comprehensive metrics analysis with all plots
cd metrics/
./run_complete_analysis_final.sh

# Results saved to: /path/to/results/lvlm/
```

## Setup for each model

1. **Gemma3** - Use `gemma3` conda env
2. **Llava** - Use `irl_torch2` conda env  
3. **QwenVL** - Use `irl_torch2` conda env
4. **Pixtral** - Needs CUDA 12, based on vLLM. Installation: [https://docs.vllm.ai/en/v0.8.1/getting_started/installation/gpu.html]
5. **Phi4** - Use any compatible env. Model needs to be downloaded locally and moved to `/path/to/FESTA-uncertainty-estimation/lvlm/`

NOTE: Check if the models are restricted on HuggingFace and require access request

## Calculating Metrics

### Individual Metrics
```bash
# FESTA AUC Score
python metrics/kl_div_fusion.py --dataset blink --model gemma3

# Brier Score  
python metrics/brier_score.py --dataset blink --model gemma3

# Expected Calibration Error
python metrics/ece_fusion.py --dataset blink --model gemma3

# Risk-Coverage Analysis
python metrics/risk_coverage.py --dataset blink --model gemma3

# KL Divergence + AUCPR
python metrics/kl_div_aucpr.py --dataset blink --model gemma3

# Negative Accuracy
python metrics/compute_neg_acc.py --dataset blink --model gemma3
```

### Batch Metric Calculation
```bash
cd metrics/
# Run specific metrics for all models/datasets
./run_brier_score.sh
./run_ece_fusion.sh  
./run_kl_div_fusion.sh
./run_kl_div_aucpr.sh
./run_risk_coverage.sh
```

### Comprehensive Analysis
```bash
cd metrics/
# Generate all metrics + plots + summary tables
python comprehensive_metrics_with_plots.py --all_models --all_datasets

# Create summary tables from results
python create_summary_tables.py --results_file /path/to/results.csv --output_dir /path/to/summary/
```

## Output Structure

```
vanilla_output/{model}_{dataset}_{type}_results.csv
perturb_sampling_output/{model}_{dataset}_{type}_results.csv
results/lvlm/comprehensive_metrics_results.csv
results/lvlm/{metric}_plots/
results/lvlm/summary_analysis/
```

## Available Metrics

- **FESTA AUC**: Area under uncertainty curve
- **AUROC**: Area under ROC curve  
- **AUPRC**: Area under Precision-Recall curve
- **Brier Score**: Calibration metric
- **ECE**: Expected Calibration Error
- **Risk-Coverage**: Risk vs coverage analysis
- **Comprehensive**: All metrics with visualizations
