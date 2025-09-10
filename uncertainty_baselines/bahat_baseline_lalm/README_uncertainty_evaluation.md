# Comprehensive Uncertainty Evaluation with Audio and Text Perturbations

This suite of scripts performs comprehensive uncertainty evaluation using ensemble predictions with input augmentations from both audio and text modalities, computing entropy-based AUC scores.

## Overview

The evaluation covers three perturbation scenarios:
1. **Audio perturbations + original text** (`audio_only`)
2. **Text perturbations + original audio** (`text_only`)  
3. **Both audio + text perturbations** (`text_audio`)

Evaluated across three models:
- **desc_llm** (Qwen2-7B-Instruct with audio descriptions)
- **qwen** (Qwen2-Audio-7B-Instruct)
- **salmonn** (SALMONN model)

And three tasks:
- **count** - Counting sound sources
- **order** - Temporal ordering of sounds
- **duration** - Duration-based reasoning

## Files Created/Modified

### Model Scripts
- `desc_llm_code_modified.py` - Modified desc_llm with 3 perturbation types
- `qwen_audio_modified.py` - Modified qwen_audio with 3 perturbation types
- `salmonn_modified.py` - Template for SALMONN (needs implementation)

### Evaluation Scripts
- `calculate_entropy_comprehensive.py` - Comprehensive entropy and AUC calculation
- `run_single_model.sh` - Run complete evaluation for one model-task pair
- `run_all_experiments.sh` - Run all experiments for all models and tasks
- `test_single_case.sh` - Test single case setup

## Usage

### 1. Single Model-Task Evaluation

Run evaluation for a specific model and task:

```bash
./run_single_model.sh desc_llm order
```

This will:
- Run inference for all 3 perturbation types (audio_only, text_only, text_audio)
- Calculate entropy and AUC scores
- Save results in `entropy_results_desc_llm_order.txt`

### 2. Complete Evaluation (All Models and Tasks)

Run everything:

```bash
./run_all_experiments.sh
```

This will:
- Run all 9 combinations (3 models × 3 tasks)
- Generate individual result files for each combination
- Create a comprehensive summary in `summary_all_results.txt`

### 3. Individual Model Inference

Run inference for a specific model, task, and perturbation type:

```bash
python desc_llm_code_modified.py count audio_only
python qwen_audio_modified.py order text_only
python salmonn_modified.py duration text_audio
```

### 4. Individual Entropy Calculation

Calculate entropy and AUC for specific combination:

```bash
python calculate_entropy_comprehensive.py desc_llm count audio_only
```

## Data Requirements

### Input Files Required:
1. **Audio-only perturbations**: `{task}_audio_perturbations_only.csv`
2. **Text-only perturbations**: `{task}_text_perturbations_only.csv`
3. **Both perturbations**: `/home/debarpanb/VLM_project/TREA_dataset/{task}/{task}_perturbed.csv`
4. **Vanilla results**: `{model}_results/{model}_{task}_vanilla.txt`
5. **Ground truth**: `{task}_subset_100samples.csv`

### Audio Description Files:
- Audio-only: `{task}_audio_perturbations_only_desc/`
- Text-only: Original audio descriptions directory
- Both perturbations: `/home/debarpanb/VLM_project/TREA_dataset/{task}/perturbed_audio_desc/`

## Output Files

### Individual Results:
- `entropy_results_{model}_{task}.txt` - AUC scores for each perturbation type
- `{model}_results/{model}_{exp_type}_{task}.txt` - Raw model predictions

### Summary Results:
- `summary_all_results.txt` - Comprehensive summary of all experiments

### Example Output Format:
```
desc_llm - order:
audio_only: AUC-ROC = 0.7234
text_only: AUC-ROC = 0.6891
text_audio: AUC-ROC = 0.7456
```
## SALMONN Setup Requirements

The SALMONN model requires additional dependencies that may not be installed in your current environment. Before running SALMONN experiments, ensure you have:


### SALMONN Model Path:
The script expects SALMONN to be located at:
`/path/to/SALMONN`

### If SALMONN Dependencies Are Missing:
If you cannot install the SALMONN dependencies, the experiments will still run for desc_llm and qwen models. The bash scripts will skip SALMONN with a warning message.


