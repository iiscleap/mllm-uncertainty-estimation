# Output Entropy Baseline — LVLM (Vision)

Computes uncertainty from the entropy of model outputs aggregated over multiple runs (sampling) for BLINK and VSR.

## Models
- gemma3, llava, phi4, pixtral, qwenvl

## Quick Start

Run entropy AUC for all models/datasets:
```bash
./run_entropy_calculation_lvlm.sh
```

Run single combination:
```bash
python calculate_entropy_lvlm.py \
  --model gemma3 \
  --dataset blink \
  --method sampling \
  --exp_type orig
```

## Metrics and Plots

Comprehensive metrics + plots:
```bash
python comprehensive_metrics_with_plots.py
```

Batch comprehensive:
```bash
./run_comprehensive_all.sh
```

## Inputs/Outputs

**Inputs**: Model prediction tallies per option across repeated runs (sampling) arranged under `{model}/{exp_type}/{method}/...`

**Outputs**:
- `entropy_auroc_results_lvlm.txt`: AUC-ROC summary
- `../output_sampling_arrays/{model}_{dataset}_{exp_type}_{method}_y_true.npy`
- `../output_sampling_arrays/{model}_{dataset}_{exp_type}_{method}_y_prob.npy`

## Method
1. Count predictions per option (A/B/C/D) across multiple sampling runs
2. Compute Shannon entropy from prediction distribution
3. Convert entropy to confidence score (normalized 1/entropy)
4. Evaluate AUC-ROC against ground truth correctness
