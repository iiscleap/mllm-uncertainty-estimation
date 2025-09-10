# Output Entropy Baseline — LALM (Audio)

Computes uncertainty from the entropy of model outputs aggregated over multiple runs (sampling) for TREA tasks: count, duration, order.

## Models
- salmonn, qwen, desc_llm

## Quick Start

Run entropy AUC for all models/tasks:
```bash
./run_entropy_calculation_lalm.sh
```

Run single combination:
```bash
python calculate_entropy_lalm_original.py \
  --task duration \
  --model qwen \
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
- `entropy_auroc_results_lalm.txt`: AUC-ROC summary
- `../output_sampling_arrays/{model}_{task}_{exp_type}_{method}_y_true.npy`
- `../output_sampling_arrays/{model}_{task}_{exp_type}_{method}_y_prob.npy`

## Method
1. Count predictions per option (A/B/C/D) across multiple sampling runs
2. Compute Shannon entropy from prediction distribution
3. Convert entropy to confidence score (normalized 1/entropy)
4. Evaluate AUC-ROC against ground truth correctness
