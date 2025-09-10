# Rephrase Uncertainty — LALM (Audio)

**Note: This uses model results generated in another folder (`../../bahat_baseline_lalm/`).**

Computes toggle-based uncertainty from rephrased questions. Compares vanilla answers vs. rephrased answers to calculate uncertainty based on answer consistency.

## Models & Tasks
- Models: qwen, salmonn, desc_llm
- Tasks: order, count, duration
- Experiment types: text_only (can be extended)

## Quick Start

Calculate rephrase uncertainty for all models/tasks:
```bash
# Runs all combinations of models and tasks
./rephrase_uncertainty_lalm.sh
```

Calculate for a specific model/task:
```bash
python calculate_toggle.py \
  --model qwen \
  --task duration \
  --exp_type text_only \
  --output rephrase_uncertainty_auroc_results_lalm.txt
```

## Metrics
```bash
# Comprehensive metrics + plots
python comprehensive_metrics_with_plots.py

# Batch comprehensive run
./run_comprehensive_all.sh
```

## Input Requirements
**External dependencies** (generated elsewhere):
- `../../bahat_baseline_lalm/{model}_results/{model}_{task}_vanilla.txt`: Original answers
- `../../bahat_baseline_lalm/{model}_results/{model}_text_only_{task}.txt`: Rephrased answers
- `../../bahat_baseline_lalm/{task}_subset_100samples.csv`: Ground truth

## Output
- `rephrase_uncertainty_auroc_results_lalm.txt`: AUROC scores
- Uncertainty based on toggle frequency between vanilla and rephrased responses
