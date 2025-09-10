# Rephrase Uncertainty — LVLM (Vision)

**Note: This uses model results generated in another folder (model_results folders expected next to this code).**

Computes toggle-based uncertainty from rephrased prompts. Compares vanilla vs. rephrased predictions to derive uncertainty via answer flips.

## Models & Datasets
- Models: gemma3, llava, phi4, qwenvl, pixtral
- Datasets: blink, vsr
- Experiment types: text_only (default)

## Quick Start

Calculate rephrase uncertainty for all models/datasets:
```bash
./rephrase_uncertainty.sh
```

Calculate for a specific model/dataset:
```bash
python calculate_toggle.py \
  --model gemma3 \
  --dataset blink \
  --exp_type text_only \
  --output rephrased_uncertainty_results_lvlm.txt
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
- `{model}_results/{model}_vanilla_{dataset}.txt`: Original answers
- `{model}_results/{model}_text_only_{dataset}.txt`: Rephrased answers
- `blink_data/answer_list.txt` or `vsr_data/answers_subset.txt`: Ground truth

## Output
- `rephrased_uncertainty_results_lvlm.txt`: AUROC scores
- Uncertainty based on toggle frequency between vanilla and rephrased responses
