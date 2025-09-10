# Verbalized Baseline — LVLM (Vision)

Verbalized confidence from model outputs on BLINK and VSR. Produces per-sample confidence and AUROC-style metrics.

## Models
- Gemma3: gemma3_universal.py
- LLaVA: llava_universal.py
- Phi4: phi4_universal.py
- QwenVL: qwenvl_universal.py

## Quick start

Run all models on both datasets:
```bash
./run_all_models.sh
```

Run a single model (edit any dataset paths inside the script if needed):
```bash
python gemma3_universal.py  # BLINK/VSR handled inside
```

## Metrics
```bash
# AUROC per model/dataset
python get_auroc_universal.py --model gemma3 --dataset blink --output vlm_auc.txt

# Comprehensive metrics + plots
python comprehensive_metrics_with_plots.py

# Batch comprehensive run
./run_comprehensive_all.sh
```

## Inputs/Outputs
- Inputs: BLINK and VSR images and CSVs as expected by each script
- Outputs: CSVs under vlm_results/ with predicted answers and verbalized confidence

## Notes
- Scripts are “universal”: they parse their own prompts and extract Guess/Probability
- Ensure the correct conda env per model before running
