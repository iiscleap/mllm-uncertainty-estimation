# Prompting Sampling Baseline - Vision-Language Models (VLM)

This folder implements prompting-based uncertainty estimation for vision-language models using top-k sampling across multiple runs.

## Models Supported
- **Gemma3**: Google's Gemma3 multimodal model
- **LLaVA**: Large Language and Vision Assistant
- **Phi4**: Microsoft's Phi4 multimodal model  
- **Pixtral**: Mistral's vision-language model
- **QwenVL**: Qwen vision-language model

## Datasets
- **BLINK**: Visual reasoning dataset
- **VSR**: Visual spatial reasoning dataset

## Key Parameters
- `K = 4`: Number of answer choices
- `N = 5`: Number of sampling runs for uncertainty estimation

## Usage

### Individual Model Execution
```bash
# Run specific model on specific dataset
python gemma3_blink.py    # Gemma3 on BLINK
python llava_vsr.py       # LLaVA on VSR
python phi4_blink.py      # Phi4 on BLINK
# ... etc for other combinations
```

### Batch Execution
```bash
# Run individual model on both datasets
./run_gemma3.sh     # Gemma3 on BLINK + VSR
./run_llava.sh      # LLaVA on BLINK + VSR
./run_phi4.sh       # Phi4 on BLINK + VSR
./run_pixtral.sh    # Pixtral on BLINK + VSR
./run_qwenvl.sh     # QwenVL on BLINK + VSR

# Run all models on all datasets
./run_all_models.sh
```

### Metrics Calculation
```bash
# Calculate AUROC for specific model/dataset
python get_auroc.py --model gemma3 --dataset blink --output auroc_results.txt

# Comprehensive metrics with plots
python comprehensive_metrics_with_plots.py

# Run comprehensive analysis for all
./run_comprehensive_all.sh
```

## File Structure
- `{model}_{dataset}.py`: Individual model inference scripts
- `run_{model}.sh`: Run specific model on both datasets
- `run_all_models.sh`: Run all models
- `get_auroc.py`: Calculate AUROC scores
- `comprehensive_metrics_with_plots.py`: Generate all metrics and visualizations

## Output
Results are saved in model-specific CSV files with uncertainty scores for each sample based on answer frequency consistency across multiple sampling runs.

## Requirements
- PyTorch
- Transformers
- PIL
- NumPy
- scikit-learn
- Model-specific dependencies (see main FESTA documentation)
