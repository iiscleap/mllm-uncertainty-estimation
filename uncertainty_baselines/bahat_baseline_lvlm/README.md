# Bahat Baseline - LVLM (Vision)

**Perturbation-based uncertainty estimation for Vision-Language Models using image and text perturbations.**

## Overview
This baseline implements the Bahat et al. approach for uncertainty estimation by applying various perturbations (blur, contrast, noise, rotation, etc.) to images and questions, then measuring prediction consistency.

## Models Supported
- **Gemma3**: gemma3_combined_modified.py
- **LLaVA**: llava_combined_modified.py  
- **Phi4**: phi4_combined_modified.py
- **Pixtral**: pixtral_combined_modified.py
- **QwenVL**: qwenvl_combined_modified.py

## Datasets & Perturbations
- **BLINK**: 2002 samples with image + text perturbations
- **VSR**: 1400 samples with image + text perturbations

### Perturbation Types
- **Image**: blur, contrast, noise, rotation, shift, masking, black/white
- **Text**: Rephrasing variations (covered in separate CSVs)

## Quick Start

### Run All Models & Datasets
```bash
./run_all_experiments_vlm.sh
```

### Run Single Model
```bash
./run_single_model_vlm.sh --model gemma3 --dataset blink --perturbation_type image
```

### Calculate Uncertainty Metrics
```bash
# Entropy-based uncertainty
./run_entropy_calculation.sh

# Comprehensive metrics with plots
./run_comprehensive_all.sh
```

## Data Structure

### Main CSV Files
- `blink_image_perturbations_only.csv`: BLINK image perturbation metadata (2002 samples)
- `blink_text_perturbations_only.csv`: BLINK text perturbation metadata (2002 samples)
- `vsr_image_perturbations_only.csv`: VSR image perturbation metadata (1400 samples)  
- `vsr_text_perturbations_only.csv`: VSR text perturbation metadata (1395 samples)

### Perturbed Images
- `blink_perturbed/blink_perturbed_images/`: 2001 perturbed BLINK images
- `vsr_perturbed/vsr_perturbed_images/`: 1400 perturbed VSR images

### CSV Format
```csv
idx,orig_idx,question,transformation_type,intensity
val_Spatial_Relation_1_blur1,val_Spatial_Relation_1,Is the car beneath the cat?,blur,8
```

## Key Scripts
- `calculate_entropy_comprehensive_vlm.py`: Entropy-based uncertainty calculation
- `comprehensive_metrics_with_plots.py`: Full metrics analysis with visualizations
- `run_all_experiments_vlm.sh`: Batch runner for all model/dataset combinations

## Method
1. Apply perturbations to images/text
2. Collect model predictions on original vs perturbed inputs
3. Calculate prediction consistency as uncertainty metric
4. Generate AUROC scores and comprehensive metrics

## Output
Results include prediction files, uncertainty scores, AUROC metrics, and visualization plots showing model robustness to perturbations.

## Requirements
- PyTorch, Transformers
- PIL (for image processing)
- Model-specific dependencies
- Original BLINK/VSR images (not included in perturbation folders)
