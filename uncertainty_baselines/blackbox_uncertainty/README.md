# Blackbox Uncertainty Baselines

Prompting-based uncertainty estimation using multiple sampling runs across multimodal models.

## Folders

**`prompting_sampling_baseline_vlm/`**: Vision-Language Models (5 models: Gemma3, LLaVA, Phi4, Pixtral, QwenVL) on BLINK/VSR datasets

**`prompting_sampling_baseline_lalm/`**: Large Audio Language Models (3 models: Qwen2-Audio, DESC-LLM, SALMONN) on count/duration/order tasks

## Method
Uses top-k sampling across N=5 runs per sample. Uncertainty derived from answer consistency/frequency across runs.

## Quick Start
```bash
# VLM: Run all models on all datasets
cd prompting_sampling_baseline_vlm && ./run_all_models.sh

# LALM: Run all models on all tasks  
cd prompting_sampling_baseline_lalm && ./run_all_lalm.sh
```
