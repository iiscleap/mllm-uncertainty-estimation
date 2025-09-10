# Verbalized Baselines

Uncertainty estimation via verbalized confidence scores extracted from model outputs.

## Folders

**`lalm/`**: Audio-language models (Qwen2-Audio, DESC-LLM, SALMONN) on TREA count/duration/order tasks

**`lvlm/`**: Vision-language models (Gemma3, LLaVA, Phi4, QwenVL) on BLINK/VSR visual reasoning

## Method
Prompts models to output both guess and confidence. Extracts numeric confidence for uncertainty estimation via AUROC.

## Quick Start
```bash
# Vision models on BLINK/VSR  
cd lvlm && ./run_all_models.sh

# Audio models on TREA tasks
cd lalm && ./run_qwen_trea_all_tasks.sh
```
