# Output Entropy Baselines

Entropy-based uncertainty from repeated runs (sampling) of LVLM and LALM models.

## Folders

**`lvlm/`**: Vision models (gemma3, llava, phi4, pixtral, qwenvl) on BLINK/VSR datasets

**`lalm/`**: Audio models (salmonn, qwen, desc_llm) on TREA count/duration/order tasks

## Method
Computes Shannon entropy from prediction distributions across multiple sampling runs. Higher entropy indicates greater uncertainty.

## Quick Start
```bash
# LVLM: Vision models on BLINK/VSR
cd lvlm && ./run_entropy_calculation_lvlm.sh

# LALM: Audio models on TREA tasks  
cd lalm && ./run_entropy_calculation_lalm.sh
```

## Output
AUC-ROC scores using entropy-based confidence against ground truth correctness, plus numpy arrays for further analysis.
