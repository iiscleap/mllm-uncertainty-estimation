# Rephrase Uncertainty (Uses External Model Results)

Toggle-based uncertainty computed from vanilla vs. rephrased predictions.

Folders:
- `lvlm/`: Vision models (gemma3, llava, phi4, qwenvl, pixtral) on BLINK/VSR
- `lalm/`: Audio models (qwen, salmonn, desc_llm) on TREA tasks

Important: This code expects model result files generated in other folders (e.g., `{model}_results/...` or `../../bahat_baseline_lalm/...`). See each subfolder README for required file layouts.

Quick start:
```bash
# LVLM
cd lvlm && ./rephrase_uncertainty.sh

# LALM
cd lalm && ./rephrase_uncertainty_lalm.sh
```
