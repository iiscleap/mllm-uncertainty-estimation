#!/bin/bash
# Run Gemma3 on blink and vsr, then compute AUROC for both

python gemma3_blink.py
python gemma3_vsr.py
python get_auroc.py --model gemma3 --dataset blink
python get_auroc.py --model gemma3 --dataset vsr
