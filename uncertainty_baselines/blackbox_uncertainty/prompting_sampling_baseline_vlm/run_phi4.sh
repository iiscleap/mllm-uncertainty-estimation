#!/bin/bash
# Run Phi4 on blink and vsr, then compute AUROC for both

python phi4_blink.py
python phi4_vsr.py
python get_auroc.py --model phi4 --dataset blink
python get_auroc.py --model phi4 --dataset vsr
