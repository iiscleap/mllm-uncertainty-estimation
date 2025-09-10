#!/bin/bash
# Run QwenVL on blink and vsr, then compute AUROC for both

python qwenvl_blink.py
python qwenvl_vsr.py
python get_auroc.py --model qwenvl --dataset blink
python get_auroc.py --model qwenvl --dataset vsr
