#!/bin/bash
# Run Pixtral on blink and vsr, then compute AUROC for both

python pixtral_blink.py
python pixtral_vsr.py
python get_auroc.py --model pixtral --dataset blink
python get_auroc.py --model pixtral --dataset vsr
