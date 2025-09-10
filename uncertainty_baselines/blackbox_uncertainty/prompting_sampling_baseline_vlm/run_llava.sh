#!/bin/bash
# Run Llava on blink and vsr, then compute AUROC for both

python llava_blink.py
python llava_vsr.py
python get_auroc.py --model llava --dataset blink
python get_auroc.py --model llava --dataset vsr
