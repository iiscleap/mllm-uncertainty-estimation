#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Running phi4_vanilla.py..."
python phi4_vanilla.py

echo "Running phi4_sampling.py..."
python phi4_sampling.py

echo "Running phi4_perturb.py..."
python phi4_perturb.py

echo "Running phi4_perturb_sampling.py..."
python phi4_perturb_sampling.py

