#!/bin/bash

echo "Running llava_vanilla.py..."
python llava_vanilla.py

echo "Running llava_sampling.py..."
python llava_sampling.py

echo "Running llava_perturb.py..."
python llava_perturb.py

echo "Running llava_perturb_sampling.py..."
python llava_perturb_sampling.py
