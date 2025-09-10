#!/bin/bash

echo "Running gemma3_sampling.py..."
python gemma3_sampling.py

echo "Running gemma3_perturb.py..."
python gemma3_perturb.py

echo "Running gemma3_perturb_sampling.py..."
python gemma3_perturb_sampling.py
