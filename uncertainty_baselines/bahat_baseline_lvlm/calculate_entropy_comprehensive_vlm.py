import math
import csv
import sys
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def compute_entropy(counts):
    total = sum(counts)
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

# Get model, dataset, and experiment type from command line arguments
if len(sys.argv) != 4:
    print("Usage: python calculate_entropy_comprehensive_vlm.py <model> <dataset> <exp_type>")
    print("model: llava, gemma3, qwenvl, pixtral, phi4")
    print("dataset: blink, vsr")
    print("exp_type: image_only, text_only, text_image")
    sys.exit(1)

model = sys.argv[1]
dataset = sys.argv[2]
exp_type = sys.argv[3]

print(f"Calculating entropy for {model} - {dataset} - {exp_type}")

# File paths
vanilla_exp = f"{model}_results/{model}_vanilla_{dataset}.txt"
uncertainty_exp = f"{model}_results/{model}_{exp_type}_{dataset}.txt"

# Ground truth file
if dataset == "vsr":
    answer_file = "vsr_data/answers_subset.txt"
elif dataset == "blink":
    answer_file = "blink_data/answer_list.txt"
else:
    print(f"Unknown dataset: {dataset}")
    sys.exit(1)

# Load vanilla results
qns_orig_ans_dict = {}
try:
    with open(vanilla_exp, "r") as data2:
        for line in data2:
            line = line.strip(" \n")
            parts = line.split(" ", maxsplit=1)
            if len(parts) >= 2:
                idx, ans = parts
                qns_orig_ans_dict[idx] = ans
            else:
                print(f"Warning: Skipping malformed line in vanilla file: {line}")
except FileNotFoundError:
    print(f"Error: Vanilla results file {vanilla_exp} not found")
    sys.exit(1)

print(f"Loaded {len(qns_orig_ans_dict)} vanilla results")

# Load perturbation results and count answers
qns_count_dict = {}
try:
    with open(uncertainty_exp, "r") as data:
        for line in data:
            line = line.strip(" \n")
            parts = line.split(" ", maxsplit=1)
            if len(parts) >= 2:
                idx, ans = parts
                
                # Extract original index from perturbation index
                if exp_type == "image_only":
                    # Format: val_Spatial_Relation_1_contrast1 -> val_Spatial_Relation_1
                    orig_idx = '_'.join(idx.split('_')[:4])
                elif exp_type == "text_only":
                    # Format: val_Spatial_Relation_1_rephrased1 -> val_Spatial_Relation_1
                    orig_idx = '_'.join(idx.split('_')[:-1])
                elif exp_type == "text_image":
                    # Format: val_Spatial_Relation_1_contrast1_rephrased1 -> val_Spatial_Relation_1
                    orig_idx = '_'.join(idx.split('_')[:4])
                
                if orig_idx not in qns_count_dict:
                    qns_count_dict[orig_idx] = [0, 0]  # [A, B] for binary classification
                
                if ans in ["A", "B"]:
                    idx_map = {"A": 0, "B": 1}
                    qns_count_dict[orig_idx][idx_map[ans]] += 1
            else:
                print(f"Warning: Skipping malformed line in uncertainty file: {line}")
except FileNotFoundError:
    print(f"Error: Perturbation results file {uncertainty_exp} not found")
    sys.exit(1)

print(f"Loaded perturbation results for {len(qns_count_dict)} questions")

# Load ground truth labels and calculate entropy
entropy = []
labels = []

try:
    with open(answer_file, 'r') as ansfile:
        for line in ansfile:
            line = line.strip(" \n")
            parts = line.split(" ", maxsplit=1)
            if len(parts) >= 2:
                idx = parts[0]
                ans = parts[1].strip("()")

                if idx in qns_count_dict and idx in qns_orig_ans_dict:
                    en = compute_entropy(qns_count_dict[idx])
                    is_correct = 1 if qns_orig_ans_dict[idx] == ans else 0

                    entropy.append(en)
                    labels.append(is_correct)
                else:
                    print(f"Warning: Missing data for question {idx}")
            else:
                print(f"Warning: Skipping malformed line in answer file: {line}")
except FileNotFoundError:
    print(f"Error: Answer file {answer_file} not found")
    sys.exit(1)

print(f"Processed {len(entropy)} samples")

if len(entropy) == 0:
    print("Error: No valid samples found")
    sys.exit(1)

# Calculate reciprocal entropy for AUC calculation
epsilon = 0.001
recip_entropy = []
for e in entropy:
    if e == 0.0:
        e = e + epsilon
    recip = 1/e
    recip_entropy.append(recip)

# Normalize reciprocal entropy
min_val = min(recip_entropy)
max_val = max(recip_entropy)
if max_val > min_val:
    normalized_recip = [(val - min_val) / (max_val - min_val) for val in recip_entropy]
else:
    normalized_recip = [0.5] * len(recip_entropy)

# Calculate AUC score
try:
    auc_score = roc_auc_score(labels, normalized_recip)
    print(f"AUC-ROC ({model} - {dataset} - {exp_type}): {auc_score:.4f}")
    
except ValueError as e:
    print(f"Error calculating AUC: {e}")
    print("This might happen if all predictions are correct or all are incorrect")
