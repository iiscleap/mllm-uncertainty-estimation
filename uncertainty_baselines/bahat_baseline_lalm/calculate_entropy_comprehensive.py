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
    print("Usage: python calculate_entropy_comprehensive.py <model> <dataset> <exp_type>")
    print("model: desc_llm, qwen, salmonn")
    print("dataset: count, order, duration")
    print("exp_type: audio_only, text_only, text_audio")
    sys.exit(1)

model = sys.argv[1]
dataset = sys.argv[2]
exp_type = sys.argv[3]

print(f"Calculating entropy for {model} - {dataset} - {exp_type}")

# File paths
vanilla_exp = f"{model}_results/{model}_{dataset}_vanilla.txt"
uncertainty_exp = f"{model}_results/{model}_{exp_type}_{dataset}.txt"

# Load vanilla results
qns_orig_ans_dict = {}
try:
    with open(vanilla_exp, "r") as data2:
        for line in data2:
            line = line.strip(" \n")
            parts = line.split(" ", maxsplit=1)
            if len(parts) >= 2:
                idx, ans = parts
                ans = ans[0]
                qns_orig_ans_dict[idx] = ans
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
                ans = ans[0]
                
                # Extract original index from perturbation index
                if exp_type == "audio_only":
                    # Format: 59_silence_0 -> 59
                    orig_idx = idx.split('_')[0]
                elif exp_type == "text_only":
                    # Format: 59_rephrased1 -> 59
                    orig_idx = idx.split('_')[0]
                elif exp_type == "text_audio":
                    # Format: 59_add_delete_audio_0_rephrased1 -> 59
                    orig_idx = idx.split('_')[0]
                
                if orig_idx not in qns_count_dict:
                    qns_count_dict[orig_idx] = [0, 0, 0, 0]  # [A, B, C, D]
                
                if ans in ["A", "B", "C", "D"]:
                    idx_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    qns_count_dict[orig_idx][idx_map[ans]] += 1
except FileNotFoundError:
    print(f"Error: Perturbation results file {uncertainty_exp} not found")
    sys.exit(1)

print(f"Loaded perturbation results for {len(qns_count_dict)} questions")

# Load ground truth labels
entropy = []
labels = []
subset_file = f"./subset/{dataset}_subset_100samples.csv"

try:
    with open(subset_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']
            answer = row['correct']

            if idx in qns_count_dict and idx in qns_orig_ans_dict:
                en = compute_entropy(qns_count_dict[idx])
                is_correct = 1 if qns_orig_ans_dict[idx] == answer else 0

                entropy.append(en)
                labels.append(is_correct)
            else:
                print(f"Warning: Missing data for question {idx}")
except FileNotFoundError:
    print(f"Error: Subset file {subset_file} not found")
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
