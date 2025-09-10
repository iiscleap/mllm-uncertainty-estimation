import math
import csv
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Calculate toggle-based uncertainty and AUROC.')
parser.add_argument('--model', type=str, required=True, help='Name of the model (e.g., gemma3, llava).')
parser.add_argument('--dataset', type=str, required=True, choices=['blink', 'vsr'], help='Name of the dataset.')
parser.add_argument('--exp_type', type=str, default='text_only', help='Experiment type (e.g., text_only).')
parser.add_argument('--output', type=str, default='rephrased_uncertainty_results_lvlm.txt', help='Output file to append AUROC scores.')
args = parser.parse_args()

model = args.model
dataset = args.dataset
exp_type = args.exp_type

vanilla_exp = f"{model}_results/{model}_vanilla_{dataset}.txt"
uncertainty_exp = f"{model}_results/{model}_{exp_type}_{dataset}.txt"

if dataset == "vsr":
    answer_file = "vsr_data/answers_subset.txt"
elif dataset == "blink":
    answer_file = "blink_data/answer_list.txt"

qns_orig_ans_dict = {}
with open(vanilla_exp, "r") as data2:
    for line in data2:
        line = line.strip(" \n")
        idx,ans = line.split(" ", maxsplit = 1)
        qns_orig_ans_dict[idx] = ans

qns_count_dict = {}
with open(uncertainty_exp, "r") as data:
    for line in data:
        line = line.strip(" \n")
        
        idx,ans = line.split(" ", maxsplit = 1)
        idx_parts = idx.split('_')
        orig_idx = '_'.join(idx_parts[:4])

        if orig_idx not in qns_count_dict.keys():
            qns_count_dict[orig_idx] = [0,0] #[toggle, no_toggle]
        
        if ans != qns_orig_ans_dict[orig_idx]:
            qns_count_dict[orig_idx][0] += 1
        elif ans == qns_orig_ans_dict[orig_idx]:
            qns_count_dict[orig_idx][1] += 1

toggle = []
labels = []

with open(answer_file,'r') as ansfile:
    for line in ansfile:
        line = line.strip(" \n")
        idx, ans = line.split(" ", maxsplit = 1)
        answer = ans.strip("()")

        total = qns_count_dict[idx][0] + qns_count_dict[idx][1]
        tog = qns_count_dict[idx][0]/total
        is_correct = 1 if qns_orig_ans_dict[idx] == answer else 0

        toggle.append(tog)
        labels.append(is_correct)

epsilon = 0.001
recip_toggle = []
for e in toggle:
    if e == 0.0:
        e = e + epsilon
    recip = 1/e
    recip_toggle.append(recip)

min_val = min(recip_toggle)
max_val = max(recip_toggle)
normalized_recip = [(val - min_val) / (max_val - min_val) for val in recip_toggle]

auc_score = roc_auc_score(labels, normalized_recip)
print(f"AUC-ROC ({model} - {dataset} - {exp_type}): {auc_score:.4f}")

# Append result to output file
with open(args.output, 'a') as f:
    f.write(f"Model: {model}, Dataset: {dataset}, Exp_Type: {exp_type}, AUROC: {auc_score:.4f}\n")
