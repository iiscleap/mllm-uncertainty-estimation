import math
import csv
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Calculate toggle-based uncertainty and AUROC for LALM.')
parser.add_argument('--model', type=str, required=True, help='Name of the model (e.g., salmonn, qwen).')
parser.add_argument('--task', type=str, required=True, choices=['order', 'count', 'duration'], help='Name of the task.')
parser.add_argument('--exp_type', type=str, default='text_only', help='Experiment type (e.g., text_only, audio_only).')
parser.add_argument('--output', type=str, default='rephrase_uncertainty_auroc_results_lalm.txt', help='Output file to append AUROC scores.')
args = parser.parse_args()

model = args.model
dataset = args.task
exp_type = args.exp_type

vanilla_exp = f"../../bahat_baseline_lalm/{model}_results/{model}_{dataset}_vanilla.txt"
uncertainty_exp = f"../../bahat_baseline_lalm/{model}_results/{model}_{exp_type}_{dataset}.txt"

qns_orig_ans_dict = {}
with open(vanilla_exp, "r") as data2:
    for line in data2:
        line = line.strip(" \n")
        idx,ans = line.split(" ", maxsplit = 1)
        ans = ans[0]
        qns_orig_ans_dict[idx] = ans

qns_count_dict = {}
with open(uncertainty_exp, "r") as data:
    for line in data:
        line = line.strip(" \n")
        
        idx,ans = line.split(" ", maxsplit = 1)
        idx_parts = idx.split('_')
        orig_idx = idx_parts[0]

        ans = ans[0]

        if orig_idx not in qns_count_dict.keys():
            qns_count_dict[orig_idx] = [0,0] #[toggle, no_toggle]
        
        if ans != qns_orig_ans_dict[orig_idx]:
            qns_count_dict[orig_idx][0] += 1
        elif ans == qns_orig_ans_dict[orig_idx]:
            qns_count_dict[orig_idx][1] += 1

toggle = []
labels = []

answer_file = f"../../bahat_baseline_lalm/{dataset}_subset_100samples.csv"

with open(answer_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        idx = row['id']
        answer = row['correct']

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
    f.write(f"Model: {model}, Task: {dataset}, Exp_Type: {exp_type}, AUROC: {auc_score:.4f}\n")
