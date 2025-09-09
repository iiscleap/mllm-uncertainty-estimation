import math
import csv
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import os
import argparse

def kl_divergence(p, p_star):
    p_star = np.array(p_star)
    p = np.array(p)
    epsilon = 1e-10
    return np.sum(p_star * np.log((p_star + epsilon) / (p + epsilon)))

def normalize_sigmoid(values):
    # Using a temperature-scaled sigmoid with a fixed optimal temperature
    temperature = 15.0
    epsilon = 1e-9
    reciprocals = np.array([1 / (v + epsilon) for v in values])
    
    # Standardize the reciprocals
    mean = np.mean(reciprocals)
    std = np.std(reciprocals)
    if std == 0:
        return np.full_like(reciprocals, 0.5)
        
    scaled_values = (reciprocals - mean) / std
    
    # Apply the temperature-scaled sigmoid function
    confidences = 1 / (1 + np.exp(-scaled_values / temperature))
    return confidences


def calculate_kl_div(vanilla_exp, uncertainty_exp, p_star, answer_file):
    qns_orig_ans_dict = {}
    with open(vanilla_exp, "r") as f:
        for line in f:
            idx, ans = line.strip().split(" ", maxsplit=1)
            qns_orig_ans_dict[idx] = ans[0]

    qns_count_dict = {}
    with open(uncertainty_exp, "r") as f:
        for line in f:
            idx, ans = line.strip().split(" ", maxsplit=1)
            orig_idx = '_'.join(idx.split('_')[:4])
            ans = ans[0]

            if orig_idx not in qns_count_dict:
                qns_count_dict[orig_idx] = [0, 0]  # [no_toggle, toggle]

            if ans == qns_orig_ans_dict[orig_idx]:
                qns_count_dict[orig_idx][0] += 1
            else:
                qns_count_dict[orig_idx][1] += 1

    kl_scores = []
    labels = []
    with open(answer_file, 'r') as f:
        for line in f:
            idx, ans = line.strip().split(" ", maxsplit=1)
            gold = ans.strip("()")
            pred = qns_orig_ans_dict.get(idx)

            counts = qns_count_dict.get(idx, [0.5, 0.5])  # default neutral
            total = sum(counts)
            p = [counts[0] / total, counts[1] / total]

            kl = kl_divergence(p, p_star)
            is_correct = 1 if pred == gold else 0

            kl_scores.append(kl)
            labels.append(is_correct)

    return kl_scores, labels


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
args = parser.parse_args()

model = args.model
dataset = args.dataset

if args.dataset == "blink":
    answer_file = f"answer_list.txt"
else:
    answer_file = f"dataset/VSR/answer_list.txt"

vanilla = f"vanilla_output/{model}_vanilla_{dataset}.txt"
pos_sample = f"perturb_sampling_output/{model}_perturb_sampling_{dataset}.txt"
neg_sample = f"perturb_sampling_output/negated_{model}_perturb_sampling_{dataset}.txt"


pos_kl, labels_pos = calculate_kl_div(vanilla, pos_sample, [1.0, 0.0], answer_file)
neg_kl, labels_neg = calculate_kl_div(vanilla, neg_sample, [0.0, 1.0], answer_file)
y_true = np.array(labels_pos)

# Calculate and print the accuracy
accuracy = np.mean(y_true)
print(f"Accuracy for {model}-{dataset}: {accuracy:.4f}")

# Average the KL scores first, then normalize the result using sigmoid
combined_kl = [(pos_kl[i] + neg_kl[i]) / 2 for i in range(len(pos_kl))]
y_prob = np.array(normalize_sigmoid(combined_kl))

print(f"\nFirst 5 y_prob values for {model}-{dataset}: {y_prob[:5]}")

# Calculate AUCPR using precision-recall curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
aucpr = auc(recall, precision)
print(f"AUCPR for {model}-{dataset}: {aucpr:.4f}")
