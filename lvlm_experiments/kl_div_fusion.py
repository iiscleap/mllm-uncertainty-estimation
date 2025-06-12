import math
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import argparse

def kl_divergence(p, p_star):
    p_star = np.array(p_star)
    p = np.array(p)
    epsilon = 1e-10
    return np.sum(p_star * np.log((p_star + epsilon) / (p + epsilon)))

def normalize_and_reciprocal(values):
    epsilon = 0.001
    reciprocals = [1 / (v + epsilon) for v in values]
    min_val = min(reciprocals)
    max_val = max(reciprocals)
    return [(val - min_val) / (max_val - min_val) for val in reciprocals]

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
parser.add_argument('--model', type=str, required=True, help='Model name (e.g., gemma3, phi4, pixtral, qwenvl, llava)')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., vsr, blink)')
args = parser.parse_args()

model = args.model
dataset = args.dataset

if args.dataset == "blink":
    answer_file = f"dataset/BLINK/answer_list.txt"
else:
    answer_file = f"dataset/VSR/answer_list.txt"

vanilla = f"vanilla_output/{model}_vanilla_{dataset}.txt"
pos_sample = f"perturb_sampling_output/{model}_perturb_sampling_{dataset}.txt"
neg_sample = f"perturb_sampling_output/negated_{model}_perturb_sampling_{dataset}.txt"

pos_kl, labels_pos = calculate_kl_div(vanilla, pos_sample, [1.0, 0.0], answer_file)
neg_kl, labels_neg = calculate_kl_div(vanilla, neg_sample, [0.0, 1.0], answer_file)
y_true = labels_pos

combined_kl = [(pos_kl[i] + neg_kl[i]) / 2 for i in range(len(pos_kl))]
y_prob = normalize_and_reciprocal(combined_kl)

auc_score = roc_auc_score(y_true, y_prob)
print(f"AUC for {model}-{dataset}: {auc_score:.4f}")
