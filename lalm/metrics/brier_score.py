#!/usr/bin/env python3
import math
import csv
import numpy as np
import argparse
import os

def kl_divergence(p, p_star):
    p_star = np.array(p_star, dtype=float)
    p = np.array(p, dtype=float)
    eps = 1e-10
    return np.sum(p_star * np.log((p_star + eps) / (p + eps)))

def normalize_and_reciprocal(entropy):
    epsilon = 0.001
    recip_entropy = []
    for e in entropy:
        if e == 0.0:
            e += epsilon
        recip_entropy.append(1 / e)

    min_val = min(recip_entropy)
    max_val = max(recip_entropy)
    normalized_recip = [(val - min_val) / (max_val - min_val) for val in recip_entropy]
    return normalized_recip

def calculate_kl_div(orig_vanilla, uncertainty_exp, p_star, task):
    subset_idx_list = []
    qns_orig_ans_dict = {}
    with open(orig_vanilla, "r") as data2:
        for line in data2:
            line = line.strip()
            idx, ans = line.split(" ", maxsplit=1)
            qns_orig_ans_dict[idx] = ans[0]
            
    qns_count_dict = {}
    with open(uncertainty_exp, "r") as data:
        for line in data:
            line = line.strip()
            idx, ans = line.split(" ", maxsplit=1)
            orig_idx = idx.split("_")[0]
            if orig_idx not in subset_idx_list:
                subset_idx_list.append(orig_idx)
            ans = ans[0]

            if orig_idx not in qns_count_dict:
                qns_count_dict[orig_idx] = [0, 0]  # [no_toggle, toggle]
            if ans == qns_orig_ans_dict.get(orig_idx):
                qns_count_dict[orig_idx][0] += 1
            else:
                qns_count_dict[orig_idx][1] += 1

    entropy = []
    labels = []

    with open(f"dataset/TREA_dataset/{task}/{task}.csv", mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']
            if idx in subset_idx_list:
                answer = row['correct']
                counts = qns_count_dict.get(idx, [1, 1])
                total = sum(counts)
                p = [counts[0] / total, counts[1] / total]
                en = kl_divergence(p, p_star)
                is_correct = 1 if qns_orig_ans_dict.get(idx) == answer else 0
                entropy.append(en)
                labels.append(is_correct)

    return entropy, labels

def compute_brier_score(probabilities, labels):
    """
    Compute the Brier Score.
    probabilities: confidence scores (higher = more confident)
    labels: binary ground truth (1 = correct, 0 = incorrect)
    """
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    
    # Brier Score = mean((probability - actual)^2)
    return np.mean((probabilities - labels) ** 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of the model")
    parser.add_argument("--task", type=str, required=True, choices=["count", "order", "duration"], help="Task type")

    args = parser.parse_args()
    model = args.model
    task = args.task

    orig_vanilla = f"old_vanilla_output/{model}_{task}_vanilla.txt"
    neg_uncertainty_exp = f"old_perturb_sampling_output/negated_{model}_{task}_perturb_sampling.txt"
    pos_uncertainty_exp = f"old_perturb_sampling_output/{model}_{task}_perturb_sampling.txt"

    pos_kl, labels1 = calculate_kl_div(orig_vanilla, pos_uncertainty_exp, [1.0, 0.0], task)
    neg_kl, _ = calculate_kl_div(orig_vanilla, neg_uncertainty_exp, [0.0, 1.0], task)

    kl_only_list = [(p + n) / 2 for p, n in zip(pos_kl, neg_kl)]
    y_prob = normalize_and_reciprocal(kl_only_list)
    y_true = labels1

    # Calculate Brier Score
    brier = compute_brier_score(y_prob, y_true)
    print(f"Brier Score for {model}-{task}: {brier:.4f}\n")
