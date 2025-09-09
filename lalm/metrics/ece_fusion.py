import math
import csv
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def kl_divergence(p, p_star):
    p_star = np.array(p_star)
    p = np.array(p)
    epsilon = 1e-10
    return np.sum(p_star * np.log((p_star + epsilon) / (p + epsilon)))

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

def compute_accuracy(probabilities, labels):
    return np.mean(np.array(probabilities >= 0.5) == np.array(labels))

def compute_ece(probabilities, labels, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        if i == num_bins - 1:  # Last bin includes upper boundary
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            bin_accuracy = labels[in_bin].mean()
            bin_confidence = probabilities[in_bin].mean()
            ece += np.abs(bin_accuracy - bin_confidence) * prop_in_bin
            
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(in_bin.sum())
        else:
            bin_confidences.append(0)
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    return ece, bin_confidences, bin_accuracies, bin_counts

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
    y_prob = np.array(normalize_and_reciprocal(kl_only_list))
    y_true = np.array(labels1)

    # Calculate ECE
    ece_score, bin_confidences, bin_accuracies, bin_counts = compute_ece(y_prob, y_true)
    accuracy = compute_accuracy(y_prob, y_true)
    
    print(f"ECE for {model}-{task}: {ece_score:.4f}")
    print(f"Accuracy for {model}-{task}: {accuracy:.4f}\n")
