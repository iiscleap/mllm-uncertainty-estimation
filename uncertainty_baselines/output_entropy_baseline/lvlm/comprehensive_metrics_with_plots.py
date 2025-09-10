import math
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
import os
import argparse
import json
import pandas as pd

def compute_entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

def parse_ground_truth(gt_file, dataset):
    gt_dict = {}
    if dataset == "blink" or dataset == "vsr":
        with open(gt_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" (")
                    if len(parts) == 2:
                        idx = parts[0]
                        answer = parts[1].rstrip(")")
                        gt_dict[idx] = answer
    return gt_dict

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            bin_weight = bin_count / len(y_prob)
            ece += np.abs(bin_confidence - bin_accuracy) * bin_weight
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)
        else:
            bin_confidences.append(0)
            bin_accuracies.append(0)
            bin_counts.append(0)
    return ece, bin_confidences, bin_accuracies, bin_counts

def risk_coverage_peel_off_low(conf, correct):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, int)
    N = len(conf)
    order_low_first = np.argsort(conf, kind="mergesort")
    corr_low_first = correct[order_low_first]
    cum_correct_low = np.concatenate(([0], np.cumsum(corr_low_first)))
    total_correct = correct.sum()
    k = np.arange(N+1)
    kept = N - k
    correct_kept = total_correct - cum_correct_low
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = correct_kept / np.maximum(kept, 1)
        risk = 1.0 - accuracy
    coverage = kept / N
    mask = kept > 0
    aurc = np.trapz(risk[mask][::-1], coverage[mask][::-1])
    return coverage, risk, accuracy, aurc

def plot_risk_coverage_desc(coverage, risk, model, dataset, method, exp_type, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(coverage, risk, linewidth=2, color='r')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Risk (error on answered set)")
    plt.title(f"Risk–Coverage for {model} on {dataset} [{method}|{exp_type}]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0)
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{dataset}_{method}_{exp_type}_risk_coverage.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_accuracy_coverage(coverage, accuracy, model, dataset, method, exp_type, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(coverage, accuracy, linewidth=2, color='g')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Accuracy on Answered Set")
    plt.title(f"Accuracy vs. Coverage for {model} on {dataset} [{method}|{exp_type}]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0)
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{dataset}_{method}_{exp_type}_accuracy_coverage.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, model, dataset, method, exp_type, save_dir, num_bins=10):
    os.makedirs(save_dir, exist_ok=True)
    bin_centers = np.linspace(0.05, 0.95, num_bins)
    counts_arr = np.array(bin_counts)
    min_c, max_c = np.min(counts_arr), np.max(counts_arr)
    norm_counts = (counts_arr - min_c) / (max_c - min_c + 1e-9)
    colors = plt.cm.Blues(norm_counts)
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, bin_accuracies, width=0.08, color=colors, edgecolor='black', label='Bin Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibration Plot for {model} on {dataset} [{method}|{exp_type}]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f"{model}_{dataset}_{method}_{exp_type}_calibration.png")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    return plot_filename

def plot_uncertainty_distribution(y_prob, y_true, model, dataset, method, exp_type, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    correct_scores = y_prob_np[y_true_np == 1]
    incorrect_scores = y_prob_np[y_true_np == 0]
    plt.figure(figsize=(10, 6))
    y_jitter_correct = np.random.normal(1, 0.05, size=len(correct_scores))
    y_jitter_incorrect = np.random.normal(0, 0.05, size=len(incorrect_scores))
    plt.scatter(correct_scores, y_jitter_correct, label='Correct (y_true=1)', color='blue', alpha=0.6, s=15)
    plt.scatter(incorrect_scores, y_jitter_incorrect, label='Incorrect (y_true=0)', color='red', alpha=0.6, s=15)
    plt.xlabel('Confidence Score')
    plt.ylabel('Prediction Outcome')
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.title(f'Confidence Score Distribution for {model} on {dataset} [{method}|{exp_type}]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    out = os.path.join(save_dir, f"{model}_{dataset}_{method}_{exp_type}_uncertainty_distribution.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out

def main():
    parser = argparse.ArgumentParser(description='Comprehensive metrics and plots for LVLM entropy baseline')
    parser.add_argument('--model', required=True, choices=['llava', 'gemma3', 'phi4', 'pixtral', 'qwenvl', 'internvl3'])
    parser.add_argument('--method', required=True, choices=['perturb_sampling', 'sampling', 'perturb'])
    parser.add_argument('--dataset', required=True, choices=['blink', 'vsr'])
    parser.add_argument('--exp_type', default='orig', choices=['orig', 'neg'])
    parser.add_argument('--output_dir', type=str, default='comprehensive_results')
    args = parser.parse_args()
    # Set ground truth file path
    if args.dataset == "blink":
        gt_file = "path/to/blink_data/answer_list.txt"
    elif args.dataset == "vsr":
        gt_file = "path/to/vsr_data/answers_subset.txt"
    gt_dict = parse_ground_truth(gt_file, args.dataset)
    # File paths and data logic identical to calculate_entropy_lvlm.py
    # Construct file paths based on LVLM structure
    model_dir = f"{args.model}/{args.model}_results"
    # Vanilla file (baseline predictions)
    if args.exp_type == "orig":
        if args.dataset == "blink":
            vanilla_file = f"{model_dir}/{args.model}_vanilla.txt"
        else:  # vsr
            vanilla_file = f"{model_dir}/{args.model}_vanilla_vsr.txt"
    else:  # neg
        if args.dataset == "blink":
            vanilla_file = f"{model_dir}/negated_{args.model}_vanilla.txt"
        else:  # vsr
            vanilla_file = f"{model_dir}/negated_{args.model}_vanilla_vsr.txt"
    # Uncertainty file (perturbed predictions)
    if args.exp_type == "orig":
        if args.dataset == "blink":
            uncertainty_file = f"{model_dir}/{args.model}_{args.method}.txt"
        else:  # vsr
            uncertainty_file = f"{model_dir}/{args.model}_{args.method}_vsr.txt"
    else:  # neg
        if args.dataset == "blink":
            uncertainty_file = f"{model_dir}/negated_{args.model}_{args.method}.txt"
        else:  # vsr
            uncertainty_file = f"{model_dir}/negated_{args.model}_{args.method}_vsr.txt"
    if not os.path.exists(vanilla_file):
        print(f"ERROR: Vanilla file not found: {vanilla_file}")
        return
    if not os.path.exists(uncertainty_file):
        print(f"ERROR: Uncertainty file not found: {uncertainty_file}")
        return
    # Read vanilla experiment results
    qns_orig_ans_dict = {}
    with open(vanilla_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    idx, ans = parts
                    qns_orig_ans_dict[idx] = ans
    # Read uncertainty experiment results and count option occurrences
    qns_count_dict = {}
    with open(uncertainty_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    idx, ans = parts
                    orig_idx = "_".join(idx.split("_")[:4])
                    if orig_idx not in qns_count_dict:
                        qns_count_dict[orig_idx] = {"A": 0, "B": 0, "C": 0, "D": 0}
                    if ans in qns_count_dict[orig_idx]:
                        qns_count_dict[orig_idx][ans] += 1
    # Process samples and compute entropy
    entropy = []
    labels = []
    processed_samples = 0
    for idx in gt_dict.keys():
        if idx in qns_count_dict and idx in qns_orig_ans_dict:
            counts = [
                qns_count_dict[idx]["A"],
                qns_count_dict[idx]["B"],
                qns_count_dict[idx]["C"],
                qns_count_dict[idx]["D"]
            ]
            en = compute_entropy(counts)
            vanilla_ans = qns_orig_ans_dict[idx]
            ground_truth = gt_dict[idx]
            is_correct = 1 if vanilla_ans == ground_truth else 0
            entropy.append(en)
            labels.append(is_correct)
            processed_samples += 1
    if len(entropy) == 0:
        print("No samples processed - check file paths and data")
        return
    # Calculate uncertainty scores (same as original)
    epsilon = 0.001
    recip_entropy = []
    for e in entropy:
        if e == 0.0:
            e = e + epsilon
        recip = 1/e
        recip_entropy.append(recip)
    min_val = min(recip_entropy)
    max_val = max(recip_entropy)
    if max_val == min_val:
        normalized_recip = [0.5] * len(recip_entropy)
    else:
        normalized_recip = [(val - min_val) / (max_val - min_val) for val in recip_entropy]
    y_true = np.array(labels)
    y_prob = np.array(normalized_recip)
    os.makedirs(args.output_dir, exist_ok=True)
    # Metrics
    accuracy = np.mean(y_true)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece, bin_confidences, bin_accuracies, bin_counts = expected_calibration_error(y_true, y_prob)
    coverage, risk, accuracy_curve, aurc = risk_coverage_peel_off_low(y_prob, y_true)
    # Plots
    risk_cov_plot = plot_risk_coverage_desc(coverage, risk, args.model, args.dataset, args.method, args.exp_type, os.path.join(args.output_dir, 'risk_coverage_plots'))
    acc_cov_plot = plot_accuracy_coverage(coverage, accuracy_curve, args.model, args.dataset, args.method, args.exp_type, os.path.join(args.output_dir, 'risk_coverage_plots'))
    calib_plot = plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, args.model, args.dataset, args.method, args.exp_type, os.path.join(args.output_dir, 'ece_calibration_plots'))
    dist_plot = plot_uncertainty_distribution(y_prob, y_true, args.model, args.dataset, args.method, args.exp_type, os.path.join(args.output_dir, 'uncertainty_distribution_plots'))
    # Save metrics summary
    summary_path = os.path.join(args.output_dir, "comprehensive_metrics.csv")
    row = {
        'Model': args.model,
        'Dataset': args.dataset,
        'Method': args.method,
        'Exp_Type': args.exp_type,
        'Accuracy': accuracy,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Brier_Score': brier,
        'ECE': ece,
        'AURC': aurc,
        'Samples': len(y_true),
        'Risk_Coverage_Plot': risk_cov_plot,
        'Accuracy_Coverage_Plot': acc_cov_plot,
        'Calibration_Plot': calib_plot,
        'Distribution_Plot': dist_plot
    }
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended metrics for {args.model}-{args.dataset}-{args.method}-{args.exp_type} to {summary_path}")
    print(pd.DataFrame([row]).T)

if __name__ == "__main__":
    main()
