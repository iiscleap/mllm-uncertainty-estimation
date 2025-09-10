import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def get_ground_truth_config(dataset_name):
    if dataset_name.lower() == 'blink':
        return {
            'answer_file': "path/to/blink_data/answer_list.txt",
            'parse_format': 'space_separated',
            'expected_labels': ['A', 'B']
        }
    elif dataset_name.lower() == 'vsr':
        return {
            'answer_file': "path/to/vsr_data/answers.txt",
            'parse_format': 'space_separated',
            'expected_labels': ['A', 'B']
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'blink', 'vsr'")

def load_ground_truth(config):
    gt_dict = {}
    if not os.path.exists(config['answer_file']):
        raise FileNotFoundError(f"Ground truth file not found: {config['answer_file']}")
    with open(config['answer_file'], "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                idx = parts[0]
                label = parts[1].strip("()")
                gt_dict[idx] = label
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
    conf    = np.asarray(conf, float)
    correct = np.asarray(correct, int)
    N = len(conf)
    order_low_first = np.argsort(conf, kind="mergesort")
    corr_low_first  = correct[order_low_first]
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

def plot_risk_coverage_desc(coverage, risk, model, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(coverage, risk, linewidth=2, color='r')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Risk (error on answered set)")
    plt.title(f"Risk–Coverage for {model} on {dataset}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0)
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{dataset}_risk_coverage.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_accuracy_coverage(coverage, accuracy, model, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(coverage, accuracy, linewidth=2, color='g')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Accuracy on Answered Set")
    plt.title(f"Accuracy vs. Coverage for {model} on {dataset}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0)
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{dataset}_accuracy_coverage.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, model, dataset, save_dir, num_bins=10):
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
    plt.title(f'Calibration Plot for {model} on {dataset}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f"{model}_{dataset}_calibration.png")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    return plot_filename

def plot_uncertainty_distribution(y_prob, y_true, model, dataset, save_dir):
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
    plt.title(f'Confidence Score Distribution for {model} on {dataset}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    out = os.path.join(save_dir, f"{model}_{dataset}_uncertainty_distribution.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['blink', 'vsr'])
    parser.add_argument('--results_dir', type=str, default='vlm_results')
    parser.add_argument('--output_dir', type=str, default='comprehensive_results')
    args = parser.parse_args()
    config = get_ground_truth_config(args.dataset)
    pred_file = os.path.join(args.results_dir, f"{args.model}_guess_prob_{args.dataset.lower()}.csv")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    pred_df = pd.read_csv(pred_file)
    gt_dict = load_ground_truth(config)
    y_true = []
    y_prob = []
    for _, row in pred_df.iterrows():
        idx = str(row['idx'])
        pred_label = str(row['guess'])
        pred_prob = row['probability']
        if pred_prob == 'N/A' or pd.isna(pred_prob):
            continue
        try:
            pred_prob = float(pred_prob)
        except (ValueError, TypeError):
            continue
        true_label = gt_dict.get(idx)
        if true_label is not None:
            correct = int(pred_label == true_label)
            y_true.append(correct)
            y_prob.append(pred_prob)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    os.makedirs(args.output_dir, exist_ok=True)
    # Metrics
    accuracy = np.mean(y_true)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece, bin_confidences, bin_accuracies, bin_counts = expected_calibration_error(y_true, y_prob)
    coverage, risk, accuracy_curve, aurc = risk_coverage_peel_off_low(y_prob, y_true)
    # Plots
    risk_cov_plot = plot_risk_coverage_desc(coverage, risk, args.model, args.dataset, os.path.join(args.output_dir, 'risk_coverage_plots'))
    acc_cov_plot = plot_accuracy_coverage(coverage, accuracy_curve, args.model, args.dataset, os.path.join(args.output_dir, 'risk_coverage_plots'))
    calib_plot = plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, args.model, args.dataset, os.path.join(args.output_dir, 'ece_calibration_plots'))
    dist_plot = plot_uncertainty_distribution(y_prob, y_true, args.model, args.dataset, os.path.join(args.output_dir, 'uncertainty_distribution_plots'))
    # Save metrics summary
    summary_path = os.path.join(args.output_dir, "comprehensive_metrics.csv")
    row = {
        'Model': args.model,
        'Dataset': args.dataset,
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
    import csv
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended metrics for {args.model}-{args.dataset} to {summary_path}")
    print(pd.DataFrame([row]).T)

if __name__ == "__main__":
    main()
