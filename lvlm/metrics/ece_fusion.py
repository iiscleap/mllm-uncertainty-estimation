import math
import csv
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def compute_accuracy(probabilities, labels):

    return np.mean(np.argmax(probabilities >= 0.5) == labels)

def compute_ece(probabilities, labels, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    # print(f"\n--- ECE Bin Details ---")
    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probabilities > bin_lower) & (probabilities <= bin_upper)
        bin_count = np.sum(mask)
        
        if bin_count > 0:
            bin_accuracy = np.mean(labels[mask])
            bin_confidence = np.mean(probabilities[mask])
            bin_weight = bin_count / len(probabilities)
            ece += np.abs(bin_confidence - bin_accuracy) * bin_weight

            # print(f"  Bin {i+1:02d} ({bin_lower:.2f}-{bin_upper:.2f}): Avg Confidence={bin_confidence:.4f}, Accuracy={bin_accuracy:.4f}, Count={bin_count}")
            
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)
        else:
            # Print empty bins as well for a complete picture
            # print(f"  Bin {i+1:02d} ({bin_lower:.2f}-{bin_upper:.2f}): Empty")
            bin_confidences.append(0)
            bin_accuracies.append(0)
            bin_counts.append(0)
            
    return ece, bin_confidences, bin_accuracies, bin_counts

def kl_divergence(p, p_star):
    p_star = np.array(p_star)
    p = np.array(p)
    epsilon = 1e-10
    return np.sum(p_star * np.log((p_star + epsilon) / (p + epsilon)))

def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    """
    Adaptive normalization:
      0) take reciprocal to convert uncertainty to confidence-like score
      1) optional percentile clipping,
      2) map to empirical quantiles (ECDF),
      3) apply concave power u^gamma to expand lows & compress highs.

    gamma in (0,1): more expansion at low end and compression at high end.
    """
    x = np.asarray(x, dtype=float)
    
    # Take reciprocal to convert uncertainty to confidence
    epsilon = 1e-9
    x = 1 / (x + epsilon)
    # Standardize the reciprocals
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return np.full_like(x, 0.5)
        
    x = (x - mean) / std
    
    # robust clipping to tame extreme outliers (optional but helpful)
    if clip is not None:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)

    # ECDF / rank to [0,1]
    ranks = np.argsort(np.argsort(x)) + 1     # 1..n
    u = ranks / (len(x) + 1.0)                # avoid 0/1 endpoints

    # concave power: expands low u, compresses high u
    y = u ** gamma
    return y

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
                qns_count_dict[orig_idx] = [0, 0]
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


# Average the KL scores first, then normalize the result using quantile_power_normalize
combined_kl = [(pos_kl[i] + neg_kl[i]) / 2 for i in range(len(pos_kl))]
y_prob = np.array(quantile_power_normalize(np.array(combined_kl)))

print(f"\nFirst 5 y_prob values for {model}-{dataset}: {y_prob[:5]}")

def plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, model, dataset, num_bins=10):
    output_folder = "ece_bin_plots_folder"
    os.makedirs(output_folder, exist_ok=True)

    bin_centers = np.linspace(0.05, 0.95, num_bins)

    # Normalize bin counts for color mapping (avoid division by zero)
    counts_arr = np.array(bin_counts)
    min_c, max_c = np.min(counts_arr), np.max(counts_arr)
    norm_counts = (counts_arr - min_c) / (max_c - min_c + 1e-9)
    colors = plt.cm.Blues(norm_counts)

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, bin_accuracies, width=0.1, color=colors, edgecolor='black', label='Bin Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibration Plot for {model} on {dataset}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_folder, f"{model}_{dataset}_calibration.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Calibration plot saved to {plot_filename}")

ece_score, bin_confidences, bin_accuracies, bin_counts = compute_ece(y_prob, y_true)
print(f"ECE for {model}-{dataset}: {ece_score:.4f}")
##scatter plots for y_prob and y_true
output_folder = "ece_scatter_plots_folder"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

y_true_np = np.array(y_true)
y_prob_np = np.array(y_prob)

correct_scores = y_prob_np[y_true_np == 1]
incorrect_scores = y_prob_np[y_true_np == 0]

plt.figure(figsize=(10, 6))
# Add jitter for better visualization
y_jitter_correct = np.random.normal(1, 0.05, size=len(correct_scores))
y_jitter_incorrect = np.random.normal(0, 0.05, size=len(incorrect_scores))

plt.scatter(correct_scores, y_jitter_correct, label='Correct (y_true=1)', color='blue', alpha=0.6, s=15)
plt.scatter(incorrect_scores, y_jitter_incorrect, label='Incorrect (y_true=0)', color='red', alpha=0.6, s=15)
print(f"Combined KL scores: {combined_kl}")

plt.xlabel('Combined KL Divergence (Uncertainty Score)')
plt.ylabel('Prediction Outcome')
plt.yticks([0, 1], ['Incorrect', 'Correct'])
plt.title(f'Uncertainty Score Distribution for {model} on {dataset}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlim(right=0.02)  # Limit x-axis to 0.5
# plt.xlim(left=0.0)  # Limit y-axis to 1.05
plt.savefig(os.path.join(output_folder, f"{model}_{dataset}_scatter.png"))
plt.close()

# Generate and save the plot
plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, model, dataset)
