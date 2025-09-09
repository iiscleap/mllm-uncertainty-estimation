import math
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import argparse
import matplotlib.pyplot as plt

# -------------------- your utilities (unchanged) --------------------
def kl_divergence(p, p_star):
    p_star = np.array(p_star)
    p = np.array(p)
    epsilon = 1e-10
    return np.sum(p_star * np.log((p_star + epsilon) / (p + epsilon)))

def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    """
    Adaptive, outlier-robust confidence transform.
    Higher output = higher confidence.
    """
    x = np.asarray(x, dtype=float)
    epsilon = 1e-9
    x = 1 / (x + epsilon)                  # invert KL: lower KL -> higher confidence
    if clip is not None:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    ranks = np.argsort(np.argsort(x)) + 1  # 1..N
    u = ranks / (len(x) + 1.0)             # (0,1)
    return u ** gamma                       # gamma<1 expands low, compresses high

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

# -------------------- new: risk–coverage helpers --------------------
def risk_coverage_peel_off_low(conf, correct):
    """
    Start with answering ALL items, then gradually abstain on the lowest-confidence ones.
    conf    : (N,) confidence scores (higher = more confident)
    correct : (N,) in {0,1} for greedy answers

    Returns:
        coverage: (N+1,) coverage after abstaining on k lowest-confidence items, k=0..N
        risk:     (N+1,) risk on the remaining answered set at each abstention level
        accuracy: (N+1,) accuracy on the remaining answered set
        aurc:     area under the risk-coverage curve (lower is better)
    """
    conf    = np.asarray(conf, float)
    correct = np.asarray(correct, int)
    N = len(conf)

    # Sort *ascending* by confidence (worst first) to peel off the prefix
    order_low_first = np.argsort(conf, kind="mergesort")
    corr_low_first  = correct[order_low_first]

    # Cumulative correct count among the k lowest-confidence (abstained) items
    cum_correct_low = np.concatenate(([0], np.cumsum(corr_low_first)))  # length N+1

    total_correct = correct.sum()
    k = np.arange(N+1)  # Number of items abstained (from 0 to N)
    kept = N - k        # Number of items answered

    # Correct count among the items that are kept (answered)
    correct_kept = total_correct - cum_correct_low
    
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = correct_kept / np.maximum(kept, 1)  # Accuracy on the answered set
        risk = 1.0 - accuracy                         # Risk on the answered set

    coverage = kept / N  # Coverage from 1.0 down to 0.0

    # AURC is the area under the risk-coverage curve
    mask = kept > 0  # Use only finite points (risk is undefined at coverage=0)
    aurc = np.trapz(risk[mask][::-1], coverage[mask][::-1])

    return coverage, risk, accuracy, aurc

def plot_risk_coverage_desc(coverage, risk, model, dataset, save_dir="risk_coverage_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(coverage, risk, linewidth=2, color='r')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Risk (error on answered set)")
    plt.title(f"Risk–Coverage for {model} on {dataset}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0) # Decreasing coverage from left to right
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{dataset}_risk_coverage.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Risk–coverage plot saved to {out}")

def plot_accuracy_coverage(coverage, accuracy, model, dataset, save_dir="risk_coverage_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(coverage, accuracy, linewidth=2, color='g')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Accuracy on Answered Set")
    plt.title(f"Accuracy vs. Coverage for {model} on {dataset}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0) # Decreasing coverage from left to right
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{dataset}_accuracy_coverage.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Accuracy-coverage plot saved to {out}")

def print_fixed_abstention_table(correct, conf, step=0.1):
    """
    Prints coverage, risk, and accuracy at fixed abstention levels.
    """
    N = len(conf)
    order = np.argsort(-conf, kind="mergesort") # Sort descending by confidence
    corr_sorted = correct[order]
    
    print(f"{'Abstain':<10} | {'Coverage':<10} | {'Risk':<10} | {'Accuracy':<10}")
    print("-"*49)

    for abstain_frac in np.arange(0.0, 1.0 + 1e-9, step):
        k = int(round((1 - abstain_frac) * N)) # Number of items to keep
        if k <= 0:
            cov, rk, acc = 0.0, 0.0, 0.0
        else:
            cov = k / N
            acc = corr_sorted[:k].mean()
            rk = 1.0 - acc
        print(f"{abstain_frac:>9.0%} | {cov:<10.3f} | {rk:<10.3f} | {acc:<10.3f}")

# -------------------- main --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--abstention_step', type=float, default=0.1, help='Step for abstention grid (e.g., 0.1 for 10%)')
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

# KL and correctness
pos_kl, labels_pos = calculate_kl_div(vanilla, pos_sample, [1.0, 0.0], answer_file)
neg_kl, labels_neg = calculate_kl_div(vanilla, neg_sample, [0.0, 1.0], answer_file)
y_true = np.array(labels_pos)  # 1 if greedy prediction equals gold, else 0

# Accuracy
accuracy = np.mean(y_true)
print(f"Accuracy for {model}-{dataset}: {accuracy:.4f}")

# Combine uncertainties then produce confidence via adaptive normalization
combined_kl = [(pos_kl[i] + neg_kl[i]) / 2 for i in range(len(pos_kl))]
y_prob = quantile_power_normalize(np.array(combined_kl), gamma=0.5)

print(f"\nFirst 5 y_prob values for {model}-{dataset}: {y_prob[:5]}")

# AUROC (ranking metric; unaffected by monotone transforms)
auroc = roc_auc_score(y_true, y_prob)
print(f"AUROC for {model}-{dataset}: {auroc:.4f}")

# -------------------- Risk–Coverage + AURC --------------------
coverage, risk, accuracy, aurc = risk_coverage_peel_off_low(y_prob, y_true)
print(f"AURC for {model}-{dataset}: {aurc:.6f}")

# Save plots
plot_risk_coverage_desc(coverage, risk, model, dataset, save_dir="risk_coverage_plots")
plot_accuracy_coverage(coverage, accuracy, model, dataset, save_dir="risk_coverage_plots")

# Report a table at fixed abstention levels
print_fixed_abstention_table(y_true, y_prob, step=args.abstention_step)

# Save the curve points to CSV for later aggregation
os.makedirs("risk_coverage_plots", exist_ok=True)
csv_path = os.path.join("risk_coverage_plots", f"{model}_{dataset}_risk_coverage.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["coverage", "risk", "accuracy"])
    # Use nan_to_num to handle the point at coverage=0
    for cov, r, acc in zip(coverage, np.nan_to_num(risk), np.nan_to_num(accuracy)):
        w.writerow([f"{cov:.6f}", f"{r:.6f}", f"{acc:.6f}"])
print(f"Risk-coverage and accuracy data saved to {csv_path}")
