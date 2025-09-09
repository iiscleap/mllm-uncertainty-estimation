#!/usr/bin/env python3
import math
import csv
import numpy as np
import argparse
import os

# -------------------- KL + normalization (same as yours) --------------------
def kl_divergence(p, p_star):
    p_star = np.array(p_star, dtype=float)
    p = np.array(p, dtype=float)
    eps = 1e-10
    return np.sum(p_star * np.log((p_star + eps) / (p + eps)))

def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    """
    Adaptive, outlier-robust confidence transform.
    Input x are uncertainties (KL) -> invert, clip by percentiles, ECDF->power.
    Returns confidence in (0,1), higher = more confident.
    """
    x = np.asarray(x, dtype=float)
    eps = 1e-9
    x = 1.0 / (x + eps)                     # invert: lower KL -> higher conf
    if clip is not None:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    ranks = np.argsort(np.argsort(x)) + 1   # 1..N
    u = ranks / (len(x) + 1.0)              # (0,1)
    return u ** gamma                        # gamma<1 expands low, compresses high

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
            orig_idx = "_".join(idx.split("_")[:4])
            ans = ans[0]
            if orig_idx not in qns_count_dict:
                qns_count_dict[orig_idx] = [0, 0]  # [no_toggle, toggle]
            if ans == qns_orig_ans_dict[orig_idx]:
                qns_count_dict[orig_idx][0] += 1
            else:
                qns_count_dict[orig_idx][1] += 1

    kl_scores, labels = [], []
    with open(answer_file, "r") as f:
        for line in f:
            idx, ans = line.strip().split(" ", maxsplit=1)
            gold = ans.strip("()")
            pred = qns_orig_ans_dict.get(idx)

            counts = qns_count_dict.get(idx, [0.5, 0.5])  # neutral default
            total = sum(counts)
            p = [counts[0] / total, counts[1] / total]

            kl = kl_divergence(p, p_star)
            is_correct = 1 if pred == gold else 0

            kl_scores.append(kl)
            labels.append(is_correct)

    return np.array(kl_scores, float), np.array(labels, int)

# -------------------- Brier score (single metric) --------------------
def brier_score(y_true, y_prob):
    """
    Brier score for binary events: mean squared error between predicted
    probability and outcome (0/1). Lower is better. (Brier, 1950)
    """
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    return float(np.mean((y_prob - y_true) ** 2))

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Quantile-power gamma (<1 expands low scores)")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save per-example y_true,y_prob to CSV")
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset

    if dataset == "blink":
        answer_file = "answer_list.txt"
    else:
        answer_file = "dataset/VSR/answer_list.txt"

    vanilla = f"vanilla_output/{model}_vanilla_{dataset}.txt"
    pos_sample = f"perturb_sampling_output/{model}_perturb_sampling_{dataset}.txt"
    neg_sample = f"perturb_sampling_output/negated_{model}_perturb_sampling_{dataset}.txt"

    # KL and correctness targets (1 if greedy correct)
    pos_kl, labels_pos = calculate_kl_div(vanilla, pos_sample, [1.0, 0.0], answer_file)
    neg_kl, labels_neg = calculate_kl_div(vanilla, neg_sample, [0.0, 1.0], answer_file)
    y_true = labels_pos  # correctness of greedy baseline

    # Combine the two KLs and convert to confidence with adaptive normalization
    combined_kl = (pos_kl + neg_kl) / 2.0
    y_prob = quantile_power_normalize(combined_kl, gamma=args.gamma)

    # -------- Single reported metric: Brier score --------
    bs = brier_score(y_true, y_prob)
    print(f"Brier score for {model}-{dataset}: {bs:.6f}")

    if args.save_csv:
        os.makedirs("brier_outputs", exist_ok=True)
        out_csv = os.path.join("brier_outputs", f"{model}_{dataset}_brier.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true", "y_prob"])
            for t, p in zip(y_true.tolist(), y_prob.tolist()):
                w.writerow([t, f"{p:.6f}"])
        print(f"Saved per-example probabilities to {out_csv}")

if __name__ == "__main__":
    main()
