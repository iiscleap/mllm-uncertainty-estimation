import math
import csv
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import pandas as pd

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

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE) with bin details"""
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

    kl_scores = []
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
                kl = kl_divergence(p, p_star)
                is_correct = 1 if qns_orig_ans_dict.get(idx) == answer else 0
                kl_scores.append(kl)
                labels.append(is_correct)

    return kl_scores, labels

def plot_risk_coverage_desc(coverage, risk, model, task, save_dir="risk_coverage_plots"):
    """Plot risk-coverage curve"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(coverage, risk, linewidth=2, color='r')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Risk (error on answered set)")
    plt.title(f"Risk–Coverage for {model} on {task}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0) # Decreasing coverage from left to right
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{task}_risk_coverage.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out

def plot_accuracy_coverage(coverage, accuracy, model, task, save_dir="risk_coverage_plots"):
    """Plot accuracy-coverage curve"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(coverage, accuracy, linewidth=2, color='g')
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Accuracy on Answered Set")
    plt.title(f"Accuracy vs. Coverage for {model} on {task}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1.0, 0.0) # Decreasing coverage from left to right
    plt.ylim(0, 1)
    out = os.path.join(save_dir, f"{model}_{task}_accuracy_coverage.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out

def plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, model, task, num_bins=10):
    """Plot ECE calibration curve"""
    save_dir = "ece_calibration_plots"
    os.makedirs(save_dir, exist_ok=True)

    bin_centers = np.linspace(0.05, 0.95, num_bins)

    # Normalize bin counts for color mapping (avoid division by zero)
    counts_arr = np.array(bin_counts)
    min_c, max_c = np.min(counts_arr), np.max(counts_arr)
    norm_counts = (counts_arr - min_c) / (max_c - min_c + 1e-9)
    colors = plt.cm.Blues(norm_counts)

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, bin_accuracies, width=0.08, color=colors, edgecolor='black', label='Bin Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibration Plot for {model} on {task}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(save_dir, f"{model}_{task}_calibration.png")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    return plot_filename

def plot_uncertainty_distribution(y_prob, y_true, combined_kl, model, task):
    """Plot uncertainty score distribution"""
    save_dir = "uncertainty_distribution_plots"
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Prediction Outcome')
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.title(f'Confidence Score Distribution for {model} on {task}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out = os.path.join(save_dir, f"{model}_{task}_uncertainty_distribution.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out

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

def compute_all_metrics_with_plots(model, task):
    """Compute all metrics and generate plots for a given model and task"""
    
    orig_vanilla = f"old_vanilla_output/{model}_{task}_vanilla.txt"
    neg_uncertainty_exp = f"old_perturb_sampling_output/negated_{model}_{task}_perturb_sampling.txt"
    pos_uncertainty_exp = f"old_perturb_sampling_output/{model}_{task}_perturb_sampling.txt"

    try:
        # Calculate KL divergences
        pos_kl, labels = calculate_kl_div(orig_vanilla, pos_uncertainty_exp, [1.0, 0.0], task)
        neg_kl, _ = calculate_kl_div(orig_vanilla, neg_uncertainty_exp, [0.0, 1.0], task)

        # Combine KL scores
        combined_kl = [(p + n) / 2 for p, n in zip(pos_kl, neg_kl)]
        
        # Normalize using quantile power normalization
        y_prob = quantile_power_normalize(np.array(combined_kl))
        y_true = np.array(labels)

        # Calculate all metrics
        accuracy = np.mean(y_true)
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        
        # ECE with binning details
        ece, bin_confidences, bin_accuracies, bin_counts = expected_calibration_error(y_true, y_prob)
        
        # Risk-Coverage Analysis
        coverage, risk, accuracy_curve, aurc = risk_coverage_peel_off_low(y_prob, y_true)
        
        # Generate plots
        risk_cov_plot = plot_risk_coverage_desc(coverage, risk, model, task)
        acc_cov_plot = plot_accuracy_coverage(coverage, accuracy_curve, model, task)
        calib_plot = plot_calibration_curve(bin_confidences, bin_accuracies, bin_counts, model, task)
        dist_plot = plot_uncertainty_distribution(y_prob, y_true, combined_kl, model, task)
        
        # Save risk-coverage data
        os.makedirs("risk_coverage_plots", exist_ok=True)
        csv_path = os.path.join("risk_coverage_plots", f"{model}_{task}_risk_coverage.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["coverage", "risk", "accuracy"])
            # Use nan_to_num to handle the point at coverage=0
            for cov, r, acc in zip(coverage, np.nan_to_num(risk), np.nan_to_num(accuracy_curve)):
                w.writerow([f"{cov:.6f}", f"{r:.6f}", f"{acc:.6f}"])
        
        return {
            'model': model,
            'task': task,
            'accuracy': accuracy,
            'auroc': auroc,
            'auprc': auprc,
            'brier': brier,
            'ece': ece,
            'aurc': aurc,
            'n_samples': len(y_true),
            'plots': {
                'risk_coverage': risk_cov_plot,
                'accuracy_coverage': acc_cov_plot,
                'calibration': calib_plot,
                'distribution': dist_plot
            },
            'data_files': {
                'risk_coverage_csv': csv_path
            }
        }
        
    except Exception as e:
        print(f"Error processing {model}-{task}: {str(e)}")
        return None

def save_results_to_csv(results, output_file):
    """Save results to CSV file"""
    df_data = []
    for result in results:
        if result is not None:
            df_data.append({
                'Model': result['model'],
                'Task': result['task'],
                'Accuracy': result['accuracy'],
                'AUROC': result['auroc'],
                'AUPRC': result['auprc'],
                'Brier_Score': result['brier'],
                'ECE': result['ece'],
                'AURC': result['aurc'],
                'Samples': result['n_samples'],
                'Risk_Coverage_Plot': result['plots']['risk_coverage'],
                'Calibration_Plot': result['plots']['calibration']
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model (e.g., qwen7b)")
    parser.add_argument("--task", type=str, choices=["count", "order", "duration"], help="Task type")
    parser.add_argument("--all_models", action="store_true", help="Run for all available models")
    parser.add_argument("--all_tasks", action="store_true", help="Run for all tasks")
    parser.add_argument("--output", type=str, default="comprehensive_metrics_results.csv", help="Output CSV file")
    parser.add_argument("--abstention_step", type=float, default=0.1, help="Step for abstention table")

    args = parser.parse_args()

    results = []

    # Determine which models to process
    if args.all_models:
        # Find all available models by scanning the vanilla output directory
        vanilla_files = os.listdir("old_vanilla_output/")
        models = set()
        for file in vanilla_files:
            if file.endswith("_vanilla.txt") and not file.startswith("negated_"):
                parts = file.replace("_vanilla.txt", "").split("_")
                if len(parts) >= 2:
                    model = "_".join(parts[:-1])  # Everything except the last part (task)
                    models.add(model)
        models = sorted(list(models))
    else:
        if not args.model:
            parser.error("Either --model or --all_models must be specified")
        models = [args.model]

    # Determine which tasks to process
    if args.all_tasks:
        tasks = ["count", "order", "duration"]
    else:
        if not args.task:
            parser.error("Either --task or --all_tasks must be specified")
        tasks = [args.task]

    print(f"Processing models: {models}")
    print(f"Processing tasks: {tasks}")
    print("=" * 80)

    # Process each model-task combination
    for model in models:
        for task in tasks:
            print(f"\nProcessing {model} - {task}...")
            print("-" * 50)
            
            result = compute_all_metrics_with_plots(model, task)
            if result is not None:
                results.append(result)
                
                # Print individual results
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  AUROC: {result['auroc']:.4f}")
                print(f"  AUPRC: {result['auprc']:.4f}")
                print(f"  Brier Score: {result['brier']:.4f}")
                print(f"  ECE: {result['ece']:.4f}")
                print(f"  AURC: {result['aurc']:.6f}")
                print(f"  Samples: {result['n_samples']}")
                print(f"  Plots generated:")
                for plot_type, plot_path in result['plots'].items():
                    print(f"    {plot_type}: {plot_path}")
                
                # Print abstention table
                print(f"\nAbstention Analysis for {model}-{task}:")
                # We need to recalculate y_prob and y_true for the table
                orig_vanilla = f"old_vanilla_output/{model}_{task}_vanilla.txt"
                pos_uncertainty_exp = f"old_perturb_sampling_output/{model}_{task}_perturb_sampling.txt"
                neg_uncertainty_exp = f"old_perturb_sampling_output/negated_{model}_{task}_perturb_sampling.txt"
                
                pos_kl, labels = calculate_kl_div(orig_vanilla, pos_uncertainty_exp, [1.0, 0.0], task)
                neg_kl, _ = calculate_kl_div(orig_vanilla, neg_uncertainty_exp, [0.0, 1.0], task)
                combined_kl = [(p + n) / 2 for p, n in zip(pos_kl, neg_kl)]
                y_prob = quantile_power_normalize(np.array(combined_kl))
                y_true = np.array(labels)
                
                print_fixed_abstention_table(y_true, y_prob, step=args.abstention_step)

    # Save results to CSV
    if results:
        df = save_results_to_csv(results, args.output)
        print(f"\n{'='*80}")
        print(f"Results saved to {args.output}")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"Total processed: {len(results)} model-task combinations")
        if len(results) > 1:
            print(f"Average AUROC: {df['AUROC'].mean():.4f}")
            print(f"Average AUPRC: {df['AUPRC'].mean():.4f}")
            print(f"Average Brier: {df['Brier_Score'].mean():.4f}")
            print(f"Average ECE: {df['ECE'].mean():.4f}")
            print(f"Average AURC: {df['AURC'].mean():.6f}")
            
        print(f"\nGenerated plot directories:")
        print(f"  - risk_coverage_plots/")
        print(f"  - ece_calibration_plots/")
        print(f"  - uncertainty_distribution_plots/")
    else:
        print("No results to save.")
