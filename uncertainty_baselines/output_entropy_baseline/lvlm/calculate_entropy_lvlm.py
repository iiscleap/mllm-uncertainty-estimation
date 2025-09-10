import math
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import argparse
import json

def compute_entropy(counts):
    """Compute Shannon entropy from option counts"""
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
    """Parse ground truth file and return id -> answer mapping"""
    gt_dict = {}
    
    if dataset == "blink" or dataset == "vsr":
        # Format: "val_Spatial_Relation_1 (B)"
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

def main():
    parser = argparse.ArgumentParser(description='Calculate entropy-based AUC for LVLM models')
    parser.add_argument('--model', required=True, 
                       choices=['llava', 'gemma3', 'phi4', 'pixtral', 'qwenvl', 'internvl3'])
    parser.add_argument('--method', required=True, 
                       choices=['perturb_sampling', 'sampling', 'perturb'])
    parser.add_argument('--dataset', required=True, choices=['blink', 'vsr'])
    parser.add_argument('--exp_type', default='orig', choices=['orig', 'neg'])
    
    args = parser.parse_args()
    
    print(f"Processing LVLM: {args.model} - {args.dataset} - {args.method} - {args.exp_type}")
    
    # Set ground truth file path
    if args.dataset == "blink":
        gt_file = "path/to/blink_data/answer_list.txt"
    elif args.dataset == "vsr":
        gt_file = "path/to/vsr_data/answers_subset.txt"
    
    # Load ground truth
    print(f"Loading ground truth from: {gt_file}")
    gt_dict = parse_ground_truth(gt_file, args.dataset)
    print(f"Loaded {len(gt_dict)} ground truth answers")
    
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
    
    print(f"Vanilla file: {vanilla_file}")
    print(f"Uncertainty file: {uncertainty_file}")
    
    # Check if files exist
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
    
    print(f"Read {len(qns_orig_ans_dict)} vanilla predictions")
    
    # Read uncertainty experiment results and count option occurrences
    qns_count_dict = {}
    with open(uncertainty_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    idx, ans = parts
                    # Extract original ID from perturbed ID
                    # e.g., "val_Spatial_Relation_1_contrast1_rephrased1_sample0" -> "val_Spatial_Relation_1"
                    orig_idx = "_".join(idx.split("_")[:4])  # Keep first 4 parts for vsr/blink format
                    
                    if orig_idx not in qns_count_dict:
                        qns_count_dict[orig_idx] = {"A": 0, "B": 0, "C": 0, "D": 0}
                    
                    # Count the answer
                    if ans in qns_count_dict[orig_idx]:
                        qns_count_dict[orig_idx][ans] += 1
    
    print(f"Read uncertainty predictions for {len(qns_count_dict)} samples")
    
    # Process samples and compute entropy
    entropy = []
    labels = []
    
    processed_samples = 0
    for idx in gt_dict.keys():
        if idx in qns_count_dict and idx in qns_orig_ans_dict:
            # Get counts for this sample
            counts = [
                qns_count_dict[idx]["A"],
                qns_count_dict[idx]["B"],
                qns_count_dict[idx]["C"],
                qns_count_dict[idx]["D"]
            ]
            
            # Compute entropy
            en = compute_entropy(counts)
            
            # Check if vanilla prediction was correct
            vanilla_ans = qns_orig_ans_dict[idx]
            ground_truth = gt_dict[idx]
            is_correct = 1 if vanilla_ans == ground_truth else 0
            
            entropy.append(en)
            labels.append(is_correct)
            processed_samples += 1
    
    print(f"Processed {processed_samples} samples")
    print(f"Correct predictions: {sum(labels)}")
    if len(labels) > 0:
        print(f"Accuracy: {sum(labels)/len(labels):.3f}")
    else:
        print("Accuracy: N/A (0 samples processed)")
    
    if len(entropy) == 0:
        print("No samples processed - check file paths and data")
        return
    
    # Calculate AUC (same logic as original)
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
    
    auc_score = roc_auc_score(labels, normalized_recip)
    print(f"AUC-ROC: {auc_score:.4f}")
    
    # Save arrays
    output_dir = "../output_sampling_arrays"
    os.makedirs(output_dir, exist_ok=True)
    
    filename_prefix = f"{args.model}_{args.dataset}_{args.exp_type}_{args.method}"
    np.save(os.path.join(output_dir, f"{filename_prefix}_y_true.npy"), np.array(labels))
    np.save(os.path.join(output_dir, f"{filename_prefix}_y_prob.npy"), np.array(normalized_recip))
    
    return auc_score

if __name__ == "__main__":
    main()
