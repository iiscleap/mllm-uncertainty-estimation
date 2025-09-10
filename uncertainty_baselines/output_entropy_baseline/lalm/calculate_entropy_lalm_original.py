import math
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['count', 'duration', 'order'])
    parser.add_argument('--model', required=True, choices=['salmonn', 'qwen', 'desc_llm'])
    parser.add_argument('--method', required=True, choices=['sampling', 'perturb_sampling'])
    parser.add_argument('--exp_type', required=True, choices=['orig', 'neg'])
    
    args = parser.parse_args()
    
    task = args.task
    model = args.model
    method = args.method
    exp_type = args.exp_type
    
    print(f"Processing: {model} - {task} - {method} - {exp_type}")
    
    # Construct file paths exactly like the original script
    if exp_type == "orig":
        vanilla_exp = f"{model}/orig/vanilla/{model}_{task}_vanilla.txt"
        
        if model == "salmonn" and "perturb_sampling" in method:
            uncertainty_exp = f"{model}/orig/{method}/{model}_{task}_{method}_cleaned.txt"
        else:
            uncertainty_exp = f"{model}/orig/{method}/{model}_{task}_{method}.txt"
    
    elif exp_type == "neg":
        vanilla_exp = f"{model}/neg/vanilla/negated_{model}_{task}_vanilla.txt"
        
        if model == "salmonn" and "perturb" in method:
            uncertainty_exp = f"{model}/neg/{method}/negated_{model}_{task}_{method}_cleaned.txt"
        else:
            uncertainty_exp = f"{model}/neg/{method}/negated_{model}_{task}_{method}.txt"
    
    print(f"Vanilla file: {vanilla_exp}")
    print(f"Uncertainty file: {uncertainty_exp}")
    
    # Check if files exist
    if not os.path.exists(vanilla_exp):
        print(f"ERROR: Vanilla file not found: {vanilla_exp}")
        return
    
    if not os.path.exists(uncertainty_exp):
        print(f"ERROR: Uncertainty file not found: {uncertainty_exp}")
        return
    
    # Read vanilla experiment results (same as original)
    qns_orig_ans_dict = {}
    with open(vanilla_exp, "r") as data2:
        for line in data2:
            line = line.strip(" \n")
            if line:
                parts = line.split(" ", maxsplit=1)
                if len(parts) >= 2:
                    idx, ans = parts
                    qns_orig_ans_dict[idx] = ans[0] if ans else ""
    
    print(f"Read {len(qns_orig_ans_dict)} vanilla predictions")
    
    # Read uncertainty experiment results (same as original)
    qns_count_dict = {}
    with open(uncertainty_exp, "r") as data:
        for line in data:
            line = line.strip(" \n")
            if line:
                parts = line.split(" ", maxsplit=1)
                if len(parts) >= 2:
                    idx, ans = parts
                    orig_idx = idx.split("_")[0]
                    ans = ans[0] if ans else ""
                    
                    if orig_idx not in qns_count_dict.keys():
                        qns_count_dict[orig_idx] = [0, 0, 0, 0]
                    
                    if ans == "A":
                        qns_count_dict[orig_idx][0] += 1
                    elif ans == "B":
                        qns_count_dict[orig_idx][1] += 1
                    elif ans == "C":
                        qns_count_dict[orig_idx][2] += 1
                    elif ans == "D":
                        qns_count_dict[orig_idx][3] += 1
    
    print(f"Read uncertainty predictions for {len(qns_count_dict)} samples")
    
    # Process with subset file (same as original)
    entropy = []
    labels = []
    
    with open(f"subset/{task}_subset_100samples.csv", mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']
            answer = row['correct']
            
            if idx in qns_count_dict and idx in qns_orig_ans_dict:
                en = compute_entropy(qns_count_dict[idx])
                is_correct = 1 if qns_orig_ans_dict[idx] == answer else 0
                
                entropy.append(en)
                labels.append(is_correct)
    
    print(f"Processed {len(entropy)} samples")
    
    if len(entropy) == 0:
        print("No samples processed - check file paths and data")
        return
    
    # Calculate AUC (same as original)
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
    
    np.save(os.path.join(output_dir, f"{model}_{task}_{exp_type}_{method}_y_true.npy"), np.array(labels))
    np.save(os.path.join(output_dir, f"{model}_{task}_{exp_type}_{method}_y_prob.npy"), np.array(normalized_recip))
    
    return auc_score

if __name__ == "__main__":
    main()
