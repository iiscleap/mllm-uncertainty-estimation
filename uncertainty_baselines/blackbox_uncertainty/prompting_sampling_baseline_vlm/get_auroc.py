import pandas as pd
from sklearn.metrics import roc_auc_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name (e.g. gemma3, llava, phi4, pixtral, qwenvl)')
parser.add_argument('--dataset', type=str, required=True, choices=['blink', 'vsr'], help='Dataset/task name (blink or vsr)')
parser.add_argument('--output', type=str, default='auroc_results.txt', help='Output file to append AUROC scores')
args = parser.parse_args()

pred_file = f"vlm_results/{args.model}_topk_sampling_{args.dataset}.csv"

gt_file = None
if args.dataset == "blink":
    gt_file = "path/to/blink_data/answer_list.txt"
elif args.dataset == "vsr":
    gt_file = "path/to/vsr_data/answers_subset.txt"

pred_df = pd.read_csv(pred_file)

gt_dict = {}
with open(gt_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            idx = parts[0]
            label = parts[1].strip("()")
            gt_dict[idx] = label

y_true = []
y_score = []

for _, row in pred_df.iterrows():
    idx = row['idx']
    pred_label = row['guess']
    pred_prob = row['probability']
    
    true_label = gt_dict.get(idx)
    if true_label is not None:
        correct = int(pred_label == true_label)
        y_true.append(correct)
        y_score.append(pred_prob)

print(y_true)
print(y_score)

auc = roc_auc_score(y_true, y_score)
print(f"AUROC: {auc:.4f}")

# Save to file
with open(args.output, 'a') as out_f:
    out_f.write(f"Model: {args.model}, Dataset: {args.dataset}, AUROC: {auc:.4f}\n")
