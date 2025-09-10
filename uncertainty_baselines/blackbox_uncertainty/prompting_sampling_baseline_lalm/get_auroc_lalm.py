import pandas as pd
from sklearn.metrics import roc_auc_score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name (desc_llm, qwen, salmonn)')
parser.add_argument('--task', type=str, required=True, choices=['count', 'duration', 'order'], help='Task name')
parser.add_argument('--output', type=str, default='auroc_results_lalm.txt', help='Output file to append AUROC scores')
args = parser.parse_args()

gt_path = f'path/to/temporal_dataset_augmented_100samples/{args.task}_task/{args.task}_subset_100samples.csv'
pred_path = f'lalm_results/{args.model}_topk_sampling_{args.task}.csv'

df_gt = pd.read_csv(gt_path)
df_pred = pd.read_csv(pred_path)
df = pd.merge(df_gt, df_pred, left_on='id', right_on='idx')
df['is_correct'] = (df['guess'] == df['correct']).astype(int)
auroc = roc_auc_score(df['is_correct'], df['probability'])
print(f"AUROC: {auroc:.4f}")
with open(args.output, 'a') as out_f:
    out_f.write(f"Model: {args.model}, Task: {args.task}, AUROC: {auroc:.4f}\n")
