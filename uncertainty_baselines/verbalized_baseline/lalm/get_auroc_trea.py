import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate AUROC for TREA dataset results')
    parser.add_argument('--task', type=str, choices=['count', 'duration', 'order'], 
                       required=True, help='Task to evaluate: count, duration, or order')
    parser.add_argument('--model', type=str, default='qwen', 
                       help='Model name (for file naming)')
    parser.add_argument('--results_dir', type=str, default='lalm_results',
                       help='Directory containing model prediction results (relative to script location)')
    parser.add_argument('--dataset_path', type=str, default='/home/debarpanb/VLM_project/TREA_dataset',
                       help='Path to TREA dataset')
    
    args = parser.parse_args()
    
    # Load ground truth
    gt_path = f'{args.dataset_path}/{args.task}/{args.task}.csv'
    print(f"Loading ground truth from: {gt_path}")
    df_gt = pd.read_csv(gt_path)
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir)
    pred_path = os.path.join(results_dir, f'{args.model}_guess_prob_{args.task}.csv')
    
    print(f"Loading predictions from: {pred_path}")
    
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found: {pred_path}")
        return
    
    df_pred = pd.read_csv(pred_path)
    
    # Convert id columns to same type for merging
    df_gt['id'] = df_gt['id'].astype(str)
    df_pred['idx'] = df_pred['idx'].astype(str)
    
    # Merge ground truth and predictions
    df = pd.merge(df_gt, df_pred, left_on='id', right_on='idx', how='inner')
    
    print(f"Successfully matched {len(df)} samples out of {len(df_gt)} ground truth samples")
    
    if len(df) == 0:
        print("Error: No matching samples found between ground truth and predictions")
        return
    
    # Check for missing predictions
    missing_guess = df['guess'].isna() | (df['guess'] == '')
    missing_prob = df['probability'].isna() | (df['probability'] == '')
    
    if missing_guess.any():
        print(f"Warning: {missing_guess.sum()} samples have missing guesses")
    if missing_prob.any():
        print(f"Warning: {missing_prob.sum()} samples have missing probabilities")
    
    # Filter out samples with missing data
    df_clean = df[(~missing_guess) & (~missing_prob)].copy()
    print(f"Using {len(df_clean)} samples for AUROC calculation")
    
    if len(df_clean) == 0:
        print("Error: No valid samples for AUROC calculation")
        return
    
    # Calculate correctness
    df_clean['is_correct'] = (df_clean['guess'] == df_clean['correct']).astype(int)
    
    # Convert probability to float
    try:
        df_clean['probability'] = pd.to_numeric(df_clean['probability'], errors='coerce')
        df_clean = df_clean.dropna(subset=['probability'])
    except Exception as e:
        print(f"Error converting probabilities to numeric: {e}")
        return
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(df_clean['is_correct'], df_clean['probability'])
        
        # Print detailed results
        print(f"\n=== AUROC Results for {args.task.upper()} task ===")
        print(f"Model: {args.model}")
        print(f"Task: {args.task}")
        print(f"Total samples: {len(df_clean)}")
        print(f"Correct predictions: {df_clean['is_correct'].sum()}")
        print(f"Accuracy: {df_clean['is_correct'].mean():.4f}")
        print(f"AUROC: {auroc:.4f}")
        
        # Additional statistics
        print(f"\nProbability Statistics:")
        print(f"Mean probability: {df_clean['probability'].mean():.4f}")
        print(f"Std probability: {df_clean['probability'].std():.4f}")
        print(f"Min probability: {df_clean['probability'].min():.4f}")
        print(f"Max probability: {df_clean['probability'].max():.4f}")
        
    except Exception as e:
        print(f"Error calculating AUROC: {e}")
        print("This might happen if all predictions are correct or all are incorrect")

if __name__ == "__main__":
    main()
