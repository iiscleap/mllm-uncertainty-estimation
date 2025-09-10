import pandas as pd
import argparse
import os
from sklearn.metrics import roc_auc_score

def get_ground_truth_config(dataset_name):
    """Get ground truth file configuration for dataset."""
    if dataset_name.lower() == 'blink':
        return {
            'answer_file': "path/to/blink_data/answer_list.txt",
            'parse_format': 'space_separated',  # Format: "idx (label)"
            'expected_labels': ['A', 'B']
        }
    elif dataset_name.lower() == 'vsr':
        return {
            'answer_file': "path/to/vsr_data/answers.txt",
            'parse_format': 'space_separated',  # Format: "idx (label)"
            'expected_labels': ['A', 'B']
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'blink', 'vsr'")

def load_ground_truth(config):
    """Load ground truth answers based on dataset configuration."""
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

def main():
    parser = argparse.ArgumentParser(description='Calculate AUROC for verbalized confidence predictions')
    parser.add_argument('--dataset', type=str, required=True, choices=['blink', 'vsr'], 
                        help='Dataset used: blink or vsr')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., gemma3, llava, phi4, qwenvl)')
    parser.add_argument('--results_dir', type=str, default='vlm_results', 
                        help='Directory containing results CSV files')
    
    args = parser.parse_args()
    
    # Get dataset configuration
    config = get_ground_truth_config(args.dataset)
    
    # Load model predictions
    pred_file = os.path.join(args.results_dir, f"{args.model}_guess_prob_{args.dataset.lower()}.csv")
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    pred_df = pd.read_csv(pred_file)
    # print(f"Loaded {len(pred_df)} predictions from {pred_file}")
    
    # Load ground truth answers
    gt_dict = load_ground_truth(config)
    # print(f"Loaded {len(gt_dict)} ground truth labels")
    
    # Match predictions with ground truth and compute binary correctness
    y_true = []
    y_score = []
    
    matched = 0
    skipped = 0
    
    for _, row in pred_df.iterrows():
        idx = row['idx']
        pred_label = row['guess']
        pred_prob = row['probability']
        
        # Skip rows with missing probability
        if pred_prob == 'N/A' or pd.isna(pred_prob):
            skipped += 1
            continue
            
        try:
            pred_prob = float(pred_prob)
        except (ValueError, TypeError):
            skipped += 1
            continue
        
        # Check correctness
        true_label = gt_dict.get(idx)
        if true_label is not None:
            correct = int(pred_label == true_label)
            y_true.append(correct)
            y_score.append(pred_prob)
            matched += 1
        else:
            skipped += 1
    
    # print(f"Matched {matched} predictions with ground truth")
    # print(f"Skipped {skipped} predictions due to missing data")
    
    if len(y_true) == 0:
        print("No valid predictions found for AUROC calculation")
        return
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(y_true, y_score)
        print(f"\nDataset: {args.dataset.upper()}")
        print(f"Model: {args.model}")
        print(f"AUROC: {auroc:.4f}")
        
        # Additional statistics
        accuracy = sum(y_true) / len(y_true)
        avg_confidence = sum(y_score) / len(y_score)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average Confidence: {avg_confidence:.4f}")
        
    except ValueError as e:
        print(f"Error computing AUROC: {e}")
        print("This might happen if all predictions have the same label")

if __name__ == "__main__":
    main()
