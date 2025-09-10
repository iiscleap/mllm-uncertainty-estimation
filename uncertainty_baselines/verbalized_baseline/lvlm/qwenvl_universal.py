import re
import os
import csv
import torch
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_qns_for_idx(filename, target_idx):
    """Get question for a specific index from CSV file."""
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']
    return None

def extract_guess_and_prob(output):
    """Extract guess and probability from model output."""
    guess_match = re.search(r"Guess:\s*(.*)", output)
    prob_match = re.search(r"Probability:\s*([0-1](?:\.\d+)?)", output)

    guess = guess_match.group(1).strip() if guess_match else "N/A"
    prob = prob_match.group(1).strip() if prob_match else "N/A"

    return guess, prob

def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
    if dataset_name.lower() == 'blink':
        return {
            'qns_filepath': "path/to/blink_data/question.csv",
            'img_folder': "path/to/blink_data/orig_images",
            'idx_prefix': "val_Spatial_Relation_",
            'idx_range': range(1, 144),  # 1 to 143
            'question_suffix': "\nChoices:\nA. Yes\nB. No"
        }
    elif dataset_name.lower() == 'vsr':
        return {
            'qns_filepath': "path/to/vsr_data/questions.csv",
            'img_folder': "path/to/vsr_data/orig_images",
            'idx_prefix': "val_Spatial_Reasoning_",
            'idx_range': range(1, 341),  # 1 to 340
            'question_suffix': "\nChoices:\nA. True\nB. False"
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'blink', 'vsr'")

def main():
    parser = argparse.ArgumentParser(description='Run Qwen-VL verbalized confidence on spatial reasoning datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['blink', 'vsr'], 
                        help='Dataset to use: blink or vsr')
    parser.add_argument('--output_dir', type=str, default='vlm_results', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Get dataset configuration
    config = get_dataset_config(args.dataset)
    
    # Load model and processor
    print("Loading Qwen-VL model...")
    model_id = "Qwen/Qwen-VL-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True).eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(args.output_dir, f"qwenvl_guess_prob_{args.dataset.lower()}.csv")
    
    # Process dataset
    print(f"Processing {args.dataset.upper()} dataset...")
    with open(output_path, mode="w", newline="", encoding="utf-8") as out_file:
        fieldnames = ["idx", "guess", "probability"]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in config['idx_range']:
            idx = f"{config['idx_prefix']}{i}"
            qns = get_qns_for_idx(config['qns_filepath'], idx)
            
            if qns is None:
                print(f"Warning: Question not found for {idx}")
                continue
                
            qns = f"{qns}{config['question_suffix']}"

            img_path = f"{config['img_folder']}/{idx}.jpg"
            
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Warning: Image not found for {idx}")
                continue

            prompt = (
                "Provide your answer and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the option and probability, no other words or explanation. For example:\n\n"
                "Guess: <most likely option, as short as possible; not a complete sentence, just the option!>\n"
                "Probability: <the probability between 0.0 and 1.0 that your guess is correct>\n\n"
                f"The question is: {qns}"
            )

            query = tokenizer.from_list_format([
                {'image': img_path},
                {'text': prompt}
            ])

            with torch.inference_mode():
                response, _ = model.chat(tokenizer, query=query, history=None)

            print(f"{idx}:\n{response}\n")

            guess, prob = extract_guess_and_prob(response)

            writer.writerow({
                "idx": idx,
                "guess": guess,
                "probability": prob
            })

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
