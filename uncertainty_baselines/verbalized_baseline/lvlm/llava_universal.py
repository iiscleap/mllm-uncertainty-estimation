from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import csv
import os
import re
import argparse

def get_qns_for_idx(filename, target_idx):
    """Get question for a specific index from CSV file."""
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']
    return None

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
    parser = argparse.ArgumentParser(description='Run LLaVA verbalized confidence on spatial reasoning datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['blink', 'vsr'], 
                        help='Dataset to use: blink or vsr')
    parser.add_argument('--output_dir', type=str, default='vlm_results', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Get dataset configuration
    config = get_dataset_config(args.dataset)
    
    # Load model and processor
    print("Loading LLaVA model...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(args.output_dir, f"llava_guess_prob_{args.dataset.lower()}.csv")
    
    # Write header
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "guess", "probability"])

    print(f"Processing {args.dataset.upper()} dataset...")
    for i in config['idx_range']:
        idx = f"{config['idx_prefix']}{i}"
        qns = get_qns_for_idx(config['qns_filepath'], idx)
        
        if qns is None:
            print(f"Warning: Question not found for {idx}")
            continue
            
        qns = f"{qns}{config['question_suffix']}"

        img_path = os.path.join(config['img_folder'], f"{idx}.jpg")
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for {idx}")
            continue

        image = Image.open(img_path)

        prompt_text = (
            "Provide your answer and the probability that it is correct (0.0 to 1.0) for the following question. "
            "Give ONLY the option and probability, no other words or explanation. For example:\n\n"
            "Guess: <most likely option, as short as possible; not a complete sentence, just the option!>\n"
            "Probability: <the probability between 0.0 and 1.0 that your guess is correct>\n\n"
            f"The question is: {qns}"
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=50)

        res = processor.decode(output[0], skip_special_tokens=True)
        print(res)
        
        # Regex parsing - adapted for both datasets
        guess_match = re.search(r"\[/INST\]\s*([A-Da-d])", res)
        prob_match = re.search(r"Probability:\s*([0-9]*\.?[0-9]+)", res)

        guess = guess_match.group(1).strip() if guess_match else "N/A"
        prob = prob_match.group(1).strip() if prob_match else "N/A"

        print(f"{idx} {guess} {prob}")

        with open(output_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, guess, prob])

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
