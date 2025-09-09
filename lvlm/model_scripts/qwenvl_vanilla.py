import os
import csv
import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
import time
import warnings
from transformers import logging

# Suppress warnings and transformers logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']
    return None

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print iterations progress with animated bar"""
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # Clear the line completely and print the progress bar
    sys.stdout.write('\r' + ' ' * 120)  # Clear the line
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    
    if iteration == total:
        print()  # New line when complete

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the questions CSV file')
parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--dataset', type=str, choices=['blink', 'vsr'], required=True, help='Dataset name')
parser.add_argument('--type', type=str, choices=['orig', 'neg'], required=True, help='Type of question: orig or neg')
args = parser.parse_args()

if args.dataset == 'blink':
    indices = [f"val_Spatial_Relation_{i}" for i in range(1, 144)]
    choices = "A. Yes\nB. No"
elif args.dataset == 'vsr':
    indices = [f"val_Spatial_Reasoning_{i}" for i in range(111, 211)]
    choices = "A. True\nB. False"

output_prefix = "negated_" if args.type == "neg" else ""
output_dir = "vanilla_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{output_prefix}qwenvl_vanilla_{args.dataset}.txt")

# Clear existing file
if os.path.exists(output_file):
    open(output_file, 'w').close()

print("Loading QwenVL model...")
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded successfully!")
print()

total_samples = len(indices)
print(f"Starting {args.dataset.upper()} {args.type} vanilla processing...")
print(f"Total samples to process: {total_samples}")
print(f"Output file: {output_file}")
print("=" * 70)

start_time = time.time()

for i, idx in enumerate(indices):
    qns = get_qns_for_idx(args.input_csv, idx)
    if qns is None:
        print(f"\nWarning: Question not found for {idx}")
        continue
        
    img_path = os.path.join(args.input_image_folder, f"{idx}.jpg")
    
    if not os.path.exists(img_path):
        print(f"\nWarning: Image not found at {img_path}")
        continue

    prompt = (
        f"{qns}\nChoices:\n{choices}\n"
        "Return only the option (A or B), and nothing else.\n"
        "MAKE SURE your output is A or B"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Suppress stdout temporarily to avoid warning messages interfering with progress bar
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    with open(output_file, "a") as res:
        res.write(f"{idx} {output_text}\n")

    # Update progress bar
    elapsed = time.time() - start_time
    eta = (elapsed / (i + 1)) * (total_samples - i - 1) if i > 0 else 0
    suffix = f"({i+1}/{total_samples}) | ETA: {eta:.1f}s | Sample: {idx}"
    print_progress_bar(i + 1, total_samples, prefix='Progress:', suffix=suffix, length=40)

print()
print("=" * 70)
print(f"✓ Processing completed! Results saved to: {output_file}")
total_time = time.time() - start_time
print(f"✓ Total time: {total_time:.2f}s")
print(f"✓ Average time per sample: {total_time/total_samples:.2f}s")
