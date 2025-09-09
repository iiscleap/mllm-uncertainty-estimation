import argparse
import os
import csv
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import sys
import time
import warnings
from transformers import logging

# Suppress warnings and transformers logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print iterations progress with animated bar"""
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # Clear the line completely and print the progress bar
    sys.stdout.write('\r' + ' ' * 150)  # Clear the line (longer for perturb sampling)
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
    choices = "A. Yes\nB. No"
elif args.dataset == 'vsr':
    choices = "A. True\nB. False"

output_prefix = "negated_" if args.type == "neg" else ""
output_dir = "perturb_sampling_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{output_prefix}llava_perturb_sampling_{args.dataset}.txt")

# Clear existing file
if os.path.exists(output_file):
    open(output_file, 'w').close()

print("Loading Llava model...")
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto"
)
print("Model loaded successfully!")
print()

# Count total entries first
total_entries = 0
with open(args.input_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    total_entries = sum(1 for row in reader)

k = 10  # Number of samples per entry
total_samples = total_entries * k

print(f"Starting {args.dataset.upper()} {args.type} perturb sampling processing...")
print(f"Total CSV entries: {total_entries}")
print(f"Samples per entry: {k}")
print(f"Total samples to generate: {total_samples}")
print(f"Output file: {output_file}")
print("=" * 70)

start_time = time.time()
current_sample = 0

with open(args.input_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for entry_idx, row in enumerate(reader):
        idx = row["idx"]
        image_idx = row["image_idx"]
        qns = row["question"]

        img_path = os.path.join(args.input_image_folder, f"{image_idx}.jpg")
        
        if not os.path.exists(img_path):
            print(f"\nWarning: Image not found at {img_path}")
            current_sample += k  # Skip all samples for this entry
            continue
            
        txt = f"{qns}\nChoices:\n{choices}\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
        image = Image.open(img_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": txt},
                    {"type": "image"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

        # Generate k samples for this entry
        for i in range(k):
            new_idx = f"{idx}_sample{i}"
            
            # Suppress stdout temporarily to avoid warning messages interfering with progress bar
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                output = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    num_beams=1,
                    max_new_tokens=1,
                )
                res = processor.decode(output[0], skip_special_tokens=True).strip()
                res = res[-1]
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout

            with open(output_file, "a") as resfile:
                resfile.write(f"{new_idx} {res}\n")

            current_sample += 1
            
            # Update progress bar
            elapsed = time.time() - start_time
            eta = (elapsed / current_sample) * (total_samples - current_sample) if current_sample > 0 else 0
            suffix = f"({current_sample}/{total_samples}) | Entry: {entry_idx+1}/{total_entries} | Sample: {i+1}/{k} | ETA: {eta:.1f}s"
            print_progress_bar(current_sample, total_samples, prefix='Progress:', suffix=suffix, length=40)

print()
print("=" * 70)
print(f"✓ Processing completed! Results saved to: {output_file}")
total_time = time.time() - start_time
print(f"✓ Total time: {total_time:.2f}s")
print(f"✓ Average time per sample: {total_time/total_samples:.2f}s")
print(f"✓ Average time per entry: {total_time/total_entries:.2f}s")
