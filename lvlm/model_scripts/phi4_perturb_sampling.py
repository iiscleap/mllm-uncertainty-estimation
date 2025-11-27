import os
import argparse
import csv
import torch
import time
import sys
import warnings
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Suppress warnings
warnings.filterwarnings("ignore")

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
parser.add_argument('--model_path', type=str, required=True, help='Path to Phi4 model')
parser.add_argument('--max_per_base', type=int, default=56, help='Maximum perturbed samples to process per base id')
parser.add_argument('--k', type=int, default=10, help='Number of samples per entry')
args = parser.parse_args()
if args.max_per_base > 56:
    raise SystemExit("Error: --max_per_base cannot be greater than 56; only 56 perturbed samples are available per base id")

if args.dataset == 'blink':
    choices = "A. Yes\nB. No"
elif args.dataset == 'vsr':
    choices = "A. True\nB. False"

output_prefix = "negated_" if args.type == "neg" else ""
output_dir = "perturb_sampling_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{output_prefix}phi4_perturb_sampling_{args.dataset}.txt")

# Clear existing file
if os.path.exists(output_file):
    open(output_file, 'w').close()

print("Loading Phi4 model...")
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype='auto'
).cuda()
generation_config = GenerationConfig.from_pretrained(args.model_path, 'generation_config.json')
print("Model loaded successfully!")
print()

# Count total entries first
total_entries = 0
with open(args.input_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    total_entries = sum(1 for row in reader)

k = args.k  # Number of samples per entry
total_samples = total_entries * k

print(f"Starting {args.dataset.upper()} {args.type} perturb sampling processing...")
print(f"Total CSV entries: {total_entries}")
print(f"Samples per entry: {k}")
print(f"Total samples to generate: {total_samples}")
print(f"Output file: {output_file}")
print("=" * 70)

start_time = time.time()
current_sample = 0
base_counts = {}

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

with open(args.input_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for entry_idx, row in enumerate(reader):
        idx = row["idx"]
        image_idx = row["image_idx"]
        qns = row["question"]
        # Determine base id and skip early if we've reached the per-base cap
        base_id = '_'.join(idx.split('_')[:4])
        if base_counts.get(base_id, 0) >= args.max_per_base:
            continue

        img_path = os.path.join(args.input_image_folder, f"{image_idx}.jpg")
        if not os.path.exists(img_path):
            print(f"\nWarning: Image not found at {img_path}")
            current_sample += k  # Skip all samples for this entry
            continue

        txt = f"{qns}\nChoices:\n{choices}\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
        prompt = f'{user_prompt}<|image_1|>{txt}{prompt_suffix}{assistant_prompt}'

        image = Image.open(img_path)
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

        # Generate k samples for this entry
        for i in range(k):
            new_idx = f"{idx}_sample{i}"
            
            # Suppress stdout temporarily to avoid warning messages interfering with progress bar
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                generate_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    num_beams=1,
                    max_new_tokens=1,
                    generation_config=generation_config,
                )
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            with open(output_file, "a") as resfile:
                resfile.write(f"{new_idx} {response}\n")

            current_sample += 1
            
            # Update progress bar
            elapsed = time.time() - start_time
            eta = (elapsed / current_sample) * (total_samples - current_sample) if current_sample > 0 else 0
            suffix = f"({current_sample}/{total_samples}) | Entry: {entry_idx+1}/{total_entries} | Sample: {i+1}/{k} | ETA: {eta:.1f}s"
            print_progress_bar(current_sample, total_samples, prefix='Progress:', suffix=suffix, length=40)

        # one perturbed id processed for this base
        base_counts[base_id] = base_counts.get(base_id, 0) + 1

print()
print("=" * 70)
print(f"✓ Processing completed! Results saved to: {output_file}")
total_time = time.time() - start_time
print(f"✓ Total time: {total_time:.2f}s")
print(f"✓ Average time per sample: {total_time/total_samples:.2f}s")
print(f"✓ Average time per entry: {total_time/total_entries:.2f}s")
