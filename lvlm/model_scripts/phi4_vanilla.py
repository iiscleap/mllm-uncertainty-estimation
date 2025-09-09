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
parser.add_argument('--model_path', type=str, required=True, help='Path to Phi4 model')
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
output_file = os.path.join(output_dir, f"{output_prefix}phi4_vanilla_{args.dataset}.txt")

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

total_samples = len(indices)
print(f"Starting {args.dataset.upper()} {args.type} vanilla processing...")
print(f"Total samples to process: {total_samples}")
print(f"Output file: {output_file}")
print("=" * 70)

start_time = time.time()

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

for i, idx in enumerate(indices):
    qns = get_qns_for_idx(args.input_csv, idx)
    if qns is None:
        print(f"\nWarning: Question not found for {idx}")
        continue
        
    img_path = os.path.join(args.input_image_folder, f"{idx}.jpg")
    
    if not os.path.exists(img_path):
        print(f"\nWarning: Image not found at {img_path}")
        continue

    txt = f"{qns}\nChoices:\n{choices}\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
    prompt = f'{user_prompt}<|image_1|>{txt}{prompt_suffix}{assistant_prompt}'

    image = Image.open(img_path)
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

    # Suppress stdout temporarily to avoid warning messages interfering with progress bar
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            generation_config=generation_config,
        )
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = response.strip()

    with open(output_file, "a") as resfile:
        resfile.write(f"{idx} {response}\n")

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
