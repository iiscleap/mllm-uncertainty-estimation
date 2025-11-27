import os
import csv
import base64
import argparse
from vllm import LLM
from vllm.sampling_params import SamplingParams
import sys
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def file_to_data_url(file_path: str):
    """Convert a local image file to a data URL."""
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    return f"data:{mime_type};base64,{encoded_string}"

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
parser.add_argument('--k', type=int, default=10, help='Number of samples per entry')
parser.add_argument('--max_per_base', type=int, default=56, help='Maximum perturbed samples to process per base id')
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
output_file = os.path.join(output_dir, f"{output_prefix}pixtral_perturb_sampling_{args.dataset}.txt")

# Clear existing file
if os.path.exists(output_file):
    open(output_file, 'w').close()

print("Loading Pixtral model...")
model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=1, temperature=1.0, top_p=0.95, n=1, best_of=1)

llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    tokenizer_mode="mistral",
    load_format="mistral",
    config_format="mistral",
)
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

with open(args.input_csv, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for entry_idx, row in enumerate(reader):
        idx = row["idx"]
        image_idx = row["image_idx"]
        qns = row["question"]
        # Determine base id early and skip if we've already reached the per-base limit
        base_id = '_'.join(idx.split('_')[:4])
        if base_counts.get(base_id, 0) >= args.max_per_base:
            continue

        img_path = os.path.join(args.input_image_folder, f"{image_idx}.jpg")

        if not os.path.exists(img_path):
            print(f"\nWarning: Image not found at {img_path}")
            current_sample += k  # Skip all samples for this entry
            continue

        prompt = (
            f"{qns}\nChoices:\n{choices}\n"
            "Return only the option (A or B), and nothing else.\n"
            "MAKE SURE your output is A or B"
        )

        image_source = file_to_data_url(img_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_source}},
                ],
            },
        ]

        # Generate k samples for this entry
        for i in range(k):
            new_idx = f"{idx}_sample{i}"
            
            # Suppress stdout temporarily to avoid warning messages interfering with progress bar
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                outputs = llm.chat(messages, sampling_params=sampling_params)
                final_res = outputs[0].outputs[0].text.strip()
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout

            with open(output_file, "a") as resfile:
                resfile.write(f"{new_idx} {final_res}\n")

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
