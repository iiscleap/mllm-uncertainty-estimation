import os
import csv
import torch
import argparse
import sys
import time
import warnings
from transformers import WhisperFeatureExtractor
from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample

# Suppress warnings
warnings.filterwarnings("ignore")

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

def get_prompt(question, optionA, optionB, optionC, optionD):
    return f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

def main(args):
    # Set the device
    device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0"
    print(f"Using device: {device}")
    
    cfg_path = "configs/decode_config_calibration.yaml"

    output_folder = "vanilla_output"
    os.makedirs(output_folder, exist_ok=True)

    if args.task == "neg":
        output_file_path = os.path.join(output_folder, f"negated_salmonn_{args.task}_vanilla.txt")
    else:
        output_file_path = os.path.join(output_folder, f"salmonn_{args.task}_vanilla.txt")

    cfg = Config(argparse.Namespace(cfg_path=cfg_path, device=device, options=None))
    model = SALMONN.from_config(cfg.config.model)
    model.to(device)
    model.eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    # Count total samples for progress tracking
    total_samples = 0
    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        total_samples = sum(1 for _ in reader)

    print(f"Processing {total_samples} samples from {args.csv_path}")
    print(f"Output will be saved to: {output_file_path}")
    print("=" * 70)
    
    start_time = time.time()

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            idx = row['id']
            wav_path = os.path.join(args.wav_folder, f"{idx}.wav")
            prompt_text = get_prompt(row['question'], row['optionA'], row['optionB'], row['optionC'], row['optionD'])

            samples = prepare_one_sample(wav_path, wav_processor)
            
            # Ensure samples are on the correct device
            if isinstance(samples, dict):
                for key in samples:
                    if isinstance(samples[key], torch.Tensor):
                        samples[key] = samples[key].to(device)
            elif isinstance(samples, torch.Tensor):
                samples = samples.to(device)
            
            full_prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())]

            with torch.cuda.amp.autocast(dtype=torch.float16):
                text = model.generate(samples, cfg.config.generate, prompts=full_prompt)

            with open(output_file_path, "a") as f:
                f.write(f"{idx} {text}\n")

            # Update progress bar
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total_samples - i - 1) if i > 0 else 0
            suffix = f"({i+1}/{total_samples}) | ETA: {eta:.1f}s | Sample: {idx}"
            print_progress_bar(i + 1, total_samples, prefix='Progress:', suffix=suffix, length=40)

    print()
    print("=" * 70)
    print(f"✓ Processing completed! Results saved to: {output_file_path}")
    total_time = time.time() - start_time
    print(f"✓ Total time: {total_time:.2f}s")
    print(f"✓ Average time per sample: {total_time/total_samples:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["orig", "neg"], required=True)
    parser.add_argument("--wav_folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
