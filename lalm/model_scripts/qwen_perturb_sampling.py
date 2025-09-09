import os
import csv
import torch
import argparse
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, logging
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

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
        print()

def get_prompt(question, optionA, optionB, optionC, optionD):
    return f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

def get_base_idx(idx):
    return idx.split('_rephrased')[0]

def main(args):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

    output_folder = "perturb_sampling_output"
    os.makedirs(output_folder, exist_ok=True)

    if args.type == "neg":
        output_file_path = os.path.join(output_folder, f"negated_qwen_{args.task}_perturb_sampling.txt")
    else:
        output_file_path = os.path.join(output_folder, f"qwen_{args.task}_perturb_sampling.txt")

    # Count total samples for progress tracking
    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file_tmp:
        total_samples = sum(1 for _ in csv.DictReader(file_tmp))

    print(f"Processing {total_samples} samples from {args.csv_path}")
    print(f"Output will be saved to: {output_file_path}")
    print("=" * 70)
    start_time = time.time()

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row_idx, row in enumerate(reader):
            idx = row['idx']
            base_idx = get_base_idx(idx)
            audio_path = os.path.join(args.wav_folder, f"{base_idx}.wav")

            question = row['question']
            optionA = row['optionA']
            optionB = row['optionB']
            optionC = row['optionC']
            optionD = row['optionD']

            prompt = get_prompt(question, optionA, optionB, optionC, optionD)

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': [{'type': 'audio', 'audio_url': audio_path}, {"type": "text", "text": prompt}]}
            ]

            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            waveform, _ = librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)
                            audios.append(waveform)

            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to("cuda")

            k = 10        
            for i in range(k):
                generate_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_k=4,
                    top_p=0.95,
                    num_beams=1,
                    max_new_tokens=1,
                )
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                with open(output_file_path, "a") as f:
                    new_idx = f"{idx}_sample{i}"
                    f.write(f"{new_idx} {response}\n")

            # Update progress bar
            elapsed = time.time() - start_time
            eta = (elapsed / (row_idx + 1)) * (total_samples - row_idx - 1) if row_idx > 0 else 0
            suffix = f"({row_idx+1}/{total_samples}) | ETA: {eta:.1f}s | Sample: {idx}"
            print_progress_bar(row_idx + 1, total_samples, prefix='Progress:', suffix=suffix, length=40)

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
    parser.add_argument("--wav_folder", type=str, required=True, help="Folder containing .wav files named as {base_idx}.wav")

    args = parser.parse_args()
    main(args)
