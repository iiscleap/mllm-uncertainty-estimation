import os
import csv
import torch
import argparse
import sys
import time
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def get_prompt(desc, question, optionA, optionB, optionC, optionD):
    return (
        f"The description of the audio clip is given below:\n{desc}\n"
        f"Based on the information above, answer the following:\n"
        f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\n"
        f"Return only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"
    )

def get_base_idx(idx):
    return idx.split("_rephrased")[0]

def main(args):

    print("Loading Qwen2-7B-Instruct model...")
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    output_folder = "perturb_sampling_output"
    os.makedirs(output_folder, exist_ok=True)

    if args.type == "neg":
        output_file_path = os.path.join(output_folder, f"negated_desc_llm_{args.task}_perturb_sampling.txt")
    else:
        output_file_path = os.path.join(output_folder, f"desc_llm_{args.task}_perturb_sampling.txt")

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
            idx = row["idx"]
            base_idx = get_base_idx(idx)
            desc_file_path = os.path.join(args.desc_folder, f"{base_idx}.txt")

            with open(desc_file_path, "r") as desc_file:
                desc = desc_file.read()

            prompt = get_prompt(desc, row['question'], row['optionA'], row['optionB'], row['optionC'], row['optionD'])

            messages = [
                {"role": "system", "content": "You are a helpful assistant that must read the description and answer the question. Your response must contain only the option and nothing else"},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            k = 10
            for j in range(k):
                generated_ids = model.generate(
                                        model_inputs.input_ids,
                                        do_sample=True,
                                        temperature=1.0,
                                        top_k=4,
                                        top_p=0.95,
                                        num_beams=1,
                                        max_new_tokens=1,
                                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                with open(output_file_path, "a") as res_file:
                    new_idx = f"{idx}_sample{j}"
                    res_file.write(f"{new_idx} {response}\n")

            # Update progress bar
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total_samples - i - 1) if i > 0 else 0
            suffix = f"({i+1}/{total_samples}) | ETA: {eta:.1f}s | Sample: {idx}"
            print_progress_bar(i + 1, total_samples, prefix='Progress:', suffix=suffix, length=40)

    print()
    print("=" * 70)
    print(f"Processing completed! Results saved to: {output_file_path}")
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time/total_samples:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--type", type=str, choices=["orig", "neg"], required=True)
    parser.add_argument("--desc_folder", type=str, required=True, help="Folder containing audio description .txt files")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with questions and options")

    args = parser.parse_args()
    main(args)
