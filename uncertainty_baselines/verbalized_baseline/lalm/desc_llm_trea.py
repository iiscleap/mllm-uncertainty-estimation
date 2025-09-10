import csv
import re
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run desc_llm model on TREA dataset tasks')
    parser.add_argument('--task', type=str, choices=['count', 'duration', 'order'], 
                       required=True, help='Task to run: count, duration, or order')
    parser.add_argument('--output_dir', type=str, default='lalm_results', 
                       help='Output directory for results (relative to script location)')
    parser.add_argument('--dataset_path', type=str, default='/home/debarpanb/VLM_project/TREA_dataset',
                       help='Path to TREA dataset')
    
    args = parser.parse_args()
    
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    # Set up file paths based on task
    task = args.task
    input_file = f"{args.dataset_path}/{task}/{task}.csv"
    desc_folder = f"{args.dataset_path}/{task}/audio_desc"
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    output_file = os.path.join(output_dir, f"desc_llm_guess_prob_trea_{task}.csv")
    
    print(f"Processing task: {task}")
    print(f"Input file: {input_file}")
    print(f"Description folder: {desc_folder}")
    print(f"Output file: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "guess", "probability"])

        with open(input_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                idx = row['id']

                # Read the audio description
                desc_file = os.path.join(desc_folder, f"{idx}.txt")
                with open(desc_file, "r") as file:
                    desc = file.read()

                question = row['question']
                optionA = row['optionA']
                optionB = row['optionB']
                optionC = row['optionC']
                optionD = row['optionD']
                
                qns = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
                
                prompt = (
                    "You must provide both the answer choice (A, B, C, or D) and the probability that your answer is correct "
                    "(a number between 0.0 and 1.0). Your response MUST include BOTH the guess and the probability, "
                    "in the exact format shown below. Do NOT include any explanations, comments, or extra words.\n\n"
                    "Format:\n"
                    "Guess: <A/B/C/D>\n"
                    "Probability: <number between 0.0 and 1.0>\n\n"
                    f"The description of the audio clip is given below:\n{desc}\n\n"
                    f"The question is:\n{qns}"
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Only output the guess and probability in the specified format."},
                    {"role": "user", "content": prompt},
                ]

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=30,
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                # Extract guess and probability using regex
                guess_match = re.search(r"Guess:\s*([A-D])", response)
                prob_match = re.search(r"Probability:\s*([0-9]*\.?[0-9]+)", response)

                guess = guess_match.group(1) if guess_match else ""
                prob = prob_match.group(1) if prob_match else ""

                writer.writerow([idx, guess, prob])
                print(f"{idx},{guess},{prob}")
                print("=" * 20)

if __name__ == "__main__":
    main()
