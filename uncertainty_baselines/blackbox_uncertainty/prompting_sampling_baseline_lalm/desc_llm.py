import csv
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import collections
import re
import numpy as np
import argparse

device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

parser = argparse.ArgumentParser(description='desc_llm top-k sampling baseline for LALM (uses audio descriptions)')
parser.add_argument('--input_file', type=str, required=True, help='Path to subset CSV (e.g., .../order_task/order_subset_100samples.csv)')
parser.add_argument('--desc_folder', type=str, required=True, help='Folder with per-idx description text files')
parser.add_argument('--output_file', type=str, default='lalm_results/desc_llm_topk_sampling_order.csv', help='Output CSV path')
args = parser.parse_args()

output_file = args.output_file
input_file = args.input_file
desc_folder = args.desc_folder

K = 4
N = 5

with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "guess", "probability"])

    with open(input_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']

            with open(f"{desc_folder}/{idx}.txt", "r") as file:
                desc = file.read()

            question = row['question']
            optionA = row['optionA']
            optionB = row['optionB']
            optionC = row['optionC']
            optionD = row['optionD']
            
            qns = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"

            prompt = (
                f"Provide your {K} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. "
                "Each guess must be either 'A','B', 'C' or 'D' ONLY. Give ONLY the guesses and probabilities, no other words or explanation."
                f"You MUST follow the template given below to generate {K} guesses and probabilities\n\n"
                "G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
                "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n"
                "...\n"
                f"G{K}: <{K}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
                f"P{K}: <the probability between 0.0 and 1.0 that G{K} is correct, without any extra commentary whatsoever; just the probability!>\n\n"
                f"The description of the audio clip is given below:\n{desc}\n\n"
                f"The question is: {qns}"
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

            all_guesses = collections.defaultdict(list)

            for _ in range(N):
                generated_ids = model.generate(
                                    model_inputs.input_ids,
                                    max_new_tokens=100,
                                    do_sample=True,
                                    temperature=0.7
                                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                print(f"{idx}- Sample:\n{response}")
                group_answers = dict(re.findall(r"G(\d+):\s*(\w)", response))
                probabilities = dict(re.findall(r"P(\d+):\s*([0-1](?:\.\d+)?)", response))

                # Match by index and aggregate
                for k in group_answers:
                    if k in probabilities:
                        guess = group_answers[k].strip()
                        prob = float(probabilities[k].strip())
                        all_guesses[guess].append(prob)

            # Your regex logic
            if all_guesses:
                avg_conf = {guess: np.mean(probs) for guess, probs in all_guesses.items()}
                best_guess, best_prob = max(avg_conf.items(), key=lambda x: x[1])
            else:
                best_guess, best_prob = "N/A", 0.0
            
            print(f"{idx}, {best_guess}, {best_prob}")
            writer.writerow([idx, best_guess, round(best_prob, 3)])
