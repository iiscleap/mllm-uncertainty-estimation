import os
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import csv
import re
import numpy as np
import collections
import argparse

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

parser = argparse.ArgumentParser(description='Phi-4 top-k sampling baseline on VSR')
parser.add_argument('--model_path', type=str, required=True, help='Path or HF ID for Phi-4-multimodal-instruct')
parser.add_argument('--qns_filepath', type=str, required=True, help='Path to VSR questions CSV')
parser.add_argument('--img_folder', type=str, required=True, help='Path to VSR orig_images folder')
parser.add_argument('--output_path', type=str, default='vlm_results/phi4_topk_sampling_vsr.csv', help='Output CSV path')
args = parser.parse_args()

model_path = args.model_path

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print(processor.tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto'
).cuda()

generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'


qns_filepath = args.qns_filepath
output_path = args.output_path
img_folder = args.img_folder

K = 4
N = 5

with open(output_path, mode="w", newline="", encoding="utf-8") as out_file:
    fieldnames = ["idx", "guess", "probability"]
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(111, 211):
        idx = f"val_Spatial_Reasoning_{i}"
        qns = get_qns_for_idx(qns_filepath, idx)
        qns = f"{qns}\nChoices:\nA. True\nB. False"

        img_path = f"{img_folder}/{idx}.jpg"

        txt = (
            f"Provide your {K} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. "
            "Each guess must be either 'A' or 'B' only. Give ONLY the guesses and probabilities, no other words or explanation."
            f"You MUST follow the template given below to generate {K} guesses and probabilities\n\n"
            "G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
            "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n"
            "...\n"
            f"G{K}: <{K}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
            f"P{K}: <the probability between 0.0 and 1.0 that G{K} is correct, without any extra commentary whatsoever; just the probability!>\n\n"
            f"The question is: {qns}"
        )

        prompt = f'{user_prompt}<|image_1|>{txt}{prompt_suffix}{assistant_prompt}'
        #print(f'>>> Prompt\n{prompt}')
        image = Image.open(img_path)
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
        all_guesses = collections.defaultdict(list)

        for _ in range(N):
            generate_ids = model.generate(
                                **inputs,
                                max_new_tokens=100,
                                do_sample=True,
                                temperature=0.7
                            )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f'>>> Response\n{response}')

            matches = re.findall(r"G\d+:\s*(.*?)\s+P\d+:\s*([0-1](?:\.\d+)?)", response)
            for guess, prob in matches:
                guess = guess.strip()
                prob = float(prob.strip())
                all_guesses[guess].append(prob)

        if all_guesses:
            avg_conf = {guess: np.mean(probs) for guess, probs in all_guesses.items()}
            best_guess, best_prob = max(avg_conf.items(), key=lambda x: x[1])
        else:
            best_guess, best_prob = "N/A", 0.0

        writer.writerow({
            "idx": idx,
            "guess": best_guess,
            "probability": round(best_prob, 3)
        })