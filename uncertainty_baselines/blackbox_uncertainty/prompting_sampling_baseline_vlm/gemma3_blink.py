import re
import os
import csv
import torch
import numpy as np
import collections
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

K = 4 
N = 5

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']
    return None

model_id = "google/gemma-3-12b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)

qns_filepath = "path/to/blink_data/question.csv"
output_path = "vlm_results/gemma3_topk_sampling_blink.csv"
img_folder = "path/to/blink_data/orig_images"

with open(output_path, mode="w", newline="", encoding="utf-8") as out_file:
    fieldnames = ["idx", "guess", "probability"]
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(1, 144):
        idx = f"val_Spatial_Relation_{i}"
        qns = get_qns_for_idx(qns_filepath, idx)
        qns = f"{qns}\nChoices:\nA. Yes\nB. No"

        img_path = f"{img_folder}/{idx}.jpg"

        prompt = (
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

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        all_guesses = collections.defaultdict(list)

        for _ in range(N):
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                )
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            print(f"{idx} [Sample]:\n{decoded}\n")

            matches = re.findall(r"G\d+:\s*(.*?)\s+P\d+:\s*([0-1](?:\.\d+)?)", decoded)
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
