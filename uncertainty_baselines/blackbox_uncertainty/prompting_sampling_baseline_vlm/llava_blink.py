from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import csv
import os
import re
import collections
import numpy as np

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

# Load model and processor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

K = 4
N = 5

qns_filepath = "path/to/blink_data/question.csv"
img_folder = "path/to/blink_data/orig_images"
output_csv = "vlm_results/llava_topk_sampling_blink.csv"

# Write header
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "guess", "probability"])

for i in range(1, 144):
    idx = f"val_Spatial_Relation_{i}"
    qns = get_qns_for_idx(qns_filepath, idx)
    qns = f"{qns}\nChoices:\nA. Yes\nB. No"

    img_path = f"{img_folder}/{idx}.jpg"
    image = Image.open(img_path)

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

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
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

    best_prob = round(best_prob, 3)

    with open(output_csv, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([idx, best_guess, best_prob])

