from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login
import csv
import os
import base64
import re
import collections
import numpy as np

login(os.environ.get("HF_TOKEN"))


def file_to_data_url(file_path: str):
    """
    Convert a local image file to a data URL.
    """    
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    
    return f"data:{mime_type};base64,{encoded_string}"


def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']


model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=100, temperature=0.7)

qns_filepath = "blink_data/question.csv"
output_path = "pixtral_topk_sampling_blink.csv"
img_folder = "blink_data/orig_images"

K = 4
N = 5

llm = LLM( model=model_name,
          gpu_memory_utilization=0.95,
          max_model_len=4096,
          tokenizer_mode="mistral",
          load_format="mistral",
          config_format="mistral"
         )

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

        image_source = file_to_data_url(img_path)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_source}}]
            },
        ]

        all_guesses = collections.defaultdict(list)

        for _ in range(N):
            outputs = llm.chat(messages, sampling_params=sampling_params)
            final_res = outputs[0].outputs[0].text
            print(f"{idx} [Sample]:\n{final_res}\n")

            matches = re.findall(r"G\d+:\s*(.*?)\s+P\d+:\s*([0-1](?:\.\d+)?)", final_res)
            for guess, prob in matches:
                guess = guess.strip()
                prob = float(prob.strip())
                all_guesses[guess].append(prob)

        if all_guesses:
            avg_conf = {guess: np.mean(probs) for guess, probs in all_guesses.items()}
            best_guess, best_prob = max(avg_conf.items(), key=lambda x: x[1])
        else:
            best_guess, best_prob = "N/A", 0.0
        
        print(f"{idx}, {guess}, {best_prob}")
        writer.writerow({
            "idx": idx,
            "guess": best_guess,
            "probability": round(best_prob, 3)
        })
