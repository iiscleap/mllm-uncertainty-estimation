from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import csv
import torch 
import os
import re
import collections
import numpy as np
import argparse

# Load processor and model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

parser = argparse.ArgumentParser(description='Qwen top-k sampling baseline for LALM (audio)')
parser.add_argument('--input_file', type=str, required=True, help='Path to subset CSV (e.g., .../order_task/order_subset_100samples.csv)')
parser.add_argument('--audio_dir', type=str, required=True, help='Path to audio folder with wavs named by idx (e.g., .../ESC_50_reasoning_order_dataset/audios)')
parser.add_argument('--output_file', type=str, default='lalm_results/qwen_topk_sampling_order.csv', help='Output CSV path')
args = parser.parse_args()

output_file = args.output_file
input_file = args.input_file
audio_dir = args.audio_dir

K = 4
N = 5

# Open output CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["idx", "guess", "probability"])

    # Read input file
    with open(input_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']
            audio_path = f"{audio_dir}/{idx}.wav"

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
                f"The question is: {qns}"
            )

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': [
                    {'type': 'audio', 'audio_url': audio_path},
                    {'type': 'text', 'text': prompt}
                ]}
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            waveform, _ = librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)
                            audios.append(waveform)

            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            all_guesses = collections.defaultdict(list)

            for _ in range(N):
                generate_ids = model.generate(
                                    **inputs,
                                    max_new_tokens=100,
                                    do_sample=True,
                                    temperature=0.7
                                )
                generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print(f"{idx} [Sample]:\n{response}\n")

                group_answers = dict(re.findall(r"G(\d+):\s*(\w)", response))
                probabilities = dict(re.findall(r"P(\d+):\s*([0-1](?:\.\d+)?)", response))

                # Match by index and aggregate
                for k in group_answers:
                    if k in probabilities:
                        guess = group_answers[k].strip()
                        prob = float(probabilities[k].strip())
                        all_guesses[guess].append(prob)

            if all_guesses:
                avg_conf = {guess: np.mean(probs) for guess, probs in all_guesses.items()}
                best_guess, best_prob = max(avg_conf.items(), key=lambda x: x[1])
            else:
                best_guess, best_prob = "N/A", 0.0
            
            print(f"{idx}, {guess}, {best_prob}")
            writer.writerow([idx, best_guess, round(best_prob, 3)])

