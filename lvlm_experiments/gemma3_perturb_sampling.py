from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the questions CSV file')
parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--dataset', type=str, choices=['blink', 'vsr'], required=True, help='Dataset name')
parser.add_argument('--type', type=str, choices=['orig', 'neg'], required=True, help='Type of question: orig or neg')
args = parser.parse_args()

model_id = "google/gemma-3-12b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)

if args.dataset == 'blink':
    indices = [f"val_Spatial_Relation_{i}" for i in range(1, 144)]
    choices = "A. Yes\nB. No"
elif args.dataset == 'vsr':
    indices = [f"val_Spatial_Reasoning_{i}" for i in range(111, 211)]
    choices = "A. True\nB. False"

output_prefix = "negated_" if args.type == "neg" else ""
output_dir = "perturb_sampling_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{output_prefix}gemma3_perturb_sampling_{args.dataset}.txt")

with open(args.input_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        idx = row["idx"]
        image_idx = row["image_idx"]
        
        qns = row["question"]
        img_path = os.path.join(args.input_image_folder, f"{image_idx}.jpg")
        prompt = (
            f"{qns}\nChoices:\n{choices}\n"
            "Return only the option (A or B), and nothing else.\n"
            "MAKE SURE your output is A or B"
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

        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        k = 10
        for i in range(k):
            new_idx = f"{idx}_sample{i}"
            with torch.inference_mode():
                generation = model.generate(
                                **inputs,
                                do_sample=True,
                                temperature=1.0,
                                top_p=0.95,
                                num_beams=1,
                                max_new_tokens=1,
                            )
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            print(decoded)

            with open(output_file, "a") as res:
                res.write(f"{new_idx} {decoded}\n")

            
