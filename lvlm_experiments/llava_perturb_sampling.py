import argparse
import os
import csv
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the questions CSV file')
parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--dataset', type=str, choices=['blink', 'vsr'], required=True, help='Dataset name')
parser.add_argument('--type', type=str, choices=['orig', 'neg'], required=True, help='Type of question: orig or neg')
args = parser.parse_args()

if args.dataset == 'blink':
    choices = "A. Yes\nB. No"
elif args.dataset == 'vsr':
    choices = "A. True\nB. False"

output_prefix = "negated_" if args.type == "neg" else ""
output_dir = "perturb_sampling_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{output_prefix}llava_perturb_sampling_{args.dataset}.txt")

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto"
)

with open(args.input_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        idx = row["idx"]
        image_idx = row["image_idx"]
        qns = row["question"]

        img_path = os.path.join(args.input_image_folder, f"{image_idx}.jpg")
        txt = f"{qns}\nChoices:\n{choices}\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
        image = Image.open(img_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": txt},
                    {"type": "image"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

        k = 10 
        for i in range(k):
            new_idx = f"{idx}_sample{i}"
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                num_beams=1,
                max_new_tokens=1,
            )

            res = processor.decode(output[0], skip_special_tokens=True).strip()
            res = res[-1]

            with open(output_file, "a") as resfile:
                resfile.write(f"{new_idx} {res}\n")
