import os
import csv
import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the perturbations CSV file')
parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the folder containing perturbed images')
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
output_file = os.path.join(output_dir, f"{output_prefix}qwenvl_perturb_sampling_{args.dataset}.txt")

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

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
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        k = 10
        for i in range(k):
            new_idx = f"{idx}_sample{i}"
            generated_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                num_beams=1,
                max_new_tokens=1,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            print(f"{new_idx}:{output_text}")
            with open(output_file, "a") as res:
                res.write(f"{new_idx} {output_text}\n")
