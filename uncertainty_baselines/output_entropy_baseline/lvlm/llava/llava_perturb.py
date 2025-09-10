from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import csv

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map = "auto") 

with open("./negated_vsr_perturbed/negated_vsr_perturbations_data.csv", mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row["idx"]
            image_idx = row["image_idx"]
            qns = row["question"]
            img_path = f"./negated_vsr_perturbed/negated_vsr_perturbed_images/{image_idx}.jpg"
            txt = f"{qns}\nChoices:\nA. True\nB. False\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
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
            output = model.generate(**inputs, max_new_tokens=1)

            res = processor.decode(output[0], skip_special_tokens=True)
            res = res[-1]

            print(res)
            with open("llava_results/vsr_llava_perturb_vsr.txt", "a") as resfile:
                resfile.write(f"{idx} {res}\n")
