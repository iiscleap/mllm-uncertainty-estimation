# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import csv

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

model_id = "google/gemma-3-12b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)
qns_filepath = "vsr_data/negated_questions.csv"

for i in range(111,211):
    idx = f"val_Spatial_Reasoning_{i}"
    qns = get_qns_for_idx(qns_filepath, idx)
    img_path = f"vsr_data/orig_images/{idx}.jpg"
    prompt = f"{qns}\nChoices:\nA. True\nB. False\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"

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

    k = 20
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

        with open("gemma3_results/negated_gemma3_sampling_vsr.txt", "a") as res:
            res.write(f"{new_idx} {decoded}\n")

    
