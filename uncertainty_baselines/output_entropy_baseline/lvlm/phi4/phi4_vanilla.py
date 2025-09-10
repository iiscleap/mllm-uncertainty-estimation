import os
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import csv

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

model_path = 'path/to/Phi-4-multimodal-instruct'
kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

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


qns_filepath = "path/to/vsr_data/negated_questions.csv"

for i in range(111,211):
    idx = f"val_Spatial_Reasoning_{i}"
    qns = get_qns_for_idx(qns_filepath, idx)
    img_path = f"path/to/vsr_data/orig_images/{idx}.jpg"
    txt = f"{qns}\nChoices:\nA. True\nB. False\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"

    prompt = f'{user_prompt}<|image_1|>{txt}{prompt_suffix}{assistant_prompt}'
    print(f'>>> Prompt\n{prompt}')
    image = Image.open(img_path)
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f'>>> Response\n{response}')

    with open("phi4_results/negated_phi4_vanilla_vsr.txt", "a") as resfile:
        resfile.write(f"{idx} {response}\n")