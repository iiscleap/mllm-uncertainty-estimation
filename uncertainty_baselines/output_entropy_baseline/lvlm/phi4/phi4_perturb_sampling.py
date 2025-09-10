import os
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import csv

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

with open("path/to/negated_vsr_perturbed/negated_vsr_perturbations_data.csv", mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        idx = row["idx"]
        image_idx = row["image_idx"]
        qns = row["question"]
        img_path = f"path/to/negated_vsr_perturbed/negated_vsr_perturbed_images/{image_idx}.jpg"
        txt = f"{qns}\nChoices:\nA. True\nB. False\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"

        prompt = f'{user_prompt}<|image_1|>{txt}{prompt_suffix}{assistant_prompt}'
        image = Image.open(img_path)
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
        
        k = 10
        for i in range(k):
            new_idx = f"{idx}_sample{i}"
            generate_ids = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.95,
                            num_beams=1,
                            max_new_tokens=1,
                            generation_config=generation_config,
                        )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            with open("phi4_results/negated_phi4_perturb_sampling_vsr.txt", "a") as resfile:
                resfile.write(f"{new_idx} {response}\n")