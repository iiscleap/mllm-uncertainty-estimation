import os
import argparse
import csv
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the questions CSV file')
parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--dataset', type=str, choices=['blink', 'vsr'], required=True, help='Dataset name')
parser.add_argument('--type', type=str, choices=['orig', 'neg'], required=True, help='Type of question: orig or neg')
parser.add_argument('--model_path', type=str, required=True, help='Path to Phi4 model')
args = parser.parse_args()

if args.dataset == 'blink':
    choices = "A. Yes\nB. No"
elif args.dataset == 'vsr':
    choices = "A. True\nB. False"

output_prefix = "negated_" if args.type == "neg" else ""
output_dir = "perturb_sampling_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{output_prefix}phi4_perturb_sampling_{args.dataset}.txt")

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype='auto'
).cuda()

generation_config = GenerationConfig.from_pretrained(args.model_path, 'generation_config.json')

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

with open(args.input_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        idx = row["idx"]
        image_idx = row["image_idx"]
        qns = row["question"]

        img_path = os.path.join(args.input_image_folder, f"{image_idx}.jpg")
        txt = f"{qns}\nChoices:\n{choices}\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
        prompt = f'{user_prompt}<|image_1|>{txt}{prompt_suffix}{assistant_prompt}'

        image = Image.open(img_path)
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

        for i in range(10):
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
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            with open(output_file, "a") as resfile:
                resfile.write(f"{new_idx} {response}\n")
