import os
import csv
import torch
import sys
import warnings
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Suppress warnings
warnings.filterwarnings("ignore")

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

# Get dataset and experiment type from command line arguments
if len(sys.argv) != 3:
    print("Usage: python phi4_combined_modified.py <dataset> <exp_type>")
    print("dataset: blink, vsr")
    print("exp_type: image_only, text_only, text_image")
    sys.exit(1)

dataset = sys.argv[1]
exp = sys.argv[2]

print(f"Running phi4 on dataset: {dataset}, experiment: {exp}")

# Try to use local model path first, fallback to HuggingFace if needed
model_path = "path/to/Phi-4-multimodal-instruct"

try:
    print("Loading Phi4 model...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto'
    ).cuda()
    
    # Try to load generation config
    try:
        generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
    except:
        generation_config = None
        
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model is available locally or you have internet connectivity.")
    sys.exit(1)

# Set parameters based on dataset and experiment type
if exp == "image_only" and dataset == "blink":
    csv_filepath = "blink_image_perturbations_only.csv"
    img_column = "idx"
    img_dir = "blink_image_perturbations_only"
    option_type = "A. Yes\nB. No"

elif exp == "image_only" and dataset == "vsr":
    csv_filepath = "vsr_image_perturbations_only.csv"
    img_column = "idx"
    img_dir = "vsr_image_perturbations_only"
    option_type = "A. True\nB. False"

elif exp == "text_only" and dataset == "blink":
    csv_filepath = "blink_text_perturbations_only.csv"
    img_column = "orig_idx"
    img_dir = "blink_data/orig_images"
    option_type = "A. Yes\nB. No"

elif exp == "text_only" and dataset == "vsr":
    csv_filepath = "vsr_text_perturbations_only.csv"
    img_column = "orig_idx"
    img_dir = "vsr_data/orig_images"
    option_type = "A. True\nB. False"

elif exp == "text_image" and dataset == "blink":
    csv_filepath = "blink_perturbations_data.csv"
    img_column = "image_idx"
    img_dir = "blink_perturbed_images"
    option_type = "A. Yes\nB. No"

elif exp == "text_image" and dataset == "vsr":
    csv_filepath = "vsr_perturbations_data.csv"
    img_column = "image_idx"
    img_dir = "vsr_perturbed_images"
    option_type = "A. True\nB. False"

else:
    print(f"Invalid combination: dataset={dataset}, exp_type={exp}")
    sys.exit(1)

# Check if CSV file exists
if not os.path.exists(csv_filepath):
    print(f"CSV file not found: {csv_filepath}")
    sys.exit(1)

# Clear the output file
output_file = f"phi4_results/phi4_{exp}_{dataset}.txt"
os.makedirs("phi4_results", exist_ok=True)
with open(output_file, "w") as f:
    pass

print(f"Processing file: {csv_filepath}")
processed_count = 0
error_count = 0

# Define prompt templates based on reference file
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

with open(csv_filepath, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        idx = row["idx"]
        image_idx = row[img_column]
        qns = row["question"]
        img_path = f"{img_dir}/{image_idx}.jpg"
        
        # Check if image file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} not found, skipping...")
            error_count += 1
            continue
        
        try:
            # Use the reference file's prompt format
            txt = f"{qns}\nChoices:\n{option_type}\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
            prompt = f'{user_prompt}<|image_1|>{txt}{prompt_suffix}{assistant_prompt}'

            image = Image.open(img_path)
            inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

            generation_args = {
                "do_sample": False,  # Use greedy decoding for consistency
                "max_new_tokens": 1,
            }
            
            if generation_config:
                generation_args["generation_config"] = generation_config

            generate_ids = model.generate(
                **inputs,
                **generation_args
            )

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
            # Extract just the first character if it's A or B
            res = response.strip()
            if res and res[0] in ['A', 'B']:
                res = res[0]
            else:
                res = res[:1] if res else ''

            with open(output_file, "a") as resfile:
                resfile.write(f"{idx} {res}\n")
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} samples...")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            error_count += 1
            # Write empty response on error
            with open(output_file, "a") as resfile:
                resfile.write(f"{idx} \n")
            continue

print(f"Completed {dataset} - {exp}")
print(f"Successfully processed: {processed_count} samples")
print(f"Errors: {error_count} samples")
print(f"Results saved to: {output_file}")
