from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import csv
import sys
import os

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

# Get dataset and experiment type from command line arguments
if len(sys.argv) != 3:
    print("Usage: python llava_combined_modified.py <dataset> <exp_type>")
    print("dataset: blink, vsr")
    print("exp_type: image_only, text_only, text_image")
    sys.exit(1)

dataset = sys.argv[1]
exp = sys.argv[2]

print(f"Running llava on dataset: {dataset}, experiment: {exp}")

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto") 

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
output_file = f"llava_results/llava_{exp}_{dataset}.txt"
os.makedirs("llava_results", exist_ok=True)
with open(output_file, "w") as f:
    pass

print(f"Processing file: {csv_filepath}")
processed_count = 0
error_count = 0

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
            txt = f"{qns}\nChoices:\n{option_type}\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"
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
