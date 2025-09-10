from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
import csv
import sys
import os
import warnings
from transformers import logging

# Suppress warnings and transformers logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

# Get dataset and experiment type from command line arguments
if len(sys.argv) != 3:
    print("Usage: python gemma3_combined_modified.py <dataset> <exp_type>")
    print("dataset: blink, vsr")
    print("exp_type: image_only, text_only, text_image")
    sys.exit(1)

dataset = sys.argv[1]
exp = sys.argv[2]

print(f"Running gemma3 on dataset: {dataset}, experiment: {exp}")

model_id = "google/gemma-3-12b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)

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
output_file = f"gemma3_results/gemma3_{exp}_{dataset}.txt"
os.makedirs("gemma3_results", exist_ok=True)
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
            prompt = (
                f"{qns}\nChoices:\n{option_type}\n"
                "Return only the option (A or B), and nothing else.\n"
                "MAKE SURE your output is A or B"
            )

            # Use the correct message format as in the reference file
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
            
            # Process the inputs correctly
            inputs = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    do_sample=False,  # Use greedy decoding for consistency
                    max_new_tokens=1,
                )
                generation = generation[0][input_len:]

            response = processor.decode(generation, skip_special_tokens=True)
            
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
