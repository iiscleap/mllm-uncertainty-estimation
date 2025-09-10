from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import csv
import torch 
import os
import re
import sys

# Load processor and model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

# Get task and experiment type from command line arguments
if len(sys.argv) != 3:
    print("Usage: python qwen_audio_modified.py <task> <exp_type>")
    print("task: count, order, duration")
    print("exp_type: audio_only, text_only, text_audio")
    sys.exit(1)

task = sys.argv[1]
exp = sys.argv[2]

print(f"Running qwen_audio on task: {task}, experiment: {exp}")

if exp == "audio_only":
    csv_filepath = f"{task}_audio_perturbations_only.csv"
    audio_path_column = "new_audio_path"
    audio_base_dir = ""

elif exp == "text_only":
    csv_filepath = f"{task}_text_perturbations_only.csv"
    audio_path_column = "audio_path"
    if task != "duration":
        audio_base_dir = f"/path/to/desc_dir/{task}/perturbed_audio_desc"
    else:
        audio_base_dir = f"/path/to/desc_dir/{task}/perturbed_audio_desc"

elif exp == "text_audio":
    csv_filepath = f"/path/to/desc_dir/{task}/{task}_perturbed.csv"
    audio_path_column = "new_audio_path"
    audio_base_dir = f"/path/to/TREA_dataset/{task}"

else:
    print(f"Invalid experiment type: {exp}")
    sys.exit(1)

# Clear the output file
output_file = f"qwen_results/qwen_{exp}_{task}.txt"
with open(output_file, "w") as f:
    pass

with open(csv_filepath, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        idx = row["idx"]
        
        # Handle audio file path based on experiment type
        if exp == "audio_only":
            audio_path = row[audio_path_column]
        elif exp == "text_only":
            audio_id = row["orig_idx"]
            audio_path = f"{audio_base_dir}/audios/{audio_id}.wav"
        elif exp == "text_audio":
            # TREA dataset: use the audio file path directly
            audio_file = row[audio_path_column]
            audio_path = f"{audio_base_dir}/{audio_file}"
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file {audio_path} not found, skipping...")
            continue
            
        # Load audio
        try:
            audio_input, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
        except Exception as e:
            print(f"Warning: Could not load audio {audio_path}: {e}, skipping...")
            continue

        question = row['question']
        optionA = row['optionA']
        optionB = row['optionB'] 
        optionC = row['optionC']
        optionD = row['optionD']
        
        prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

        conversation = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that must listen to the audio and answer the question. Your response must contain only the option and nothing else."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "audio", "audio": audio_input},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text_input, audios=[audio_input], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 1)
            
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        with open(output_file, "a") as res:
            res.write(f"{idx} {response}\n")
        
        print(f"Processed {idx}")

print(f"Completed {task} - {exp}")
