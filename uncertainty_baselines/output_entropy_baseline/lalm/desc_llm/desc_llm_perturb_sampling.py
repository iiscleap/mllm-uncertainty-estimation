import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Desc LLM perturb sampling predictions for LALM')
    parser.add_argument('--task', type=str, required=True, choices=['count', 'duration', 'order'])
    parser.add_argument('--exp_type', type=str, required=True, choices=['orig', 'neg'])
    args = parser.parse_args()
    
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-medium"  # Placeholder - replace with actual desc_llm model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    task = args.task
    exp_type = args.exp_type
    
    # Create output directory
    output_dir = f"{exp_type}/perturb_sampling"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output filename
    if exp_type == "orig":
        output_file = f"{output_dir}/desc_llm_{task}_perturb_sampling.txt"
    else:
        output_file = f"{output_dir}/negated_desc_llm_{task}_perturb_sampling.txt"
    
    # Define perturbation files based on task
    aug_dataset_path = "/home/debarpanb/VLM_project/apoorva_works/Speech_LLM/temporal_dataset_augmented_100samples"
    
    if task == "count":
        perturb_files = ["reworded_add_delete_audio_perturbations.csv", "reworded_silence_perturbations.csv", "reworded_volume_perturbations.csv"]
    elif task == "duration":
        perturb_files = ["reworded_silence_perturbations.csv", "reworded_volume_perturbations.csv"]
    elif task == "order":
        perturb_files = ["reworded_silence_perturbations.csv", "reworded_volume_perturbations.csv"]
    
    for csv_file in perturb_files:
        perturb_csv_path = f"{aug_dataset_path}/{task}_task/idx_{csv_file}"
        
        with open(perturb_csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                audio_path = row['new_audio_path']
                idx = row["idx"]

                # Construct full audio path
                full_audio_path = f"{aug_dataset_path}/{task}_task/{audio_path}"

                # Load and process audio (placeholder - desc_llm specific audio processing needed)
                waveform, sample_rate = torchaudio.load(full_audio_path)

                question = row['question']
                
                # For negated experiments, modify the question
                if exp_type == "neg":
                    question = f"Is it NOT the case that: {question}"
                
                optionA = row['optionA']
                optionB = row['optionB']
                optionC = row['optionC']
                optionD = row['optionD']
                
                # Create prompt with audio description placeholder
                audio_description = "[AUDIO_DESCRIPTION]"  # This would be replaced with actual audio description
                prompt = f"Audio: {audio_description}\n{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

                k = 20
                print(f"Processing ID: {idx}")

                # Tokenize
                inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = inputs.to(model.device)
                
                for i in range(k):
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            do_sample=True,
                            temperature=1.0,
                            top_k=4,
                            top_p=0.95,
                            num_beams=1,
                            max_new_tokens=1,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                    
                    with open(output_file, "a") as res:
                        new_idx = f"{idx}_sample{i}"
                        res.write(f"{new_idx} {response}\n")
                
                print("=" * 20)

if __name__ == "__main__":
    main()
