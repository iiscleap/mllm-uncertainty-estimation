from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import csv
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Qwen sampling predictions for LALM')
    parser.add_argument('--task', type=str, required=True, choices=['count', 'duration', 'order'])
    parser.add_argument('--exp_type', type=str, required=True, choices=['orig', 'neg'])
    args = parser.parse_args()
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
    
    task = args.task
    exp_type = args.exp_type
    
    # Create output directory
    output_dir = f"{exp_type}/sampling"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output filename
    if exp_type == "orig":
        output_file = f"{output_dir}/qwen_{task}_sampling.txt"
    else:
        output_file = f"{output_dir}/negated_qwen_{task}_sampling.txt"
    
    # Dataset path
    csv_file = f"../subset/{task}_subset_100samples.csv"
    
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            audio_path = row['audio_path']
            idx = row['id']

            # Construct full audio path
            if task != "duration":
                full_audio_path = f"/home/debarpanb/VLM_project/apoorva_works/Speech_LLM/temporal_dataset/ESC_50_reasoning_{task}_dataset/{audio_path}"
            else:
                full_audio_path = f"/home/debarpanb/VLM_project/apoorva_works/Speech_LLM/temporal_dataset/ESC_50_reasoning_{task}_dataset_metadata/{audio_path}"

            question = row['question']
            
            # For negated experiments, modify the question
            if exp_type == "neg":
                question = f"Is it NOT the case that: {question}"
            
            optionA = row['optionA']
            optionB = row['optionB']
            optionC = row['optionC']
            optionD = row['optionD']
            prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": full_audio_path},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(ele['audio_url'], 
                                sr=processor.feature_extractor.sampling_rate)[0]
                            )

            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
            inputs.input_ids = inputs.input_ids.to("cuda")

            k = 20
            print(f"Processing ID: {idx}")

            for i in range(k):
                generate_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_k=4,
                    top_p=0.95,
                    num_beams=1,
                    max_new_tokens=1,
                )
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]

                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                with open(output_file, "a") as res:
                    new_idx = f"{idx}_sample{i}"
                    res.write(f"{new_idx} {response}\n")
            
            print("=" * 20)

if __name__ == "__main__":
    main()
