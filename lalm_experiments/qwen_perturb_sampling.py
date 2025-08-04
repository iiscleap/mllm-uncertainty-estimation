import os
import csv
import torch
import argparse
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def get_prompt(question, optionA, optionB, optionC, optionD):
    return f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

def get_base_idx(idx):
    return idx.split('_rephrased')[0]

def main(args):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

    output_folder = "perturb_sampling_output"
    os.makedirs(output_folder, exist_ok=True)

    if args.type == "neg":
        output_file_path = os.path.join(output_folder, f"negated_qwen_{args.task}_perturb_sampling.txt")
    else:
        output_file_path = os.path.join(output_folder, f"qwen_{args.task}_perturb_sampling.txt")

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['idx']
            base_idx = get_base_idx(idx)
            audio_path = os.path.join(args.wav_folder, f"{base_idx}.wav")

            question = row['question']
            optionA = row['optionA']
            optionB = row['optionB']
            optionC = row['optionC']
            optionD = row['optionD']

            prompt = get_prompt(question, optionA, optionB, optionC, optionD)

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': [{'type': 'audio', 'audio_url': audio_path}, {"type": "text", "text": prompt}]}
            ]

            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            waveform, _ = librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)
                            audios.append(waveform)

            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to("cuda")

            k = 10        
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
                
                with open(output_file_path, "a") as f:
                    new_idx = f"{idx}_sample{i}"
                    f.write(f"{new_idx} {response}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["orig", "neg"], required=True)
    parser.add_argument("--wav_folder", type=str, required=True, help="Folder containing .wav files named as {base_idx}.wav")

    args = parser.parse_args()
    main(args)
