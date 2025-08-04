import os
import csv
import torch
import argparse
from transformers import WhisperFeatureExtractor
from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample

def get_prompt(question, optionA, optionB, optionC, optionD):
    return f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

def main(args):
    cfg_path = "configs/decode_config_sampling.yaml"
    output_folder = "perturb_sampling_output"
    os.makedirs(output_folder, exist_ok=True)

    if args.type == "neg":
        output_file_path = os.path.join(output_folder, f"negated_salmonn_{args.task}_perturb_sampling.txt")
    else:
        output_file_path = os.path.join(output_folder, f"salmonn_{args.task}_perturb_sampling.txt")

    cfg = Config(argparse.Namespace(cfg_path=cfg_path, device="cuda:0", options=None))
    model = SALMONN.from_config(cfg.config.model)
    model.to("cuda:0")
    model.eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']
            wav_path = os.path.join(args.wav_folder, f"{idx}.wav")
            prompt_text = get_prompt(row['question'], row['optionA'], row['optionB'], row['optionC'], row['optionD'])

            samples = prepare_one_sample(wav_path, wav_processor)
            full_prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())]
            
            k = 10
            for i in range(k):
                new_idx = f"{idx}_sample{i}"
                
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    text,generated_scores = model.generate_calibration(samples, cfg.config.generate, prompts=prompt)

                with open(output_file_path, "a") as f:
                    f.write(f"{new_idx} {text}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["orig", "neg"], required=True)
    parser.add_argument("--wav_folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
