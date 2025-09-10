import os
import csv
import torch
import argparse
import sys
import time
import warnings
import re
from transformers import WhisperFeatureExtractor
import soundfile as sf
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Import SALMONN modules (adapt path based on your setup)
sys.path.append('/home/debarpanb/VLM_project/malay_works/FESTA-uncertainty-estimation/lalm_experiments/SALMONN')
from config import Config
from models.salmonn import SALMONN

def prepare_one_sample(wav_path, wav_processor):
    """Prepare audio sample for SALMONN processing"""
    audio, sr = sf.read(wav_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio[:, 0]
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 30]  # truncate audio to at most 30s

    spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]

    samples = {
        "spectrogram": spectrogram,
        "raw_wav": torch.from_numpy(audio).unsqueeze(0),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
    }
    return samples

def map_audio_path_to_16khz(original_path):
    """
    Map audio paths from 44kHz 'audios' folders to 16kHz 'wav_folder' directories.
    
    Example:
    TREA_dataset/count/audios/0.wav -> TREA_dataset/count/wav_folder/0.wav
    """
    if "/audios/" in original_path:
        return original_path.replace("/audios/", "/wav_folder/")
    elif "audios/" in original_path:
        return original_path.replace("audios/", "wav_folder/")
    else:
        return original_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run SALMONN model on TREA dataset tasks')
    parser.add_argument('--task', type=str, choices=['count', 'duration', 'order'], 
                       required=True, help='Task to run: count, duration, or order')
    parser.add_argument('--output_dir', type=str, default='lalm_results', 
                       help='Output directory for results (relative to script location)')
    parser.add_argument('--dataset_path', type=str, default='/home/debarpanb/VLM_project/TREA_dataset',
                       help='Path to TREA dataset')
    
    args = parser.parse_args()
    
    # Set the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading SALMONN model on {device}...")
    
    # Load SALMONN model
    cfg_path = "/home/debarpanb/VLM_project/malay_works/FESTA-uncertainty-estimation/lalm_experiments/SALMONN/configs/decode_config_calibration.yaml"
    cfg = Config(argparse.Namespace(cfg_path=cfg_path, device=device, options=None))
    
    model = SALMONN.from_config(cfg.config.model)
    model.to(device)
    model.eval()
    
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    # Set up file paths based on task
    task = args.task
    input_file = f"{args.dataset_path}/{task}/{task}.csv"
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    output_file = os.path.join(output_dir, f"salmonn_guess_prob_trea_{task}.csv")
    
    print(f"Processing task: {task}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open output CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["idx", "guess", "probability"])

        # Read input file
        with open(input_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            total_rows = sum(1 for _ in csv.DictReader(open(input_file, 'r')))
            print(f"Total samples to process: {total_rows}")
            
            file.seek(0)  # Reset file pointer
            reader = csv.DictReader(file)
            
            for i, row in enumerate(reader):
                original_audio_path = row['audio_path']
                idx = row['id']

                # Map the audio path to use 16kHz resampled version
                mapped_audio_path = map_audio_path_to_16khz(original_audio_path)
                
                # Construct full audio path
                full_audio_path = f"{args.dataset_path}/../{mapped_audio_path}"
                
                print(f"Sample {i+1}/{total_rows} - Mapping audio path:")
                print(f"  Original: {original_audio_path}")
                print(f"  Mapped:   {mapped_audio_path}")
                print(f"  Full:     {full_audio_path}")
                
                # Check if audio file exists
                if not os.path.exists(full_audio_path):
                    print(f"Warning: Audio file not found: {full_audio_path}")
                    continue

                question = row['question']
                optionA = row['optionA']
                optionB = row['optionB']
                optionC = row['optionC']
                optionD = row['optionD']

                qns = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
                prompt = (
                    "You must provide both the answer choice (A, B, C, or D) and the probability that your answer is correct "
                    "(a number between 0.0 and 1.0). Your response MUST include BOTH the guess and the probability, "
                    "in the exact format shown below. Do NOT include any explanations, comments, or extra words.\n\n"
                    "Format:\n"
                    "Guess: <A/B/C/D>\n"
                    "Probability: <number between 0.0 and 1.0>\n\n"
                    f"The question is:\n{qns}"
                )

                try:
                    # Prepare audio sample
                    samples = prepare_one_sample(full_audio_path, wav_processor)
                    
                    # Ensure samples are on the correct device
                    for key in samples:
                        if isinstance(samples[key], torch.Tensor):
                            samples[key] = samples[key].to(device)
                    
                    # Create full prompt for SALMONN
                    full_prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())]

                    # Generate response
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        text = model.generate(samples, cfg.config.generate, prompts=full_prompt)

                    response = text[0] if isinstance(text, list) else text
                    print(f"Sample {i+1}/{total_rows} - Response: {response}")

                    # Extract guess and probability using regex
                    guess_match = re.search(r"Guess:\s*([A-D])", response)
                    prob_match = re.search(r"Probability:\s*([0-9]*\.?[0-9]+)", response)

                    guess = guess_match.group(1) if guess_match else ""
                    prob = prob_match.group(1) if prob_match else ""

                    writer.writerow([idx, guess, prob])
                    print(f"Sample {i+1}/{total_rows} - ID: {idx}, Guess: {guess}, Probability: {prob}")
                    print("=" * 50)
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                    writer.writerow([idx, "", ""])
                    continue
    
    print(f"Processing complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
