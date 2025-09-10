import os
import csv
import torch
import sys
import time
import warnings
import re
import argparse
import soundfile as sf
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_salmonn():
    """Setup SALMONN model and dependencies"""
    try:
        # Add SALMONN path
        salmonn_path = '/path/to/SALMONN'
        if salmonn_path not in sys.path:
            sys.path.insert(0, salmonn_path)
        
        # Import required modules
        from config import Config
        from models.salmonn import SALMONN
        from transformers import WhisperFeatureExtractor
        
        return Config, SALMONN, WhisperFeatureExtractor
    
    except ImportError as e:
        print(f"Error importing SALMONN modules: {e}")
        print("Please ensure SALMONN dependencies are installed:")
        print("pip install omegaconf transformers soundfile")
        sys.exit(1)

def prepare_one_sample(wav_path, wav_processor):
    """Prepare audio sample for SALMONN processing"""
    try:
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
    except Exception as e:
        print(f"Error preparing audio sample {wav_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='SALMONN vanilla predictions for LALM')
    parser.add_argument('--task', type=str, required=True, choices=['count', 'duration', 'order'])
    parser.add_argument('--exp_type', type=str, required=True, choices=['orig', 'neg'])
    args = parser.parse_args()
    
    # Setup SALMONN
    Config, SALMONN, WhisperFeatureExtractor = setup_salmonn()
    
    # Set the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading SALMONN model on {device}...")
    
    # Load SALMONN model
    cfg_path = "/path/to/SALMONN/configs/decode_config_calibration.yaml"
    
    if not os.path.exists(cfg_path):
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)
    
    try:
        cfg = Config(argparse.Namespace(cfg_path=cfg_path, device=device, options=None))
        
        model = SALMONN.from_config(cfg.config.model)
        model.to(device)
        model.eval()
        
        wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
        print("✓ SALMONN model loaded successfully")
    
    except Exception as e:
        print(f"Error loading SALMONN model: {e}")
        sys.exit(1)
    
    task = args.task
    exp_type = args.exp_type
    
    # Create output directory
    output_dir = f"{exp_type}/vanilla"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output filename
    if exp_type == "orig":
        output_file = f"{output_dir}/salmonn_{task}_vanilla.txt"
    else:
        output_file = f"{output_dir}/negated_salmonn_{task}_vanilla.txt"
    
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

            # Prepare audio sample for SALMONN
            samples = prepare_one_sample(full_audio_path, wav_processor)
            if samples is None:
                print(f"Skipping {idx} due to audio processing error")
                continue
            
            # Move samples to device
            for key in samples:
                if isinstance(samples[key], torch.Tensor):
                    samples[key] = samples[key].to(device)
            
            question = row['question']
            
            # For negated experiments, modify the question
            if exp_type == "neg":
                question = f"Is it NOT the case that: {question}"
            
            optionA = row['optionA']
            optionB = row['optionB']
            optionC = row['optionC']
            optionD = row['optionD']
            
            # Create prompt for SALMONN
            prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

            print(f"Processing ID: {idx}")

            # Process with SALMONN
            try:
                with torch.no_grad():
                    response = model.generate(
                        samples,
                        prompt,
                        max_length=1,
                        num_beams=1,
                        temperature=1.0,
                        do_sample=False
                    )
                
                # Extract the response (assuming it returns the generated text)
                if isinstance(response, list):
                    response = response[0] if response else ""
                response = str(response).strip()
                
            except Exception as e:
                print(f"Error during generation for {idx}: {e}")
                response = "ERROR"
            
            with open(output_file, "a") as res:
                res.write(f"{idx} {response}\n")

if __name__ == "__main__":
    main()
