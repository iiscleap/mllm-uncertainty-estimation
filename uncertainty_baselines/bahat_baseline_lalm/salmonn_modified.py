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

def map_to_resampled_path(original_path, exp, task, audio_base_dir):
    """Map original audio paths to resampled 16kHz paths"""
    
    if exp == "text_only":
        # For text_only: audios/59.wav -> wav_folder/59.wav
        if original_path.startswith("audios/"):
            filename = original_path.replace("audios/", "")
            return f"{audio_base_dir}/wav_folder/{filename}"
        else:
            return original_path
            
    elif exp == "audio_only":
        # For audio_only, paths are already correct
        return original_path
        
    elif exp == "text_audio":
        # For text_audio, map '.../perturbed_audio/...' to '.../wav_folder/...'
        if "perturbed_audios" in original_path:
            return original_path.replace("perturbed_audios", "resampled_perturbed_audios")
        elif "/audios/" in original_path:
            return original_path.replace("/audios/", "/wav_folder/")
        else:
            return original_path
    
    return original_path

# Get task and experiment type from command line arguments
if len(sys.argv) != 3:
    print("Usage: python salmonn_modified.py <task> <exp_type>")
    print("task: count, order, duration")
    print("exp_type: audio_only, text_only, text_audio")
    sys.exit(1)

task = sys.argv[1]
exp = sys.argv[2]

print(f"Running salmonn on task: {task}, experiment: {exp}")

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

# Set file paths based on experiment type
if exp == "audio_only":
    csv_filepath = f"{task}_audio_perturbations_only.csv"
    audio_path_column = "new_audio_path"
    audio_base_dir = ""

elif exp == "text_only":
    csv_filepath = f"{task}_text_perturbations_only.csv"
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

# Check if CSV file exists
if not os.path.exists(csv_filepath):
    print(f"CSV file not found: {csv_filepath}")
    sys.exit(1)

# Clear the output file
output_file = f"salmonn_results/salmonn_{exp}_{task}.txt"
os.makedirs("salmonn_results", exist_ok=True)
with open(output_file, "w") as f:
    pass

print(f"Processing file: {csv_filepath}")
processed_count = 0
error_count = 0

with open(csv_filepath, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        idx = row["idx"]
        
        # Handle audio file path based on experiment type
        if exp == "audio_only":
            original_audio_path = row[audio_path_column]
            audio_path = original_audio_path  # Keep as-is for audio_only
            
        elif exp == "text_only":
            audio_id = row["orig_idx"]
            # Use the resampled wav_folder path
            audio_path = f"{audio_base_dir}/wav_folder/{audio_id}.wav"
            
        elif exp == "text_audio":
            # TREA dataset: use the audio file path and map to wav_folder
            original_audio_file = row[audio_path_column]
            mapped_audio_file = map_to_resampled_path(original_audio_file, exp, task, audio_base_dir)
            audio_path = f"{audio_base_dir}/{mapped_audio_file}"
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file {audio_path} not found, skipping...")
            error_count += 1
            continue

        question = row['question']
        optionA = row['optionA']
        optionB = row['optionB']
        optionC = row['optionC']
        optionD = row['optionD']

        qns = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
        prompt = f"Please answer the following question based on the audio. Return only the option (A, B, C, or D), and nothing else.\n\n{qns}"

        try:
            # Prepare audio sample
            samples = prepare_one_sample(audio_path, wav_processor)
            if samples is None:
                error_count += 1
                continue
            
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
            
            # Extract just the first character/option from response
            response_clean = response.strip()
            if response_clean and response_clean[0] in ['A', 'B', 'C', 'D']:
                answer = response_clean[0]
            else:
                # Try to find A, B, C, or D in the response
                match = re.search(r'\b([A-D])\b', response_clean)
                answer = match.group(1) if match else response_clean[:1]

            with open(output_file, "a") as res:
                res.write(f"{idx} {answer}\n")
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} samples...")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            error_count += 1
            # Write empty response on error
            with open(output_file, "a") as res:
                res.write(f"{idx} \n")
            continue

print(f"Completed {task} - {exp}")
print(f"Successfully processed: {processed_count} samples")
print(f"Errors: {error_count} samples")
print(f"Results saved to: {output_file}")
