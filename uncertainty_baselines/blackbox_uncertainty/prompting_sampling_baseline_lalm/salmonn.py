import csv
import collections
import numpy as np
import re
import argparse
import os
import sys
import torch
import warnings
import soundfile as sf
from typing import Dict, List, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

K = 4
N = 5

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['id'] == target_idx:
                return row['question'], row['optionA'], row['optionB'], row['optionC'], row['optionD']
    return None, None, None, None, None

def setup_salmonn():
    """Setup SALMONN model and dependencies (similar to output_entropy baseline)."""
    try:
        salmonn_path = os.environ.get('SALMONN_HOME')
        if not salmonn_path:
            raise EnvironmentError("SALMONN_HOME environment variable is not set. Please export SALMONN_HOME=/absolute/path/to/SALMONN")
        if salmonn_path not in sys.path:
            sys.path.insert(0, salmonn_path)
        from config import Config
        from models.salmonn import SALMONN
        from transformers import WhisperFeatureExtractor
        return Config, SALMONN, WhisperFeatureExtractor
    except Exception as e:
        print(f"Error setting up SALMONN: {e}")
        raise

def load_salmonn_model():
    """Load SALMONN config, model, and wav processor; return (model, device, wav_processor)."""
    Config, SALMONN, WhisperFeatureExtractor = setup_salmonn()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    salmonn_home = os.environ.get('SALMONN_HOME')
    if not salmonn_home:
        raise EnvironmentError("SALMONN_HOME environment variable is not set. Please export SALMONN_HOME=/absolute/path/to/SALMONN")
    cfg_path = os.path.join(salmonn_home, 'configs', 'decode_config_calibration.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"SALMONN config not found: {cfg_path}")
    cfg = Config(argparse.Namespace(cfg_path=cfg_path, device=device, options=None))
    model = SALMONN.from_config(cfg.config.model)
    model.to(device)
    model.eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    return model, device, wav_processor

def prepare_one_sample(wav_path: str, wav_processor, device: str) -> Dict[str, torch.Tensor]:
    """Prepare audio sample for SALMONN processing (mono, <=30s, spectrogram)."""
    try:
        audio, sr = sf.read(wav_path)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        if len(audio) < sr:
            import numpy as _np
            sil = _np.zeros(sr - len(audio), dtype=float)
            audio = _np.concatenate((audio, sil), axis=0)
        audio = audio[: sr * 30]
        spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
        samples = {
            "spectrogram": spectrogram,
            "raw_wav": torch.from_numpy(audio).unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }
        # move to device
        for k in list(samples.keys()):
            if isinstance(samples[k], torch.Tensor):
                samples[k] = samples[k].to(device)
        return samples
    except Exception as e:
        print(f"Error preparing audio {wav_path}: {e}")
        return {}

input_file = "/home/debarpanb/VLM_project/apoorva_works/Speech_LLM/temporal_dataset_augmented_100samples/order_task/order_subset_100samples.csv"
output_file = "lalm_results/salmonn_topk_sampling_order.csv"
audio_dir = "/home/debarpanb/VLM_project/apoorva_works/Speech_LLM/temporal_dataset/ESC_50_reasoning_order_dataset/audios"

with open(output_file, mode="w", newline="", encoding="utf-8") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["idx", "guess", "probability"])

    with open(input_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']
            audio_path = f"{audio_dir}/{idx}.wav"

            question = row['question']
            optionA = row['optionA']
            optionB = row['optionB']
            optionC = row['optionC']
            optionD = row['optionD']

            qns = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
            prompt = (
                f"Provide your {K} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. "
                "Each guess must be either 'A','B', 'C' or 'D' ONLY. Give ONLY the guesses and probabilities, no other words or explanation."
                f"You MUST follow the template given below to generate {K} guesses and probabilities\n\n"
                "G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
                "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n"
                "...\n"
                f"G{K}: <{K}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
                f"P{K}: <the probability between 0.0 and 1.0 that G{K} is correct, without any extra commentary whatsoever; just the probability!>\n\n"
                f"The question is: {qns}"
            )

            # Prepare samples
            samples = prepare_one_sample(audio_path, wav_processor, device)
            if not samples:
                continue

            all_guesses = collections.defaultdict(list)
            for _ in range(N):
                try:
                    with torch.no_grad():
                        generate_text = model.generate(
                            samples,
                            prompt,
                            max_length=256,
                            num_beams=1,
                            temperature=0.7,
                            do_sample=True,
                        )
                    if isinstance(generate_text, list):
                        generate_text = generate_text[0] if generate_text else ""
                    response = str(generate_text)
                    print(f"{idx} [Sample]:\n{response}\n")

                    group_answers = dict(re.findall(r"G(\d+):\s*([A-D])", response))
                    probabilities = dict(re.findall(r"P(\d+):\s*([0-1](?:\.\d+)?)", response))

                    for k_i in group_answers:
                        if k_i in probabilities:
                            guess = group_answers[k_i].strip()
                            prob = float(probabilities[k_i].strip())
                            all_guesses[guess].append(prob)
                except Exception as e:
                    print(f"SALMONN generation error for {idx}: {e}")
            if all_guesses:
                avg_conf = {guess: np.mean(probs) for guess, probs in all_guesses.items()}
                best_guess, best_prob = max(avg_conf.items(), key=lambda x: x[1])
            else:
                best_guess, best_prob = "N/A", 0.0
            writer.writerow([idx, best_guess, round(best_prob, 3)])
