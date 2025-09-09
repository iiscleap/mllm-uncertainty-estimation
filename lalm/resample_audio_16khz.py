#!/usr/bin/env python3
"""
Audio Resampling Script for SALMONN
Resamples audio files from 44.1kHz to 16kHz to fix the SALMONN model error.
"""

import os
import sys
import shutil
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm

def resample_audio_file(input_path, output_path, target_sr=16000):
    """
    Resample a single audio file to target sample rate.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        target_sr: Target sample rate (default: 16000)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=None)
        
        # Resample if needed
        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_resampled = audio
            
        # Save resampled audio
        sf.write(output_path, audio_resampled, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def resample_audio_directory(input_dir, output_dir, target_sr=16000):
    """
    Resample all audio files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_sr: Target sample rate (default: 16000)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f'*{ext}'))
        audio_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return 0
    
    print(f"Found {len(audio_files)} audio files to resample")
    print(f"Resampling from {input_dir} to {output_dir}")
    
    success_count = 0
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Resampling"):
        output_file = output_path / f"{audio_file.stem}.wav"
        
        if resample_audio_file(str(audio_file), str(output_file), target_sr):
            success_count += 1
        
    print(f"Successfully resampled {success_count}/{len(audio_files)} files")
    return success_count

def main():
    parser = argparse.ArgumentParser(description='Resample audio files to 16kHz for SALMONN')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for resampled audio files')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    resample_audio_directory(args.input_dir, args.output_dir, args.target_sr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
