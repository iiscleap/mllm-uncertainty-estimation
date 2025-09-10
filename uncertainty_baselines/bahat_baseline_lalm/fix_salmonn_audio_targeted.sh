#!/bin/bash

echo "==========================================="
echo "SALMONN Audio Resampling - Targeted Paths"
echo "==========================================="
echo "This script will resample audio files from the exact paths"
echo "referenced in salmonn_modified.py"
echo ""

# Check if librosa and soundfile are available
python3 -c "import librosa, soundfile" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required libraries not found. Installing..."
    pip install librosa soundfile tqdm
fi

# Define the specific audio directories used in salmonn_modified.py
audio_directories=(
    "/path/to/TREA_dataset_negated/count/audios"
    "/path/to/TREA_dataset_negated/count/perturbed_audios"
    "/path/to/TREA_dataset_negated/order/audios"
    "/path/to/TREA_dataset_negated/order/perturbed_audios"
    "/path/to/TREA_dataset_negated/duration/audios"
    "/path/to/TREA_dataset_negated/duration/perturbed_audios"
)

echo "Processing the following directories:"
for dir in "${audio_directories[@]}"; do
    echo "  - $dir"
done
echo ""

# Process each audio directory
for audio_dir in "${audio_directories[@]}"; do
    echo "=========================================="
    echo "Processing: $audio_dir"
    
    if [ ! -d "$audio_dir" ]; then
        echo "Warning: Directory $audio_dir does not exist, skipping..."
        continue
    fi
    
    # Create resampled folder instead of wav_folder
    base_folder_name=$(basename "$audio_dir")
    parent_dir=$(dirname "$audio_dir")
    wav_folder_dir="${parent_dir}/resampled_${base_folder_name}"
    
    echo "  Input:  $audio_dir"
    echo "  Output: $wav_folder_dir"
    
    mkdir -p "$wav_folder_dir"
    
    # Count files before processing
    audio_count=$(find "$audio_dir" -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" -o -name "*.ogg" | wc -l)
    
    if [ "$audio_count" -gt 0 ]; then
        echo "  Found $audio_count audio files to process"
        
        existing_count=$(ls "$wav_folder_dir"/*.wav 2>/dev/null | wc -l)
        if [ "$existing_count" -eq "$audio_count" ]; then
            echo "  ✓ Already processed: $existing_count files exist in $wav_folder_dir"
        else
            echo "  Resampling audio files..."
            python3 resample_audio_16khz.py --input_dir "$audio_dir" --output_dir "$wav_folder_dir"
            
            processed_count=$(ls "$wav_folder_dir"/*.wav 2>/dev/null | wc -l)
            echo "  ✓ Processed: $processed_count files in $wav_folder_dir"
        fi
    else
        echo "  No audio files found in $audio_dir"
    fi
    
    echo ""
done


echo "==========================================="
echo "Audio resampling completed!"
echo ""
echo "Summary of resampled directories:"
for audio_dir in "${audio_directories[@]}"; do
    base_folder_name=$(basename "$audio_dir")
    parent_dir=$(dirname "$audio_dir")
    resampled_dir="${parent_dir}/resampled_${base_folder_name}"
    if [ -d "$resampled_dir" ]; then
        count=$(ls "$resampled_dir"/*.wav 2>/dev/null | wc -l)
        echo "  $resampled_dir: $count files"
    else
        echo "  $resampled_dir: (not found)"
    fi
done
echo "==========================================="
