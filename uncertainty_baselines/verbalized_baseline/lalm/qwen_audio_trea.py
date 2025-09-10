from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import csv
import torch 
import os
import re
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Qwen Audio model on TREA dataset tasks')
    parser.add_argument('--task', type=str, choices=['count', 'duration', 'order'], 
                       required=True, help='Task to run: count, duration, or order')
    parser.add_argument('--output_dir', type=str, default='lalm_results', 
                       help='Output directory for results (relative to script location)')
    parser.add_argument('--dataset_path', type=str, default='/home/debarpanb/VLM_project/TREA_dataset',
                       help='Path to TREA dataset')
    
    args = parser.parse_args()
    
    # Load processor and model
    print(f"Loading Qwen2-Audio-7B-Instruct model...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
    
    # Set up file paths based on task
    task = args.task
    input_file = f"{args.dataset_path}/{task}/{task}.csv"
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    output_file = os.path.join(output_dir, f"qwen_guess_prob_trea_{task}.csv")
    
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
                audio_path = row['audio_path']
                idx = row['id']

                # Construct full audio path
                full_audio_path = f"{args.dataset_path}/../{audio_path}"
                
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

                conversation = [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': [
                        {'type': 'audio', 'audio_url': full_audio_path},
                        {'type': 'text', 'text': prompt}
                    ]}
                ]
                
                try:
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

                    audios = []
                    for message in conversation:
                        if isinstance(message["content"], list):
                            for ele in message["content"]:
                                if ele["type"] == "audio":
                                    waveform, _ = librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)
                                    audios.append(waveform)

                    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
                    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                    generate_ids = model.generate(**inputs, max_new_tokens=20)
                    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

                    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
