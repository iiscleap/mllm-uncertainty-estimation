from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import csv
import collections
import re
import numpy as np

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

qns_filepath = "path/to/vsr_data/questions.csv"
output_path = "vlm_results/qwenvl_topk_sampling_vsr.csv"
img_folder = "path/to/vsr_data/orig_images"

K = 4
N = 5

with open(output_path, mode="w", newline="", encoding="utf-8") as out_file:
    fieldnames = ["idx", "guess", "probability"]
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(111, 211):
        idx = f"val_Spatial_Reasoning_{i}"
        qns = get_qns_for_idx(qns_filepath, idx)
        qns = f"{qns}\nChoices:\nA. True\nB. False"

        img_path = f"{img_folder}/{idx}.jpg"

        prompt = (
            f"Provide your {K} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. "
            "Each guess must be either 'A' or 'B' only. Give ONLY the guesses and probabilities, no other words or explanation."
            f"You MUST follow the template given below to generate {K} guesses and probabilities\n\n"
            "G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
            "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n"
            "...\n"
            f"G{K}: <{K}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"
            f"P{K}: <the probability between 0.0 and 1.0 that G{K} is correct, without any extra commentary whatsoever; just the probability!>\n\n"
            f"The question is: {qns}"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]


        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        all_guesses = collections.defaultdict(list)

        for _ in range(N):
            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=100,
                                do_sample=True,
                                temperature=0.7
                            )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(f"{idx} [Sample]:\n{output_text[0]}\n")

            matches = re.findall(r"G\d+:\s*(.*?)\s+P\d+:\s*([0-1](?:\.\d+)?)", output_text[0])
            for guess, prob in matches:
                guess = guess.strip()
                prob = float(prob.strip())
                all_guesses[guess].append(prob)


        if all_guesses:
            avg_conf = {guess: np.mean(probs) for guess, probs in all_guesses.items()}
            best_guess, best_prob = max(avg_conf.items(), key=lambda x: x[1])
        else:
            best_guess, best_prob = "N/A", 0.0
        
        print(f"{idx}, {guess}, {best_prob}")
        writer.writerow({
            "idx": idx,
            "guess": best_guess,
            "probability": round(best_prob, 3)
        })