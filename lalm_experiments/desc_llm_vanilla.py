import os
import csv
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_prompt(desc, question, optionA, optionB, optionC, optionD):
    return (
        f"The description of the audio clip is given below:\n{desc}\n"
        f"Based on the information above, answer the following:\n"
        f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\n"
        f"Return only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"
    )

def main(args):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    output_folder = "vanilla_output"
    os.makedirs(output_folder, exist_ok=True)

    if args.task == "neg":
        output_file_path = os.path.join(output_folder, f"negated_desc_llm_{args.task}_vanilla.txt")
    else:
        output_file_path = os.path.join(output_folder, f"desc_llm_{args.task}_vanilla.txt")

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            idx = row["id"]
            desc_file_path = os.path.join(args.desc_folder, f"{idx}.txt")

            with open(desc_file_path, "r") as desc_file:
                desc = desc_file.read()

            prompt = get_prompt(desc, row['question'], row['optionA'], row['optionB'], row['optionC'], row['optionD'])

            messages = [
                {"role": "system", "content": "You are a helpful assistant that must read the description and answer the question. Your response must contain only the option and nothing else"},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            with open(output_file_path, "a") as res_file:
                res_file.write(f"{idx} {response}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--type", type=str, choices=["orig", "neg"], required=True)
    parser.add_argument("--desc_folder", type=str, required=True, help="Folder containing audio description .txt files")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with questions and options")

    args = parser.parse_args()
    main(args)
