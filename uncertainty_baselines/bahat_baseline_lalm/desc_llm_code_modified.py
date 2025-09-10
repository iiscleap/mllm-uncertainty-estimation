import csv
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# Get task and experiment type from command line arguments
if len(sys.argv) != 3:
    print("Usage: python desc_llm_code_modified.py <task> <exp_type>")
    print("task: count, order, duration")
    print("exp_type: audio_only, text_only, text_audio")
    sys.exit(1)

task = sys.argv[1]
exp = sys.argv[2]

print(f"Running desc_llm on task: {task}, experiment: {exp}")

if exp == "audio_only":
    csv_filepath = f"{task}_audio_perturbations_only.csv"
    desc_column = "idx"
    desc_dir = f"{task}_audio_perturbations_only_desc"

elif exp == "text_only":
    csv_filepath = f"{task}_text_perturbations_only.csv"
    desc_column = "orig_idx"
    desc_dir = f"/path/to/desc_dir"

elif exp == "text_audio":
    csv_filepath = f"/path/to/desc_dir/{task}_perturbed.csv"
    desc_dir = f"/path/to/desc_dir/{task}/perturbed_audio_desc"

else:
    print(f"Invalid experiment type: {exp}")
    sys.exit(1)

# Clear the output file
output_file = f"desc_llm_results/desc_llm_{exp}_{task}.txt"
with open(output_file, "w") as f:
    pass

with open(csv_filepath, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        idx = row["idx"]
        
        # Handle description file lookup based on experiment type
        if exp == "text_audio":
            # For TREA dataset: 59_add_delete_audio_0_rephrased1 -> 59_add_delete_audio_0.txt
            parts = idx.split('_')
            if 'rephrased' in idx:
                rephrased_idx = next(i for i, part in enumerate(parts) if part.startswith('rephrased'))
                desc_idx = '_'.join(parts[:rephrased_idx])
            else:
                desc_idx = idx
        else:
            desc_idx = row[desc_column]

        try:
            with open(f"{desc_dir}/{desc_idx}.txt", "r") as file:
                desc = file.read()
        except FileNotFoundError:
            print(f"Warning: Description file {desc_dir}/{desc_idx}.txt not found, skipping...")
            continue
        
        question = row['question']
        optionA = row['optionA']
        optionB = row['optionB']
        optionC = row['optionC']
        optionD = row['optionD']
        prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
        prompt = f"The description of the audio clip is given below:\n{desc}\nBased on the information above, answer the following:\n{prompt}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

        messages = [
            {"role": "system", "content": "You are a helpful assistant that must read the description and answer the question. Your response must contain only the option and nothing else"},
            {"role": "user", "content": prompt},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=1)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        with open(output_file, "a") as res:
            res.write(f"{idx} {response}\n")
        
        print(f"Processed {idx}")

print(f"Completed {task} - {exp}")
