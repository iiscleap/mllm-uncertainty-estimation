import transformers
import torch
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Rephrase questions using LLM.")
parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
args = parser.parse_args()


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def extract_questions(text):
    return [q.strip() for q in text.split('\n') if q.strip()]

def generate_variants(question):
    prompt = f"Generate 4 rephrased versions of the following statement while ensuring that the meaning is the EXACT same: {question}\nYour response should contain the 4 statements in 4 different lines ONLY."

    messages = [
        {"role": "system", "content": "You are an assistant that rephrases statements while keeping their meaning intact."},
        {"role": "user", "content": prompt},
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    
    generated_text = outputs[0]["generated_text"][-1]['content']
    print(generated_text)
    print("=" * 25)
    return extract_questions(generated_text)

def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    output_rows = []

    for _, row in df.iterrows():
        question = row["question"]
        original_idx = row["idx"]
        variants = generate_variants(question)

        for i, variant in enumerate(variants, 1):
            new_row = row.copy()
            new_row["question"] = variant
            new_row["image_idx"] = original_idx
            new_row["idx"] = f"{original_idx}_rephrased{i}"
            output_rows.append(new_row)
            print(f"{new_row['idx']}: {variant}")
        print("*" * 30)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved as: {output_csv}")

process_csv(args.input_csv, args.output_csv)
