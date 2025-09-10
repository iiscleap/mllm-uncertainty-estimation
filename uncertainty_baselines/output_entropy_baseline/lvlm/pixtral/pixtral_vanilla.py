from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login
import csv
import os
import base64

login(os.environ.get("HF_TOKEN"))


def file_to_data_url(file_path: str):
    """
    Convert a local image file to a data URL.
    """    
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    
    return f"data:{mime_type};base64,{encoded_string}"


def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']


model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=1)
qns_filepath = "blink_data/question.csv"

llm = LLM( model=model_name,
          gpu_memory_utilization=0.95,
          max_model_len=4096,
          tokenizer_mode="mistral",
          load_format="mistral",
          config_format="mistral"
         )

for i in range(1,144):
    idx = f"val_Spatial_Relation_{i}"
    qns = get_qns_for_idx(qns_filepath, idx)
    img_path = f"blink_data/orig_images/val_Spatial_Relation_{i}.jpg"
    prompt = f"{qns}\nChoices:\nA. Yes\nB. No\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"

    image_source = file_to_data_url(img_path)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_source}}]
        },
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    final_res = outputs[0].outputs[0].text

    # print(outputs[0].outputs[0].text)

    with open("pixtral_results/pixtral_vanilla.txt", "a") as resfile:
        resfile.write(f"{idx} {final_res}\n")

