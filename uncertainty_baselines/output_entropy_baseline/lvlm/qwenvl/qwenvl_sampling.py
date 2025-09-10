from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import csv

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
qns_filepath = "blink_data/question.csv"

for i in range(1,144):
    idx = f"val_Spatial_Relation_{i}"
    qns = get_qns_for_idx(qns_filepath, idx)
    img_path = f"blink_data/orig_images/val_Spatial_Relation_{i}.jpg"
    prompt = f"{qns}\nChoices:\nA. Yes\nB. No\nReturn only the option (A or B), and nothing else.\nMAKE SURE your output is A or B"

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

    k = 20
    for i in range(k):
        new_idx = f"{idx}_sample{i}"
        generated_ids = model.generate(
                                    **inputs,
                                    do_sample=True,
                                    temperature=1.0,
                                    top_p=0.95,
                                    num_beams=1,
                                    max_new_tokens=1,
                                )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output_text[0])

        with open("qwen_results/qwenvl_sampling.txt", "a") as res:
            res.write(f"{new_idx} {output_text[0]}\n")
