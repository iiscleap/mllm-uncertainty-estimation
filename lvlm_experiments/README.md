# FESTA-uncertainty-estimation
Uncertainty estimation using equivalent and complimentary input sampling

## Dataset Structure

There are two datasets: `BLINK` and `VSR`, each containing image and question files.
1. `BLINK` : `/home/debarpanb/VLM_project/BLINK`
2. `VSR` : `/home/debarpanb/VLM_project/VSR`

Copy the datasets from the given paths to `FESTA-uncertainty-estimation/lvlm_experiments/dataset`

### Folder Overview

Each of `BLINK/` and `VSR/` contains:

- **Image Folders**:
  - `orig_images/`: Images for both original questions and negated questions
  - `perturbed_images/`: Perturbed images corresponding to original questions.
  - `perturbed_negated_images/`: Perturbed images corresponding to negated questions.

- **Text and CSV Files**:
  - `answer_list.txt`: Ground truth answers for the original questions.
  - `questions.csv`: Original set of questions
  - `negated_questions.csv`: Negated versions of the original questions.
  - `perturbed_questions.csv`: Perturbed questions for the original set
  - `perturbed_negated_questions.csv`: Perturbed questions for the negated set


## Running the scripts

There are 2 different scripts - vanilla and perturb_sampling.

Vanilla is to be run on the `orig_images` and `questions.csv` OR `orig_images` and` negated_questions.csv`. Outputs are saved in `vanilla_output`

Perturb_sampling is to be run on the `perturbed_images` and `perturbed_questions.csv` OR `perturbed_negated_images` and` perturbed_negated_questions.csv`. Outputs are saved in `perturb_sampling_output`

```bash
python gemma3_vanilla.py \
  --input_csv path/to/questions.csv \
  --input_image_folder path/to/images \
  --dataset blink \
  --type neg
```
## Setup for each model

1. Gemma3 - Can be directly run on the Prajna cluster using the `gemma3` conda env
2. Llava - Can be directly run on the Prajna cluster using the `irl_torch2` conda env
3. QwenVL - Can be run on the Prajna cluster using the `irl_torch2` conda env
4. Pixtral - Needs CUDA 12 so it can only be run on GCP. Pixtral is based on vLLM. vLLM installation instructions can be found at [https://docs.vllm.ai/en/v0.8.1/getting_started/installation/gpu.html].
5. Phi4 - Can be run on the Prajna cluster using the `phi4` conda env. Model is downloaded locally and needs to be moved from `/home/debarpanb/VLM_project/Phi-4-multimodal-instruct` to `FESTA-uncertainty-estimation/lvlm_experiments/`

NOTE: Check if the models are restricted on HuggingFace and require access request

## Calculating the FESTA AUC

The outputs have already been saved in `vanilla_output/` and `perturb_sampling_output/`
To obtain the AUC score:

```bash
python kl_div_fusion.py \
  --dataset gemma3 \
  --model qwenvl
```
