# FESTA-uncertainty-estimation
uncertainty estimation using equivalent and complimentary input sampling

## TREA Dataset Structure

There are two versions of the TREA dataset:
1. `TREA_dataset` : `/home/debarpanb/VLM_project/TREA_dataset`
2. `TREA_dataset_negated` : `/home/debarpanb/VLM_project/TREA_dataset_negated`

Copy the datasets from the paths to `lalm_experiments/dataset/`

---

### 1. TREA_dataset/

This is the original version of the dataset.

#### └── count/
Contains data related to the "count" task.

- **audio_desc/**: Textual audio descriptions for each example.
- **audios/**: Original `.wav` audio files.
- **perturbed_audio_desc/**: Descriptions for perturbed versions of audio.
- **perturbed_audios/**: Perturbed `.wav` audio files.
- **count.csv**: CSV file for original questions
- **count_perturbed.csv**: CSV file for perturbed questions
- **count_with_metadata.csv**: Combined metadata with additional info.

---

### 2. TREA_dataset_negated/

This contains a "negated" version of the same data as `TREA_dataset`, designed for contrastive analysis.

#### └── count/
- **audio_desc/**: Negated descriptions.
- **audios/**: Negated `.wav` audio files.
- **perturbed_audio_desc/**: Descriptions for perturbed, negated audios.
- **perturbed_audios/**: Perturbed negated audio files.
- **count_negated.csv**: CSV file for negated questions
- **count_negated_perturbed.csv**: CSV file for perturbed negated questions

---

### Other Tasks
In both datasets:
- `duration/` and `order/` folders follow a similar structure as `count/`.

---
## Vanilla Inference Script
### What it does
- Loads original audio and question data from a CSV containing original questions.
- Generates prompts with audio and text.
- Predicts options (A/B/C/D) using the model.
- Saves outputs to `vanilla_output/`.

### How to run (Qwen Example)
```bash
python qwen_vanilla.py \
  --task count \
  --csv_path path/to/count.csv \
  --type orig \
  --wav_folder path/to/audios
```

* `task`: `count`, `duration`, or `order`
* `type`: `orig` or `neg`

Use `neg` when running vanilla and perturb sampling scripts on negated dataset
Set the `csv_path` and `wav_folder` according to the `task` and `type`

---

## Perturb Sampling Inference Script
### What it does
- Loads questions and corresponding audio paths from a CSV containing perturbed questions.
- Uses top-k sampling to generate **10 varied outputs** per question.
- Saves results to `perturb_sampling_output/`.

### How to run (Qwen Example)
```bash
python qwen_perturb_sampling.py \
  --task count \
  --csv_path path/to/count_perturbed.csv \
  --type orig \
  --wav_folder path/to/audios
```
---
## Running SALMONN Scripts

To run the `salmonn_vanilla.py` and `salmonn_perturb_sampling.py` scripts, follow these steps:

1. **Download and set up SALMONN**  
   Clone and set up the official SALMONN repository from [https://github.com/bytedance/SALMONN](https://github.com/bytedance/SALMONN).

2. **Move configuration files**  
   Move the `.yaml` configuration files provided in this repository to the `configs` folder inside the SALMONN directory.

3. **Replace `salmonn.py`**  
   Replace the `salmonn.py` file in `SALMONN/models` with the version provided in this repository.

4. **Add script files**  
   - Move `salmonn_vanilla.py` to the root of the SALMONN directory.  
   - Move `salmonn_perturb_sampling.py` to the root of the SALMONN directory.

---

## Calculating the FESTA AUC
The outputs have already been saved in `vanilla_output/` and `perturb_sampling_output/`
To obtain the AUC score:

```bash
python kl_div_fusion.py \
  --task count \
  --model qwen
```

