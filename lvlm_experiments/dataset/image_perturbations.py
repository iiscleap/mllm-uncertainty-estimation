import numpy as np
import cv2
import random
import csv
import os
import argparse

random.seed(42)

def increase_contrast(image, x, min_val=0, max_val=255):
    img = image.astype(np.float32)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    factor = 1 + 0.1 * x
    img = (img - mean) * factor + mean
    return np.clip(img, min_val, max_val).astype(image.dtype)

def BW(image):
    img = image.astype(np.float32)
    base_weights = np.array([0.299, 0.587, 0.114])
    noise = np.random.normal(0, 0.05, 3)
    weights = np.clip(base_weights + noise, 0, None)
    weights /= weights.sum()
    gray = np.dot(img[..., :3], weights)
    gray3 = np.repeat(gray[..., np.newaxis], 3, axis=2)
    return np.clip(gray3, 0, 255).astype(image.dtype)

def gamma_transform(image, x, min_val=0, max_val=255):
    img = np.clip(image.astype(np.float32), min_val, max_val)
    gamma = 0.1 * x
    img_norm = (img - min_val) / (max_val - min_val)
    img_gamma = np.power(img_norm, gamma)
    img = img_gamma * (max_val - min_val) + min_val
    return np.clip(img, min_val, max_val).astype(image.dtype)

def blur(image, x):
    return image.copy() if x < 2 else cv2.blur(image, (x, 1))

def add_gaussian_noise(image, std, min_val=0, max_val=255):
    img = image.astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=std, size=img.shape)
    return np.clip(img + noise, min_val, max_val).astype(image.dtype)

def masking(image, x, min_val=0, max_val=255):
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]
    num_pixels = h * w
    num_mask = int(0.01 * x * num_pixels)
    idx = np.random.permutation(num_pixels)[:num_mask]
    ys, xs = np.unravel_index(idx, (h, w))
    mid_val = (max_val - min_val) / 2
    img[ys, xs] = mid_val if img.ndim == 2 else np.repeat(mid_val, img.shape[2])
    return np.clip(img, min_val, max_val).astype(image.dtype)

def shift_image(image, shift_ratio=5):
    shift_ratio *= 0.01
    h, w = image.shape[:2]
    tx = int(shift_ratio * w)
    ty = int(shift_ratio * h)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def rotate_image(image, angle=5):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def get_qns_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['question']
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['vsr', 'blink'], required=True)
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--input_image_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--num_perturbations', type=int, default=2,
                        help='Number of perturbations per transformation')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    indices = []
    if args.dataset == 'vsr':
        indices = [f"val_Spatial_Reasoning_{i}" for i in range(111, 211)]
    else:
        indices = [f"val_Spatial_Relation_{i}" for i in range(1, 144)]

    with open(args.output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'orig_idx', 'question', 'transformation_type', 'intensity'])

        for idx in indices:
            image_path = os.path.join(args.input_image_folder, f"{idx}.jpg")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read {image_path}")
                continue

            question = get_qns_for_idx(args.input_csv, idx)

            contrast_blur_rotate_shift_vals = random.sample(range(3, 9), args.num_perturbations)
            noise_masking_vals = random.sample(range(5, 11), args.num_perturbations)

            transform_params = {}
            transformed_images = {}

            for i, val in enumerate(contrast_blur_rotate_shift_vals):
                suffix = str(i + 1)
                transform_params[f'contrast{suffix}'] = val
                transformed_images[f'contrast{suffix}'] = increase_contrast(image, val)

                transform_params[f'blur{suffix}'] = val
                transformed_images[f'blur{suffix}'] = blur(image, val)

                transform_params[f'rotate{suffix}'] = val
                transformed_images[f'rotate{suffix}'] = rotate_image(image, val)

                transform_params[f'shift{suffix}'] = val
                transformed_images[f'shift{suffix}'] = shift_image(image, val)

            for i, val in enumerate(noise_masking_vals):
                suffix = str(i + 1)
                transform_params[f'noise{suffix}'] = val
                transformed_images[f'noise{suffix}'] = add_gaussian_noise(image, val)

                transform_params[f'masking{suffix}'] = val
                transformed_images[f'masking{suffix}'] = masking(image, val)

            for i in range(args.num_perturbations):
                key = f'bw{i+1}'
                transform_params[key] = "NA"
                transformed_images[key] = BW(image)

            for name, out_img in transformed_images.items():
                new_idx = f"{idx}_{name}"
                out_path = os.path.join(args.output_folder, f"{new_idx}.jpg")
                cv2.imwrite(out_path, out_img)
                print(f"Saved {out_path}")
                transform_type = ''.join(filter(str.isalpha, name))
                intensity = transform_params[name]
                writer.writerow([new_idx, idx, question, transform_type, intensity])

if __name__ == "__main__":
    main()
