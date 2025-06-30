import os
import random
import csv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import re
from torchvision.transforms import functional as TF

input_dir = "manuscript_images2"
OUTPUT_DIR = 'augmented_full2'

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def get_augmentation_pipeline():
    return [
        lambda x: TF.resize(x, (256, 256)),  # Base resize (consistent size)
        lambda x: TF.adjust_contrast(x, 1.5),  # Boost contrast to enhance texture
        lambda x: TF.adjust_brightness(x, 0.8),  # Slightly darken for stain visibility
        lambda x: TF.adjust_saturation(x, 0.5),  # Reduce color noise, emphasize tone differences
        lambda x: TF.gaussian_blur(x, kernel_size=3, sigma=1),  # Simulate scan smudge, stain persistence
    ]

def save_image(img, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    img.save(path)
    return path

def create_augmented_images(image_files, output_dir, num_transforms=5):
    transforms_list = get_augmentation_pipeline()

    for i in tqdm(range(len(image_files) - 2)):
        imgA_path = image_files[i]
        imgB_path = image_files[i + 2]

        imgA = Image.open(imgA_path).convert('RGB')
        imgB = Image.open(imgB_path).convert('RGB')

        baseA = os.path.basename(imgA_path).split('.')[0]
        baseB = os.path.basename(imgB_path).split('.')[0]

        for j in range(num_transforms):
            transform = transforms_list[j]

            augA = transform(imgA)
            augB = transform(imgB)

            nameA = f"{baseA}_t{j}.png"
            nameB = f"{baseB}_t{j}.png"

            save_image(augA, nameA, output_dir)
            save_image(augB, nameB, output_dir)


def get_page_number(filename):
    match = re.search(r'(\d+)', filename)  # \d: find a digit. +: an any other digit after that one
    return int(match.group(1)) if match else -1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_pairs = []

    image_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
    ], key=lambda x: get_page_number(os.path.basename(x)))

    create_augmented_images(image_files, OUTPUT_DIR)


if __name__ == "__main__":
    main()
