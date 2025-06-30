# saliency_runner.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from voynich import (
    load_and_preprocess_image,
    euclidean_distance,
    get_top_matches,
    load_images_from_folder
)
from saliency import compute_saliency_map, save_saliency_on_image

IMAGE_FOLDER = "cropped_voynich_images"
MODEL_PATH = "siamese_model.keras"
INPUT_SHAPE = (256, 256, 3)
TOP_MATCH_CSV = "csv/top_matches_per_page.csv"

def load_top_matches(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    top_matches = {}
    for _, row in df.iterrows():
        fname = row["Image"]
        if fname not in top_matches:
            top_matches[fname] = []
        top_matches[fname].append((row["Match"], row["Distance"]))
    return top_matches

def main():
    output_dir = "saliency_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={"euclidean_distance": euclidean_distance}
    )

    print("Loading images and top matches...")
    images, filenames = load_images_from_folder(IMAGE_FOLDER)
    top_matches = load_top_matches(TOP_MATCH_CSV)

    filename_to_image = {fname: img for fname, img in zip(filenames, images)}

    for img1_name, matches in top_matches.items():
        if not matches:
            continue
        top_match_name = matches[0][0]

        img1 = filename_to_image.get(img1_name)
        img2 = filename_to_image.get(top_match_name)

        if img1 is None or img2 is None:
            print(f"[!] Missing image for {img1_name} or {top_match_name}")
            continue

        try:
            saliency = compute_saliency_map(model, img1, img2, input_index=0)

            base_name = os.path.splitext(img1_name)[0]
            save_path = os.path.join(output_dir, f"{base_name}_saliency.png")
            save_saliency_on_image(img1, saliency, save_path=save_path)

            print(f"Saved saliency for {img1_name} → {top_match_name} to {save_path}")

        except Exception as e:
            print(f"Error processing {img1_name}: {e}")

if __name__ == "__main__":
    main()
