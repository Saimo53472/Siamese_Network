import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import pandas as pd
from voynich import load_and_preprocess_image, euclidean_distance
from saliency import compute_saliency_map, save_saliency_on_image

IMAGE_FOLDER = "voynich_images"
MODEL_PATH = "siamese_model.keras"
TOP_MATCH_CSV = "csv/best_distances.csv"
OUTPUT_DIR = "saliency_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Saliency map for the Voynich manuscript pages
def main():
    print("Loading model...")
    model = load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={"euclidean_distance": euclidean_distance}
    )

    df = pd.read_csv(TOP_MATCH_CSV)

    for _, row in df.iterrows():
        img1_name = row["filename"]
        top_match_name = row["best_match_filename"]

        img1_path = os.path.join(IMAGE_FOLDER, img1_name)
        img2_path = os.path.join(IMAGE_FOLDER, top_match_name)

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"Missing file: {img1_name} or {top_match_name}")
            continue

        img1 = load_and_preprocess_image(img1_path)
        img2 = load_and_preprocess_image(img2_path)

        try:
            resized_img_path = os.path.join(OUTPUT_DIR, os.path.splitext(img1_name)[0] + "_resized.png")
            plt.imsave(resized_img_path, np.clip(img1, 0, 1))

            saliency = compute_saliency_map(model, img1, img2, input_index=0)

            base_name = os.path.splitext(img1_name)[0]
            saliency_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_saliency.png")
            save_saliency_on_image(img1, saliency, save_path=saliency_save_path)

            print(f"Saved: {saliency_save_path} (match: {top_match_name})")

        except Exception as e:
            print(f"Error processing {img1_name}: {e}")

if __name__ == "__main__":
    main()
