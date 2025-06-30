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
    chosen_img1_name = "page_188_cropped.jpg"

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

    top_match = top_matches.get(chosen_img1_name, [])[0]
    if not top_match:
        raise ValueError(f"No top matches found for {chosen_img1_name}")
    chosen_img2_name = top_match[0]

    img1 = load_and_preprocess_image(os.path.join(IMAGE_FOLDER, chosen_img1_name))
    img2 = load_and_preprocess_image(os.path.join(IMAGE_FOLDER, chosen_img2_name))

    # Save the original image
    plt.imsave("voynich_img1_seen.png", np.clip(img1, 0, 1))

    print("Computing saliency map...")
    saliency = compute_saliency_map(model, img1, img2, input_index=0)
    save_saliency_on_image(img1, saliency, save_path="voynich_saliency_img1.png")
    print("Saliency saved to voynich_saliency_img1.png")

if __name__ == "__main__":
    main()
