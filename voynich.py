import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from saliency import compute_saliency_map, save_saliency_on_image

IMAGE_FOLDER = "cropped_voynich_images"
MODEL_PATH = "siamese_model.keras"
INPUT_SHAPE = (256, 256, 3)
THRESHOLD = 0.5

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def extract_page_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_and_preprocess_image(path):
    img = load_img(path)
    img = img_to_array(img) / 255.0 
    img = resize(img, INPUT_SHAPE[:2])
    return img.numpy()

def load_images_from_folder(folder):
    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ], key=extract_page_number)

    images = [load_and_preprocess_image(os.path.join(folder, fname)) for fname in image_files]
    return images, image_files

import random

def compute_all_distances(model, images, filenames):
    num_images = len(images)
    distances = np.zeros((num_images, num_images))

    for i in tqdm(range(num_images), desc="Computing distances"):
        for j in range(i + 1, num_images):
            img1 = np.expand_dims(images[i], axis=0)
            img2 = np.expand_dims(images[j], axis=0)
            dist = model.predict([img1, img2])[0][0]  # Assuming output shape is (1, 1)
            distances[i, j] = dist
            distances[j, i] = dist  # symmetric

    return distances

def save_distance_matrix_csv(distances, filenames, output_path="csv/distance_matrix.csv"):
    df = pd.DataFrame(distances, index=filenames, columns=filenames)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved distance matrix to {output_path}")


def get_top_matches(distances, filenames, top_k=3):
    results = {}
    for i, fname in enumerate(filenames):
        dists = distances[i]
        sorted_indices = np.argsort(dists)
        # Skip self-match at index 0
        top_matches = [(filenames[j], dists[j]) for j in sorted_indices if j != i][:top_k]
        results[fname] = top_matches
    return results


def save_top_matches(results, output_path="csv/top_matches_per_page.csv"):
    rows = []
    for fname, matches in results.items():
        for rank, (match_name, dist) in enumerate(matches, 1):
            rows.append({
                'Image': fname,
                f'Rank': rank,
                'Match': match_name,
                'Distance': dist
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[✓] Saved top matches to {output_path}")

def main():
    print("Loading model and images...")
    model = load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={'euclidean_distance': euclidean_distance}
    )
    images, filenames = load_images_from_folder(IMAGE_FOLDER)

    distances = compute_all_distances(model, images, filenames)
    save_distance_matrix_csv(distances, filenames)

    top_matches = get_top_matches(distances, filenames, top_k=3)
    save_top_matches(top_matches)

if __name__ == "__main__":
    main()