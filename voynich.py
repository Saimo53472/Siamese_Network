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

def get_quire_and_leaf(page_num):
    # We only process even pages (versos): 4, 6, ..., 112
    if page_num >= 26:
        leaf_index = ((page_num - 4) // 2) + 1   # Skip missing folio 12 (pages 23–24)
    else:
        leaf_index = (page_num - 4) // 2

    quire = leaf_index // 8
    leaf_in_quire = (leaf_index % 8) + 1
    return quire, leaf_in_quire

def is_valid_comparison(page_i, page_j):
    quire_i, leaf_i = get_quire_and_leaf(page_i)
    quire_j, leaf_j = get_quire_and_leaf(page_j)
    ok = 0

    if page_i == page_j:
        return False, ok
    
    if quire_i == quire_j:
        # Conjoint leaves: (1,8), (2,7), (3,6), (4,5)
        if (leaf_i, leaf_j) in [(1,8), (2,7), (3,6), (4,5)]:
            ok = 1
            return True, ok

    if (leaf_i <= 4 and leaf_j <= 4) or (leaf_i > 4 and leaf_j > 4):
        return True, ok

    return False, ok

def compute_all_distances(model, images, filenames):
    num_images = len(images)
    distances = np.full((num_images, num_images), np.nan)  # Use NaN for invalid

    page_numbers = [extract_page_number(f) for f in filenames]

    for i in tqdm(range(num_images), desc="Computing distances"):
        for j in range(i + 1, num_images):
            page_i = page_numbers[i]
            page_j = page_numbers[j]

            v, ok = is_valid_comparison(page_i, page_j)

            if not v:
                continue

            img1 = np.expand_dims(images[i], axis=0)
            img2 = np.expand_dims(images[j], axis=0)
            dist = model.predict([img1, img2])[0][0]
            distances[i, j] = dist
            if(ok == 0):
                distances[j, i] = dist

    return distances

def save_distance_matrix_csv(distances, filenames, output_path="csv/distance_matrix.csv"):
    df = pd.DataFrame(distances, index=filenames, columns=filenames)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved distance matrix to {output_path}")

def save_best_distances_per_page(distances, filenames, output_path="csv/best_distances.csv"):
    best_distances = []
    best_matches = []

    for i, fname in enumerate(filenames):
        valid_indices = np.where(~np.isnan(distances[i, :]))[0]
        if len(valid_indices) == 0:
            best_dist = np.nan
            best_match = None
        else:
            # Find the index of the minimum distance among valid distances
            min_idx_in_valid = np.argmin(distances[i, valid_indices])
            best_idx = valid_indices[min_idx_in_valid]
            best_dist = distances[i, best_idx]
            best_match = filenames[best_idx]

        best_distances.append(best_dist)
        best_matches.append(best_match)

    df_best = pd.DataFrame({
        'filename': filenames,
        'best_match_filename': best_matches,
        'best_distance': best_distances
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_best.to_csv(output_path, index=False)
    print(f"Saved best distances and matches per page to {output_path}")

def main():
    print("Loading model and images...")
    model = load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={'euclidean_distance': euclidean_distance}
    )
    images, filenames = load_images_from_folder(IMAGE_FOLDER)

     # Filter images to keep only even-numbered pages
    filtered_images = []
    filtered_filenames = []

    for img, fname in zip(images, filenames):
        page_num = extract_page_number(fname)
        if 4 <= page_num <= 112 and page_num % 2 == 0:  
            filtered_images.append(img)
            filtered_filenames.append(fname)

    distances = compute_all_distances(model, filtered_images, filtered_filenames)
    save_distance_matrix_csv(distances, filtered_filenames)
    save_best_distances_per_page(distances, filtered_filenames)

if __name__ == "__main__":
    main()