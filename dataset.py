import os
import re
import csv
import random
from PIL import Image
import numpy as np
from numpy import asarray
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Extract numeric page number
def extract_page_number(filename):
    match = re.search(r'page_(\d+)', filename)
    return int(match.group(1)) if match else None

# Extract augmentation type from filename
def extract_aug_type(filename):
    match = re.search(r'_(t\d+)', filename)
    return match.group(1) if match else None

# Create pairs of images from base_dir:
    # positive pairs = images two pages apart (page i and i+2),
    # negative pairs = randomly selected non-adjacent pages,
    # grouping by augmentation type or "plain" if none.
    # Returns list of pairs and csv rows for saving.
def make_pairs(base_dir):
    files = os.listdir(base_dir)
    grouped = {}
    for f in files:
        aug_type = extract_aug_type(f)
        page_num = extract_page_number(f)
        if page_num is not None:
            key = aug_type if aug_type else "plain"
            grouped.setdefault(key, []).append((page_num, f))
    
    pairs = []
    csv_rows = []

    for aug_type, file_list in grouped.items():
        file_list.sort() 

        for i in range(0, len(file_list) - 4):
            page_i, file_i = file_list[i]
            path_i = os.path.join(base_dir, file_i)

            page_next, file_next = file_list[i + 2]
            if page_next == page_i + 2:
                path_j = os.path.join(base_dir, file_next)
                pairs.append((path_i, path_j, 1))
                csv_rows.append((path_i, path_j, 1))

                j_candidates = list(range(i + 4, len(file_list), 2))
                if j_candidates:
                    j = random.choice(j_candidates)
                    _, file_neg = file_list[j]
                    path_neg = os.path.join(base_dir, file_neg)
                    pairs.append((path_i, path_neg, 0))
                    csv_rows.append((path_i, path_neg, 0))

    return pairs, csv_rows

def preprocess_image(path, size=(256,256)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image = image / 255.0
    return image

# Generate image pairs from base_dir, save CSV splits (train/val/test),
    # return TensorFlow datasets for each split.
def load_datasets(base_dir, save_dir="csv"):
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.normpath(base_dir))
    pairs, rows = make_pairs(base_dir)

    all_csv_path = os.path.join(save_dir, f"{dataset_name}_all_pairs.csv")
    with open(all_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img1", "img2", "label"])
        writer.writerows(rows)

    df = pd.DataFrame(pairs, columns=["img1", "img2", "label"]).sample(frac=1).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    train_df.to_csv(os.path.join(save_dir, f"{dataset_name}_train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, f"{dataset_name}_val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, f"{dataset_name}_test.csv"), index=False)

    def df_to_dataset(dataframe):
        img1_paths = dataframe["img1"].values
        img2_paths = dataframe["img2"].values
        labels = dataframe["label"].values.astype('float32')

        dataset = tf.data.Dataset.from_tensor_slices((img1_paths, img2_paths, labels))

        def load_and_preprocess(img1_path, img2_path, label):
            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            return (img1, img2), label

        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    print(f"\nSaved split CSVs to `{save_dir}` with prefix `{dataset_name}_`")
    return df_to_dataset(train_df), df_to_dataset(val_df), df_to_dataset(test_df)

# Create combined dataset from multiple directories,
# save CSV splits and return TensorFlow datasets,
# print label distributions for data splits.
def load_and_merge_datasets(base_dirs, output_prefix="combined", save_dir="csv"):
    os.makedirs(save_dir, exist_ok=True)

    all_pairs = []
    for base_dir in base_dirs:
        print(f"Processing directory: {base_dir}")
        pairs, _ = make_pairs(base_dir)
        all_pairs.extend(pairs)

    print(f"Total pairs collected from all datasets: {len(all_pairs)}")

    df = pd.DataFrame(all_pairs, columns=["img1", "img2", "label"]).sample(frac=1).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    train_df.to_csv(os.path.join(save_dir, f"{output_prefix}_train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, f"{output_prefix}_val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, f"{output_prefix}_test.csv"), index=False)

    def df_to_dataset(dataframe):
        img1_paths = dataframe["img1"].values
        img2_paths = dataframe["img2"].values
        labels = dataframe["label"].values.astype('float32')

        dataset = tf.data.Dataset.from_tensor_slices((img1_paths, img2_paths, labels))

        def load_and_preprocess(img1_path, img2_path, label):
            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            return (img1, img2), label

        return dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
                      .cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    print("\nSaved merged CSVs to:", save_dir)
    print("Train label distribution:\n", train_df['label'].value_counts(normalize=True))
    print("Validation label distribution:\n", val_df['label'].value_counts(normalize=True))
    print("Test label distribution:\n", test_df['label'].value_counts(normalize=True))

    return df_to_dataset(train_df), df_to_dataset(val_df), df_to_dataset(test_df)

