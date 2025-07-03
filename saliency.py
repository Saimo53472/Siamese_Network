from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import AUC
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from dataset import load_datasets, load_and_merge_datasets
from sklearn.metrics import accuracy_score, roc_auc_score
from callback import DistanceMetricsCallback
from plot import plot_validation_metrics
from metrics import evaluate_model_on_dataset, summarize_test_performance
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize

import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import cv2
import os

INPUT_SHAPE = (256, 256, 3)

def load_and_preprocess_image(path):
    img = load_img(path)
    img = img_to_array(img) / 255.0 
    img = resize(img, INPUT_SHAPE[:2])
    return img.numpy()

# Compute the saliency map by taking the gradient of output wrt input image
def compute_saliency_map(model, img1, img2, input_index=0):
    img1 = tf.convert_to_tensor(np.expand_dims(img1, axis=0), dtype=tf.float32)
    img2 = tf.convert_to_tensor(np.expand_dims(img2, axis=0), dtype=tf.float32)

    img1 = tf.Variable(img1)  # Ensure gradient flow
    img2 = tf.Variable(img2)

    with tf.GradientTape() as tape:
        if input_index == 0:
            tape.watch(img1)
        else:
            tape.watch(img2)

        distance = model([img1, img2], training=False)
    
    gradients = tape.gradient(distance, img1 if input_index == 0 else img2)[0]  # [H, W, C]

    saliency = np.max(np.abs(gradients), axis=-1)  # Take max over channels
    saliency = cv2.resize(saliency, (img1.shape[2], img1.shape[1]))
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)  # Normalize

    return saliency

def save_saliency_on_image(image, saliency, save_path, alpha=0.6, cmap='jet'):
    heatmap = matplotlib.colormaps[cmap](saliency)
    heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel

    overlay = image.astype(np.float32) / 255.0 * (1 - alpha) + heatmap[..., :3] * alpha
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Contrastive loss with margin (1 for similar, 0 for dissimilar)
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return y_true * (1/2) * square_pred + (1 - y_true) * (1/2) * margin_square
    return contrastive_loss

def get_cnn_block(filters):
    return Sequential([
        Conv2D(filters, 3, padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.1)
    ])

# Compute Euclidean distance between two vectors
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Output shape for Lambda layer computing Euclidean distance
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# Build Siamese model using CNN blocks and Euclidean distance
def build_model(input_shape, margin=1):
    DEPTH = 64

    cnn = Sequential([
        get_cnn_block(DEPTH),
        get_cnn_block(DEPTH * 2),
        get_cnn_block(DEPTH * 4),
        get_cnn_block(DEPTH * 8),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu')
    ])

    img_A_inp = Input(shape=input_shape)
    img_B_inp = Input(shape=input_shape)

    feature_vector_A = cnn(img_A_inp)
    feature_vector_B = cnn(img_B_inp)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feature_vector_A, feature_vector_B])

    model = Model(inputs=[img_A_inp, img_B_inp], outputs=distance)
    return model

# Main training, evaluation, and saliency visualization function
def main():
    input_shape = (256, 256, 3)
    model = build_model(input_shape=input_shape, margin=1)

    model.compile(
        loss=contrastive_loss_with_margin(margin=1),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )

    train_ds, val_ds, tst_ds = load_datasets("manuscript_images2")
    metrics_callback = DistanceMetricsCallback(val_data=val_ds, threshold=0.5)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=25,
                        batch_size=32,
                        callbacks=[metrics_callback],
                        verbose=2)

    summarize_test_performance(model, tst_ds, threshold=0.5)

    page1_name = "manuscript_images2/page_186.jpg"
    page2_name = "manuscript_images2/page_188.jpg"
   
    img1 = load_and_preprocess_image(page1_name)
    img2 = load_and_preprocess_image(page2_name)

    plt.imsave("img1_seen.png", np.clip(img1, 0, 1))
    plt.imsave("img2_seen.png", np.clip(img2, 0, 1))

    saliency = compute_saliency_map(model, img1, img2, input_index=0)
    save_saliency_on_image(img1, saliency, save_path="saliency_img1.png")
    saliency2 = compute_saliency_map(model, img2, img1, input_index=0)
    save_saliency_on_image(img2, saliency2, save_path="saliency_img2.png")


if __name__ == "__main__":
    main()