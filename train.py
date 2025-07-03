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

# Contrastive loss with margin function:
# This returns a custom loss function used for training Siamese networks.
# The loss encourages pairs with label 1 to be close, and pairs with label 0 to be at least 'margin' apart.
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return y_true * (1/2) * square_pred + (1 - y_true) * (1/2) * margin_square
    return contrastive_loss

# Function to create a CNN block:
def get_cnn_block(filters):
    return Sequential([
        Conv2D(filters, 3, padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.1)
    ])

# Computes Euclidean distance between two feature vectors.
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Defines the output shape of the Lambda layer computing Euclidean distance.
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# Builds the Siamese network model.
# Two inputs go through shared CNN blocks, and their embeddings are compared via Euclidean distance.
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

# Main function for training and evaluating the Siamese network.
# Loads multiple datasets, merges them, trains the model, saves it, and plots metrics.
def main():
    input_shape = (256, 256, 3)
    model = build_model(input_shape=input_shape, margin=1)

    model.compile(
        loss=contrastive_loss_with_margin(margin=1),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )

    base_dirs = ["augmented_full", "manuscript_images2", "augmented_full3"]
    train_ds, val_ds, tst_ds = load_and_merge_datasets(base_dirs, output_prefix="augmented_combo")
    metrics_callback1 = DistanceMetricsCallback(val_data=val_ds, threshold=0.5)
    metrics_callback2 = DistanceMetricsCallback(val_data=train_ds, threshold=0.5)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=25,
                        batch_size=32,
                        callbacks=[metrics_callback1, metrics_callback2],
                        verbose=2)

    model.save("siamese_model.keras")

    summarize_test_performance(model, tst_ds, threshold=0.5)
    plot_validation_metrics(metrics_callback1.history, save_path="validation_metrics.png")
    plot_validation_metrics(metrics_callback2.history, save_path="train_metrics.png")

if __name__ == "__main__":
    main()