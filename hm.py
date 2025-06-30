import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Input, Lambda
from tensorflow.keras.models import Sequential, Model

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return y_true * (1/2) * square_pred + (1 - y_true) * (1/2) * margin_square
    return contrastive_loss

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def main():
    # Load your model
    model = keras.models.load_model("siamese_model.keras", safe_mode=False,
    custom_objects={
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss_with_margin
    }) 

    for layer in model.layers:
        if 'sequential_4' in layer.name:
            for layer2 in layer.layers:
                if 'sequential_3' in layer2.name:
                    for layer4 in layer2.layers:
                        if 'conv2d_3' in layer4.name:
                            filters, biases = layer4.get_weights()
                            print(layer4.name, filters.shape)

if __name__ == "__main__":
    main()
