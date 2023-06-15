import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, losses
from keras import backend as K

import os
import sys

CLASSES_MODULE_PATH = "../../../"
WEIGHT_FILE_PATH = "../"
MODELS_PATH = CLASSES_MODULE_PATH + "models"

# appending a path
sys.path.append(CLASSES_MODULE_PATH)  # CHANGE THIS LINE

from src.injection_sites_generator import *


### PERFORMING INJECTIONS ON A LINEAR MODEL
#
# To correctly perform injections on a linear model we first have to select the layer we want to inject.
# We can iterate through model.layers to find the corresponding index and store it (line 109)
# Then we define the number of tests that we want to perform (line 107)
# and execute the function inject_layer each time (lines 115 - 116)
# We need to pass the following parameters
# 		1. model: the model we are targeting
#		2. img: an image on which we will perform inference
#		3. selected layer index: the index referring to the layer we selected for the injection
#		4. operator type: a value from the OperatorType enum that defines the type of the layer we are injecting
#		5. shape: the output shape of the layer as a string in the following format: (None, channels, widht, height)
#       6. models_path: the folder in src/ where we placed the error models, it can be specified if the user does not
#          want to use the defaults one.
#
# The function will return the corrupted output of the model

def generate_injection_sites(sites_count, layer_type, size, models_path):
    injection_site = InjectableSite(layer_type, size)

    injection_sites = InjectionSitesGenerator([injection_site], models_path) \
            .generate_random_injection_sites(sites_count)

    return injection_sites


def build_model(input_shape, saved_weights=None):
    """
    Saved weights should be the path to a h5 file if you have already trained the model
    """
    if saved_weights is not None:
        model = keras.models.load_model(saved_weights)
        return model
    inputs = keras.Input(shape=input_shape, name='input')
    conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', name='conv1')(inputs)
    pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='maxpool1')(conv1)
    conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same",
                          name='conv2')(pool1)
    pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='maxpool2')(conv2)
    conv3 = layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same",
                          name='conv3')(pool2)
    flatten = layers.Flatten(name='flatten')(conv3)
    dense1 = layers.Dense(84, activation='relu', name='dense1')(flatten)
    outputs = layers.Dense(10, activation='softmax', name='dense3')(dense1)

    return keras.Model(inputs=(inputs,), outputs=outputs)


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)

    x_val = x_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val


def inject_layer(model, img, selected_layer_idx, layer_type, layer_output_shape_cf, models_path='models'):
    get_selected_layer_output = K.function([model.layers[0].input], [model.layers[selected_layer_idx].output])
    get_model_output = K.function([model.layers[selected_layer_idx + 1].input], [model.layers[-1].output])

    output_selected_layer = get_selected_layer_output([np.expand_dims(img, 0)])[0]
    range_min = np.min(output_selected_layer)
    range_max = np.max(output_selected_layer)

    injection_site = generate_injection_sites(1, layer_type, layer_output_shape_cf, models_path)

    if len(injection_site) > 0:
        for idx, value in injection_site[0].get_indexes_values():
            channel_last_idx = (idx[0], idx[2], idx[3], idx[1])
            output_selected_layer[channel_last_idx] = value.get_value(range_min, range_max)

    model_output = get_model_output(output_selected_layer)

    return model_output


NUM_INJECTIONS = 100
NUM = 42
SELECTED_LAYER_IDX = 3

x_train, y_train, x_val, y_val = load_data()
path_weights = os.path.join(WEIGHT_FILE_PATH, 'weights.h5')
print(f"Load weights from => {path_weights}")
model = build_model(x_train[0].shape, saved_weights=path_weights)
errors = 0


for _ in range(NUM_INJECTIONS):
    res = inject_layer(model, x_val[NUM], SELECTED_LAYER_IDX, 'conv_gemm', '(None, 16, 27, 27)',
                       models_path=MODELS_PATH)
    if np.argmax(res) != y_val[NUM]:
        errors += 1

print(f'Number of misclassification over {NUM_INJECTIONS} injections: {errors}')
