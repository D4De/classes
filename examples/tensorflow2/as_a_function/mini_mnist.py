import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, losses
from keras import backend as K

import os
import sys 
# To make this script work, you should have working tensorflow 2 enviorment
# and you must move this script in the repository top level folder (3 folders up)
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
#
# The function will return the corrupted output of the model

def generate_injection_sites(sites_count, layer_type, size, models_path):
    injection_site = InjectableSite(layer_type, size)

    
    injection_sites = InjectionSitesGenerator([injection_site], models_path) \
            .generate_random_injection_sites(sites_count)


    return injection_sites


def build_model(input_shape, saved_weights=None):
    if saved_weights is not None:
        model = keras.models.load_model(saved_weights)
        return model
    inputs = keras.Input(shape=input_shape, name='input')
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))(inputs)
    maxpool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(maxpool1)
    maxpool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu')(maxpool2)
    flatten = layers.Flatten()(conv3)
    dense1 = layers.Dense(64, activation='relu')(flatten)
    dense2 = layers.Dense(10)(dense1)

    model = keras.Model(inputs=(inputs,), outputs=(dense2,))
    return model


def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
    return train_images / 255.0, test_images / 255.0, train_labels, test_labels


def inject_layer(model, img, selected_layer_idx, layer_type, layer_output_shape_cf, models_path):
    get_selected_layer_output = K.function([model.layers[0].input], [model.layers[selected_layer_idx].output])
    get_model_output = K.function([model.layers[selected_layer_idx + 1].input], [model.layers[-1].output])

    output_selected_layer = get_selected_layer_output([np.expand_dims(img, 0)])[0]
    range_min, range_max = np.min(output_selected_layer), np.max(output_selected_layer)

    injection_site = generate_injection_sites(1, layer_type, layer_output_shape_cf, models_path)
    
    if len(injection_site) > 0:
        print(f"Injected from: {next(injection_site[0].get_indexes_values())[0]}")
        for idx, value in injection_site[0].get_indexes_values():
            # Reorder idx if format is NHWC
            channel_last_idx = (idx[0], idx[2], idx[3], idx[1])
            output_selected_layer[channel_last_idx] = value.get_value(range_min, range_max)

    print(output_selected_layer.shape)
    model_output = get_model_output(output_selected_layer)

    return model_output


NUM_INJECTIONS = 10
NUM = 40
SELECTED_LAYER_IDX = 0


x_train, x_test, lab_train, lab_test = load_data()
print(x_train.shape)
print(x_test.shape)
path_weights = os.path.join(WEIGHT_FILE_PATH,'weights.h5')
print(f"Load weights from => {path_weights}")
model = build_model(x_train[0].shape, saved_weights=path_weights)
errors = 0

for _ in range(NUM_INJECTIONS):
    res = inject_layer(model, x_train[NUM], SELECTED_LAYER_IDX, 'conv_gemm', '(None, 3, 32, 32)',
                       models_path=MODELS_PATH)
    if np.argmax(res) != lab_train[NUM]:
        errors += 1

print(f'Number of misclassification over {NUM_INJECTIONS} injections: {errors}')
