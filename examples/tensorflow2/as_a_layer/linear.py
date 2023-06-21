import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, losses
from keras import backend as K
import numpy as np

import os
import sys 

CLASSES_MODULE_PATH = "../../../"
WEIGHT_FILE_PATH = "../"
MODELS_PATH = CLASSES_MODULE_PATH + "models"

# appending a path
sys.path.append(CLASSES_MODULE_PATH) #CHANGE THIS LINE

from src.injection_sites_generator import *
from src.error_simulator_keras import ErrorSimulator, create_injection_sites_layer_simulator


### PERFORMING INJECTIONS WITH CUSTOM LAYER
#
# To correctly perform injections using the error simulator layer we first have to select the layer we want to inject.
# When we selected the layer we insert the error simulator into the model (line 57)
# Then we define the number of tests that we want to perform (line 81)
# Then we generate a number of injection sites equal or greater than the number of tests we want to perform by invoking
# the function create_injection_sites_layer_simulator. We then build the model with the simulator by passing
# the following three parameters:
#   1. available_injection_sites: all the available injection sites
#   2. masks: masks used by the injector
#   3. len(available_injection_sites): the number of available injection sites
#
# At lines 100 - 103 we manually copy the weights from a pre-trained model without the simulator into the modified one
# with the simulator. We then perform the error simulation campaign by simply invoking the predict function of the
# modified model


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


def build_model_with_simulator(input_shape, available_injection_sites, masks, num_inj_sites):
    inputs = keras.Input(shape=input_shape, name='input')
    conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', name='conv1')(inputs)
    pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='maxpool1')(conv1)
    conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same",
                          name='conv2')(pool1)
    simulator = ErrorSimulator(available_injection_sites, masks, num_inj_sites)(conv2)
    pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='maxpool2')(simulator)
    conv3 = layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same",
                          name='conv3')(pool2)
    flatten = layers.Flatten(name='flatten')(conv3)
    dense1 = layers.Dense(84, activation='relu', name='dense1')(flatten)
    outputs = layers.Dense(10, activation='softmax', name='dense3')(dense1)

    return keras.Model(inputs=(inputs,), outputs=outputs)


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)

    x_val = x_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val


NUM_INJECTIONS = 100
NUM = 42

layer_type = 'conv_gemm'
layer_output_shape_cf = '(None, 16, 27, 27)'
layer_output_shape_cl = '(None, 27, 27, 16)'

num_requested_injection_sites = NUM_INJECTIONS * 5

available_injection_sites, masks = create_injection_sites_layer_simulator(num_requested_injection_sites,
                                                                          layer_type,
                                                                          layer_output_shape_cf, layer_output_shape_cl,
                                                                          models_folder=MODELS_PATH)

x_train, y_train, x_val, y_val = load_data()
path_weights = os.path.join(WEIGHT_FILE_PATH,'weights.h5')
print(f"Load weights from => {path_weights}")
model = build_model(x_train[0].shape, saved_weights=path_weights)
model_with_simulator = build_model_with_simulator(x_train[0].shape, available_injection_sites, masks,
                                                  len(available_injection_sites))

injector_model_layers = [layer.name for layer in model_with_simulator.layers]
model_base_layers = [layer.name for layer in model.layers]
for layer_name in model_base_layers:
    model_with_simulator.get_layer(layer_name).set_weights(model.get_layer(layer_name).get_weights())

errors = 0

for _ in range(NUM_INJECTIONS):
    res = model_with_simulator.predict(np.expand_dims(x_val[NUM], 0))
    if np.argmax(res) != y_val[NUM]:
        errors += 1

print(f'Number of misclassification over {NUM_INJECTIONS} injections: {errors}')
