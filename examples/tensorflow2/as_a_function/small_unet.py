import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from keras import backend as K

import os
import sys 

CLASSES_MODULE_PATH = "../../../"
WEIGHT_FILE_PATH = "../"

# appending a path
sys.path.append(CLASSES_MODULE_PATH) #CHANGE THIS LINE

from src.injection_sites_generator import *


### How to target a model with skip connection
#   In this example we will show how to use classes to simulate errors
#   targeting a model with skip connections. We define a small unet
#   model with a single skip connection (function build_tiny_unet) and we will target
#   the first convolution after the skip. To do so we need to find the idx of the layer we
#   want to target by passing through the list model.layers, the index we are looking for is 5.
#   The next step consists in defining all the functions needed to perform the injection correctly,
#   this means that we need to manually provide each connection with all the inputs required.
#   Lines 140 - 144 is where we define the following functions
#       1. get_selected_layer_output: executes the model from the input layer through the selected layer,
#           produces the output of the target layer
#       2. get_conv2_output:  executes the model from the input layer through the second convolutional layer,
#           the output produced will be the first input of the concatenate layer
#       3. get_input_concatenate: executes the model from the target layer through the one before the concatenate layer,
#           the input for this function must be the corrupted output of the target layer
#       4. get_final_output: executes the model from the concatenation through the final output.
#
#   It should be noted that this code doesn't work without some modifications, we currently do not load a real dataset
#   and do not provide an evaluation function for the output which is strictly dependent on the domain of the model.

def generate_injection_sites(sites_count, layer_type, layer_name, size, models_path):
    """
    models_path: relative path form the pwd to the models folder
    """
    injection_site = InjectableSite(layer_type, layer_name, size)

    injection_sites = InjectionSitesGenerator([injection_site], models_path) \
            .generate_random_injection_sites(sites_count)

    return injection_sites


def double_conv_block(x, n_filters, nameType=None):
    x = layers.Conv2D(n_filters, 3,
                      padding="same", activation="relu",
                      kernel_initializer="he_normal",
                      name=f'conv1_{nameType}_{n_filters}')(x)
    x = layers.Conv2D(n_filters, 3,
                      padding="same", activation="relu",
                      kernel_initializer="he_normal",
                      name=f'conv2_{nameType}_{n_filters}')(x)

    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters, nameType='downsample')
    p = layers.MaxPool2D(2, name=f'maxpool_{n_filters}')(f)
    p = layers.Dropout(0.3, name=f'dropout_{n_filters}')(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same",
                               name=f'upconv_{n_filters}')(x)
    x = layers.concatenate([x, conv_features], name=f'concat_{n_filters}')
    x = layers.Dropout(0.3, name=f'dropout_upsample_{n_filters}')(x)
    x = double_conv_block(x, n_filters, nameType='upsample')

    return x


def build_tiny_unet(output_channels):
    inputs = layers.Input(shape=(128, 128, 3), name='input')

    f1, p1 = downsample_block(inputs, 32)
    bottleneck = double_conv_block(p1, 64, nameType='bottleneck')
    u1 = upsample_block(bottleneck, f1, 32)
    outputs = layers.Conv2D(output_channels, 1, padding="same", activation="softmax",
                            name='output')(u1)
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model


def load_images():
    """
    Dummy function, it should load the dataset that the model should use
    """
    return 0


def evaluate_output(val):
    """
    Dummy function, it should evaluate the obtained output based on a metric chosen by the developer
    """
    return 0


def main():
    path_weights = os.path.join(WEIGHT_FILE_PATH,'weights.h5')
    print(f"Load weights from => {path_weights}")
    model = keras.models.load_model('../tiny_unet.h5', compile=False)

    img = load_images()
    selected_layer_idx = 5
    conv2_idx = 2
    conv2d_transpose_idx = 7
    concatenate_idx = 8

    get_selected_layer_output = K.function([model.layers[0].input], [model.layers[selected_layer_idx].output])
    get_conv2_output = K.function([model.layers[0].input], [model.layers[conv2_idx].output])
    get_input_concatenate = K.function([model.layers[selected_layer_idx + 1].input],
                                       [model.layers[conv2d_transpose_idx].output])
    get_final_output = K.function([model.layers[concatenate_idx].input], [model.layers[-1].output])

    selected_layer_output = get_selected_layer_output(img)
    range_min, range_max = np.min(selected_layer_output), np.max(selected_layer_output)
    injection_site = generate_injection_sites(1, 'conv_gemm', '(None, 64, 64, 64)', models_path='models')

    if len(injection_site) > 0:
        for idx, value in injection_site[0].get_indexes_values():
            channel_last_idx = (idx[0], idx[2], idx[3], idx[1])
            selected_layer_output[channel_last_idx] = value.get_value(range_min, range_max)

    conv2_output = get_conv2_output(img)
    input_concatenate = get_input_concatenate(selected_layer_output)
    final_output = get_final_output([conv2_output, input_concatenate])

    evaluate_output(final_output)


if __name__ == '__main__':
    main()
