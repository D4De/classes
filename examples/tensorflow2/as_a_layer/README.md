# TensorFlow2 simulator as a layer
The error simulator has been developed as a custom layer for TensorFlow2.
Unfortunately due to how TensorFlow works we are not able to execute any python code after
the graph of the model has been built. For this reason if we want to use the framework as a layer we need to create
all the injection sites before the model is created.

## How to use
The steps required to setup the simulator are the following 
1. Create a set of injection sites using the function [`create_injection_sites_layer_simulator`](../../../src/error_simulator_keras.py) defined in [`error_simulator_keras.py`](../../../src/error_simulator_keras.py) that takes the following parameters:
    - num_requested_injection_sites: an integer representing the number of injection sites we want to create
    - layer_type: an [OperatorType](../../../src/operators.py) element defining what kind of layer we are targeting
    - layer_output_shape_cf: a string defining the targeted layer output shape in the format (None, channels, width, height)
    - layer_output_shape_cl: a string defining the targeted layer output shape in the format (None, width, height, channels)
2. Define a new model that will be a copy of the one we are targeting with the addition of the simulator layer
3. Instantiate the model and provide it the injection sites
4. Copy the weights from the original model to the new one
5. Invoke the `predict` function to perform the simulation

To understand the steps let's see an example, we start from a simple model 
```python
def build_model(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape, name='input')
    conv = layers.Conv2D(filters=..., kernel_size=..., activation='relu', name='conv')(inputs)
    pool = layers.MaxPool2D(pool_size=..., strides=..., name='maxpool')(conv)
    flatten = layers.Flatten(name='flatten')(pool)
    dense = layers.Dense(..., activation='relu', name='dense')(flatten)
    outputs = layers.Dense(n_classes, activation='softmax', name='dense')(dense)
    return keras.Model(inputs=(inputs,), outputs=outputs)
```
We also define the new model with the simulator 
```python
def build_model_with_simulator(input_shape, n_classes, available_injection_sites, masks, num_inj_sites):
    inputs = keras.Input(shape=input_shape, name='input')
    conv = layers.Conv2D(filters=..., kernel_size=..., activation='relu', name='conv')(inputs)
    simulator = ErrorSimulator(available_injection_sites, masks, num_inj_sites)(conv)
    pool = layers.MaxPool2D(pool_size=..., strides=..., name='maxpool')(simulator[0])
    flatten = layers.Flatten(name='flatten')(pool)
    dense = layers.Dense(..., activation='relu', name='dense')(flatten)
    outputs = layers.Dense(n_classes, activation='softmax', name='dense')(dense)
    return keras.Model(inputs=(inputs,), outputs=outputs)
```
Having done this we can generate the injection sites and instantiate the model
```python
# We define how many injections to perform
NUM_INJECTIONS = 100
# We set the parameters of the layer that will be targeted
layer_type = OperatorType['Conv2D']
layer_output_shape_cf = '(None, channels, width, height)'
layer_output_shape_cl = '(None, width, height, channels)'
num_requested_injection_sites = NUM_INJECTIONS * 5

# We create the injection sites
available_injection_sites, masks = create_injection_sites_layer_simulator(num_requested_injection_sites,
                                                                          layer_type,
                                                                          layer_output_shape_cf, layer_output_shape_cl)
# We build both models
model = keras.models.load_model('weights.h5')
model_with_simulator = build_model_with_simulator(input_shape, available_injection_sites, masks,
                                                  len(available_injection_sites))

# Finally we copy the weights from the original model to the new one and we are ready to perform the simulation
injector_model_layers = [layer.name for layer in model_with_simulator.layers]
model_base_layers = [layer.name for layer in model.layers]
for layer_name in model_base_layers:
    model_with_simulator.get_layer(layer_name).set_weights(model.get_layer(layer_name).get_weights())
```

## Example
A complete and fully documented example that shows how to use the ErrorSimulator can be found in [`linear.py`](linear.py)