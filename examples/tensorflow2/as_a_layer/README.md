# TensorFlow2 simulator as a layer
The error simulator has been developed as a custom layer for TensorFlow2.
Unfortunately due to how TensorFlow works we are not able to execute any python code after
the graph of the model has been built. For this reason if we want to use the framework as a layer we need to create
all the injection sites before the model is created.

## How to use
To set up the simulator correctly we invoke the function [`create_injection_sites_layer_simulator`](../../../src/injection_sites_generator.py)
that takes the following parameters
1. num_requested_injection_sites: an integer representing the number of injection sites we want to create
2. layer_type: an [OperatorType](https://github.com/D4De/classes/blob/c0bd0446c8d97ae13a3d5fbffed6b392cd368e19/src/injection_sites_generator.py) element defining what kind of layer we are targeting
3. layer_output_shape_cf: a string defining the targeted layer output shape in the format (None, channels, width, height) 
4. layer_output_shape_cl: a string defining the targeted layer output shape in the format (None, width, height, channels)

The function returns two lists, `available_injection_sites` and `masks`. 
The ErrorSimulator layer takes these two lists as parameters and a third parameter which is the length of `available_injection_sites`.
As any custom layer it can be inserted in a model, it should be placed after the targeted layer. 
When the layer is correctly placed in the model to execute the error simulation the user must simply invoke the `predict` function of the model itself.

## Example
A fully documented example that shows how to use the ErrorSimulator can be found in [`linear.py`](linear.py)