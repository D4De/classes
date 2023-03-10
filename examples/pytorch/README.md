# PyTorch simulator as a layer

The error simulator for PyTorch has been developed as a custom layer that can 
be used in a model as any other layer.
## How to use
The user has to place the simulator after the targeted layer, passing the following
two parameters
1. layer_type: defines the type of layer we are injecting, it must be a element from the OperatorType enum
2. size: the output shape of the target layer, must be a string with the following format '(None, channels, width, height)'

After having inserted the layer in the model the user must manually set the weights of the other layers. 
Then whenever we perform inference with the model the error simulator will automatically generate a valid
set of injection sites and perform an error simulation. <br>
## Example
A commented example on how to use the error simulator can be found in [lenet.py](lenet.py)