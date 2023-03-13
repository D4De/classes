# PyTorch simulator as a layer

The error simulator for PyTorch has been developed as a custom layer that can 
be used in a model as any other layer.
## How to use
The user has to place the simulator after the targeted layer, passing the following
two parameters
1. layer_type: defines the type of layer we are injecting, it must be a element from the [OperatorType](../../src/operators.py) enum
2. size: the output shape of the target layer, must be a string with the following format '(None, Channels, Width, Height)'

As an example of this process let's take this simple model
```python
class ExampleModel(nn.Module):
     def __init__(self, n_classes):
        super(ExampleModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=..., stride=...)
        self.pool = nn.AvgPool2d(kernel_size=...)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(in_features=..., out_features=n_classes)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.tanh(x)
        x = torch.flatten(x, 1)
        logits = self.linear1(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
        
```
If we want to target the convolutional layer we have to build a new model like this
```python
class ExampleModelSimulator(nn.Module):
     def __init__(self, n_classes, layer_type, output_shape):
        super(ExampleModelSimulator, self).__init__()
        self.conv = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=..., stride=...)
        self.simulator = Simulator(layer_type, output_shape) # We add the simulator to the model
        self.pool = nn.AvgPool2d(kernel_size=...)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(in_features=..., out_features=n_classes)
         
    def forward(self, x):
        x = self.conv(x)
        x = self.simulator(x) # We make sure that the simulator is invoked during the forward pass 
        x = self.pool(x)
        x = self.tanh(x)
        x = torch.flatten(x, 1)
        logits = self.linear1(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
        
```

After having inserted the layer in the model the user must manually set the weights of the other layers. 
Then whenever we perform inference with the model the error simulator will automatically generate a valid
set of injection sites and perform an error simulation.
To conclude the example we show how to copy the weights for the `ExampleModelSimulator` model. We firstly create the model and then 
assign the values from the original trained model.
```python
model = ExampleModel(N_CLASSES)
model.load_state_dict(torch.load('weights.pth'))
model_simulator = ExampleModelSimulator(N_CLASSES, OperatorType['Conv2D'], '(None, channels, width, height)')

for name, param in model.named_parameters():
    eval(f'model_simulator.{name.split(".")[0]}').weight = nn.Parameter(
        torch.ones_like(eval(f'model.{name.split(".")[0]}').weight))
```
## Example
A complete and commented example on how to use the error simulator can be found in [lenet.py](lenet.py)