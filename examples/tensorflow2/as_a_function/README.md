
Here we use Keras backend functions to split the execution of the model into two submodels. While this approach is trivial in case of linear models particular attention must be given to models with skip connections. 
## U-Net

As an example we show a small U-Net model ![](unet.png) <br>
If we want to inject an error into the third convolutional layer we need to do the following
1. Execute the model from the input layer through the targeted convolutional layer
2. Modify the output of the convolutional layer based on the selected injection sites
3. Execute the model from the convolutional layer through the concatenate layer
4. Calculate the second input of the concatenate layer
5. Execute the model from the concatenate layer through the output layer

To do so we need to use the following functions>
```python
get_selected_layer_output = K.function([model.layers[0].input], [model.layers[selected_layer_idx].output])
get_conv2_output = K.function([model.layers[0].input], [model.layers[conv2_idx].output])
get_input_concatenate = K.function([model.layers[selected_layer_idx + 1].input], [model.layers[conv2d_transpose_idx].output])
get_final_output = K.function([model.layers[concatenate_idx].input], [model.layers[-1].output])
```
They respectively produce the following results

1. `get_selected_layer_output`: output of the selected layer
2. `get_conv2_output`: output of the second convolutional layer, the first input of the concatenate layer
3. `get_input_concatenate`: output of the conv2d_transpose layer, second input of the concatenate layer
4. `get_final_output`:  final output of the model starting from the concatenate layer

## Linear model
If we have a model with no skip connections ![](linear.png)
the process is straightforward. 
As an example let's suppose we want to target the second convolutional layer, the only functions we need are the following 
```python
get_selected_layer_output = K.function([model.layers[0].input], [model.layers[selected_layer_idx].output])
get_output_injected = K.function([model.layers[selected_layer_idx + 1].input], [model.layers[-1].output])
```
The first one produces the output of the selected layer, we can then corrupt it and pass it to the second function that produces the final output of the model.

## Examples
`lenet.py` and `small_unet.py` are fully documented examples.