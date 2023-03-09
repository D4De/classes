
Here we use Keras backend functions to split the execution of the model into two submodels. While this approach is trivial in case of linear models particular attention must be given to models with skip connections. As an example we show a small U-Net model ![](unet.png) <br>
If we want to inject an error into the third convolutional layer we need to store the output of the second convolution in order to concate it and produce the final output.
Our injection function will thus be like this 
```python
get_selected_layer_output = K.function([model.layers[0].input], [model.layers[selected_layer_idx].output])
get_conv2_output = K.function([model.layers[0].input], [model.layers[conv2_idx].output])
get_input_concatenate = K.function([model.layers[selected_layer_idx + 1].input], [model.layers[conv2d_transpose_idx].output])
get_final_output = K.function([model.layers[concatenate_idx].input], [model.layers[-1].output])
```
The functions produce the following results

1. `get_selected_layer_output`: produces the output of the selected layer
2. `get_conv2_output`: produces the output of the second convolutional layer, the ones that must be given to the concatenate layer
3. `get_input_concatenate`: produces the output of the conv2d_transpose layer given the corrupted output of our target convolutional layer
4. `get_final_output`: produces the final output of the model starting from the concatenate layer

This process is clearly time consuming and does not easily scale with very complex models, for this reason we suggest to use this version of the error simulator with linear models and models with few skip connections while preferring the layer adaptation of the framework for more complex models.