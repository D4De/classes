# Introduction 

The tool described here has been developed in order to use CLASSES with models for which the user is only in possession of the saved weights and no information about the code used to build the model is provided.
We assume that such file is in the h5 format, if not there are multiple tools available online to convert between the most common formats (e.g., ONNX, SavedModel) to h5.

# Requirements

The `h5py` library is required for this tool to work correctly, it can be installed with the following command
```
pip install h5py
```

# Usage

The tool is composed by multiple scripts that perform different steps, the following section describes the order in which they should be executed and their functionalities. 

## Modify config file
The first step in the process is to modify the [`config.json`](config.py) to add all the info about the simulation we are going to perform. The keys of the file are the following:
1. model: path to the weights.h5 file that describes the model we are targeting.
2. new_model: path were the script will save the weights.h5 file of the new model we are creating.
2. injection_sites: an int representing the number of injections we want to perform.
3. layer_type: the name of the type of the layer we are targeting. It is required for selecting the correct error model. It should be the same as one of the json provided in the models folder without the extension (e.g, avgpool.json -> avgpool).
4. layer_output_shape_cf: a string with the output shape of the targeted layer in the format NCHW. (e.g, '(None, 16, 27, 27)').
5. layer_output_shape_cl: a string with the output shape of the targeted layer in the format NHWC. (e.g, '(None, 27, 27, 16)').
6. models_folder: path to the folder `model, by default "../../models"
7. range_min: an int that, given the possible values that the targeted layer produces, defines the minimum possible value.
8. range_max: an int that, given the possible values that the targeted layer produces, defines the maximum possible value.

## Dump Description
The first script to execute is [`dump_description.py`](dump-description.py) with the following command
```
python dump_description.py
```

Starting from the weights file this program will create a json file called `model_description.json` that, as the name suggests, contains the description of the model's graph. This will allow us to add our custom simulator layer to the model.

## Create Injection Sites
The second step required is to create the necessary injection sites and masks. To do so run the script [`gen_and_save_injections.py`](gen_and_save_injections.py). If the `config.json` has been correctly configured the script will create two numpy files called `injections.npy` and `masks.npy`. 

## Add Custom Layer

We now need to add our custom layer to the model description. To do so open the file `model_description.json` and identify the location in which you want to add the error simulator. 
Suppose we want to inject the output of a convolutional layer just before a maxpool like the following example
```
{
    "class_name": "Conv2D",
    "config":
    {
            "somedata"     
    },
    "name": "conv2d",
    "inbound_nodes":
    [
        [
            [
                "input",
                0,
                0,
                {}
            ]
        ]
    ]
},
{
    "class_name": "MaxPooling2D",
    "config":
    {
        "somedata"
    },
    "name": "max_pooling2d",
    "inbound_nodes":
    [
        [
            [
                "conv2d",
                0,
                0,
                {}
            ]
        ]
    ]
},
```
We need to add the description of our custom layer which will always be the following 
```
{
    "class_name": "ErrorSimulator",
    "config":
    {
        "name": "simulator",
        "trainable": false,
        "dtype": "float32",
        "available_injection_sites": "INJ_SITES",
        "masks": "MASKS",
        "num_inj_sites": "NUM_INJECTION_SITES"
    },
    "name": "simulator",
    "inbound_nodes":
    [
        [
            [
                INBOUND_NODE,
                0,
                0,
                {}
            ]
        ]
    ]
},
```
We need to change the string `INBOUND_NODE` with the name of the node that we want to inject, in our case it will be `conv2d`.
We also need to change the inbound node of the layer that originally followed the targeted one. In our case it is the `MaxPooling2D` layer for which we will change the parameter from 
```
"inbound_nodes":
[
    [
        [
            "conv2d",
            0,
            0,
            {}
        ]
    ]
]
``` 
to 
```
"inbound_nodes":
[
    [
        [
            "simulator",
            0,
            0,
            {}
        ]
    ]
]
``` 
The three placeholders `INJ_SITES`, `MASKS`, `NUM_INJECTION_SITES` will be automatically modified by the next script by adding the values of the numpy arrays we previously created. 

## Finalize Model

The last step in the process is to run the [`finalize_model.py`](finalize_model.py) script.
It will load the original h5 file, upload our new configuration and dump it in a file called `new_weights.h5`. We can now load and run inference on this file and perform error simulation.