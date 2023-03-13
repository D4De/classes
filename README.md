# Classes
CLASSES (Cross-Layer AnalysiS framework for Soft-Errors effectS in CNNs), a novel cross-layer framework for an early, accurate and fast reliability analysis of CNNs accelerated onto GPUs when affected by SEUs.
A theoretical description of the implemented framework can be found in:
C. Bolchini, L. Cassano, A. Miele and A. Toschi, "Fast and Accurate Error Simulation for CNNs Against Soft Errors," in IEEE Transactions on Computers, 2022, doi: 10.1109/TC.2022.3184274. <br>

If you use Classes in your research, we would appreciate a citation to:

>@ARTICLE{bc+2022ea,<br>
>  author={Bolchini, Cristiana and Cassano, Luca and Miele, Antonio and Toschi, Alessandro},<br>
>  journal={IEEE Transactions on Computers}, <br>
>  title={{Fast and Accurate Error Simulation for CNNs Against Soft Errors}}, <br>
>  year={2022},<br>
>  volume={},<br>
>  number={},<br>
>  pages={1-14},<br>
>  doi={10.1109/TC.2022.3184274}<br>
>}

## Table of Contents

1. [Copyright & License](#copyright--license)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [TensorFlow2 - As a K function](#as-a-k-function)
    2. [TensorFlow2 - As a layer](#as-a-layer)

## Copyright & License

Copyright (C) 2023 Politecnico di Milano.

This framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/) for more details.

Neither the name of Politecnico di Milano nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

## Dependencies 
The following libraries are required in order to correctly use the framework. <br>
N.B. This framework has been developed for TensorFlow1, TensorFlow2 and PyTorch. If you want to use it with one of the frameworks you don't need to install also the others.

1. Python3 
2. Numpy package
3. TensorFlow 2.10 or lower
4. TensorFlow 1
5. PyTorch

## Installation 

To install the framework you only need to clone the repository 
```
git clone https://github.com/D4De/CLASSES.git
```
and import the `src` folder.

## How it works
To fully understand how Classes works we strongly suggest to read our paper _Fast and Accurate Error Simulation for CNNs Against Soft Errors_.
Nonetheless, we want to provide a small description of its operation. 
The following image provides a high level representation of the whole framework, we executed an error injection campaign
at GPU level using NVBitFI for each of the [supported layers](#operators-supported) in order to create a database of error models.
<br>
![](framework.png)
<br>
These models are then used by Classes in conjunction with either TensorFlow or PyTorch to simulate the presence of an error
during the execution of a given model.

Here we can see more in depth how Classes works.
<br>
![](classes.png)
<br>
It uses the concept of a saboteur, either developed [as a layer](#as-a-layer) or [as a backend function](#as-a-k-function).
The idea is to interrupt the execution of the model after the target layer has been executed, corrupt the output obtained
up until that point and then resume the execution. For this reason the error models adopted represent the effect of an error at the
output of each layer.

Due to how TensorFlow2 works a layer could contain both an operator and an activation function, to target each part the 
user of Classes must execute two distinct campaign selecting different [OperatorType](src/operators.py) accordingly to
the components of the target layer.
## Usage
The framework is composed by two distinct modules, the injection sites generator and the error simulator. The first one, as the name suggests, is responsible for the creaton of the injection sites and it is common among all the different implementations of the error simulator. The error simulator itself has been implemented in four different ways in order to work with both TensorFlow (1 and 2) and PyTorch. We provide here an high level description of both modules, a more precise description with examples can be found in the `example` folder. 

### Injection sites generator 

This module is used to create the injection sites needed for the error simulation campaign. The whole module is accessed through the function 
`generate_injection_sites(sites_count, layer_type, layer_name, size, models_mode='')`
its inputs are:
- sites_count: the number of injection sites to create
- layer_type: what kind of layer we are injecting
- layer_name: a string representing the layer name, can be empty
- size: a string that defines the output shape of the layer we want to target. It must be defined with the following format '(None, Channel, Height, Width)' 

This function returns three lists: 
- injection_sites: list of injection sites, each injection site has an index and a value, the value itself has two parameters, value_type that describes what kind of value has been selected based on our error models and raw_value which is the numerical value. 
- cardinalities: list of cardinalities selected.
- patterns: list of patterns selected.
- 
### Operators supported
The operators supported for the error simulation are the ones described in the enum [`OperatorType`](src/operators.py). Currently, we support the following layers:
- Conv2D1x1: Convolution 2D with kernel size of 1.
- Conv2D3x3: Convolution 2D with kernel size of 3.
- Conv2D3x3S2: Convolution 2D with kernel size of 3 and stride of 2.
- AddV2: Add between two tensors.
- BiasAdd: Add between a tensor and a vector.
- Mul: Multiplication between a tensor and a scalar.
- FusedBatchNormV3: Batch normalization.
- RealDiv: Division between a tensor and a scalar.
- Exp: Exp activation function.
- LeakyRelu: Leaky Relu activation function.
- Sigmoid: Sigmoid activation function.
- Add: Add between two tensors.
- Conv2D: Convolution that does not fit the previous three descriptions
- FusedBatchNorm: Batch normalization, needed to support some implementations of the layer

### Fault injector

The module used to perform the fault injection campaign has been developed in four different shapes. 

### Tensorflow2
### As a K function

Using Keras backend functions allows us to split a model into two separate submodels, one that performs the computation from the input layer to the one we selected for the injection and one that produces the final result from the output of the selected layer. With this approach we can obtain the selected layer's output and modify it before feeding it to the second submodel.
If the model is not linear (e.g. a unet) this approach requires us to make sure that each skip connection is correctly set for the final output to be correctly produced. In the `example/tensorflow2/k_function` folder we provide two examples showing how to use this version of the error simulator on a linear model and on a model with skip connections.

### As a layer

We also developed the injector as a layer that can be placed into the module like any other keras layer. This approach simplifies the process in case of models with skip connections since we do not need to manually restore them but it has some drawbacks. The most important one is that due to how TensorFlow2 works we are not able to generate injection sites each time we perform a prediction, instead we need to create them at setup time and pass them as inputs to the layer. The injector will randomly select one at each prediction.

### PyTorch
### As a layer
For the PyTorch version of the framework we only developed the layer version since we can freely execute python code
during inference thus avoiding the problem of creating all the error models beforehand.