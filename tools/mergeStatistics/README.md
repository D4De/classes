# Introduction

Merge Statistics is a script able to merge the results of spatial error models given a set of spatial models and their
weights.
Using merge statistics, different results can be melded together in order to obtain a single spatial model that
describes the correct distribution of the error classes in the experiments.

Authors are Filippo Balzarini ([@filomba01](https://github.com/filomba01)) and Matteo
Bettiati ([@matteobettiati](https://github.com/matteobettiati))

## Table of Contents

1. [Copyright & License](#Copyright-&-License)
2. [Dependencies](#Dependencies)
3. [Installation](#Installation)
4. [How it works](#How-it-works)
5. [Usage](#Usage)

### Copyright & License

Copyright (C) 2023 Politecnico di Milano.

This framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Neither the name of Politecnico di Milano nor the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

### Dependencies

The following libraries is required in order to correctly use the framework.

* Python3

### Installation

To install the script it is necessary to clone the repository.

```
git clone https://github.com/filomba01/CNN_ErrorClassification.git
```

If only this script is necessary, you can clone only the subdirectory

```
git clone --depth 1 --no-checkout https://github.com/filomba01/CNN_ErrorClassification.git

cd mergeStatistics

git sparse-checkout set CNN_ErrorClassification/mergeStatistics
```

### How it works

Merge Statistics is a simple script that, given a set of spatial error models and their weights merge and return a
new spatial model that describes the error pattern distribution. The script is useful for stick together different
experiment's results without losing information.

Weights should be chosen, for example, based on the dimension of the
experiments, given two experiment's results, if the first experiment collected data from 200 corrupted tensors, and a
second experiment, that collected only 100 corrupted tensors, it would be useful to assign 0,67 (200/300) to the first
experiment and 0.33 to the second.

### Usage

This script is structured to receive as arguments a precise sequence of data.
You just have to write a json file that contains the experiments to merge and their relative weights.
You have to set the relative path of the experiments and the weights must be a decimal number from 0 to 1.
Here's an example:

```
    [
        {
            "file": "relative/path/to/the/experiment1",
            "weigth": 0.8
        },
        {
            "file": "relative/path/to/the/experiment2",
            "weigth": 0.2
        }
    ]
```
Remember that the sum of the weights must be equals to one.

Now you can insert the line to run from terminal like this:
```
    python mergeStatistics.py jsonfile.json
```
Finally just run the code.