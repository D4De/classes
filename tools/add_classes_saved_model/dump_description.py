import h5py
import numpy as np
import json
from sys import argv

with open('config.json', 'r') as f:
	data = json.load(f)
# Open the original model file
original_model_path = data['model']
original_model_file = h5py.File(original_model_path, 'r')

# Get the description of the model
description = original_model_file.attrs['model_config']

# Dump the data in a json file
with open('model_description.json', 'w') as f:
	json.dump(json.loads(description), f, indent=1)