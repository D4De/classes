import h5py
import numpy as np
import json
import os 

with open('config.json', 'r') as f:
	data = json.load(f)
	
# Open the original model file
original_model_path = data['model']
original_model_file = h5py.File(original_model_path, 'r')

# Create a new model file to save the modified model
modified_model_path = os.path.join(data['new_model'], 'new_weights.h5')
modified_model_file = h5py.File(modified_model_path, 'w')

with open('model_description.json', "r") as f:
	new_config = json.load(f)

masks = np.load('masks.npy')
injections = np.load('injections.npy')

for layer in new_config['config']['layers']:
	if layer['class_name'] == 'ErrorSimulator':
		layer['config']['INJ_SITES'] = injections.tolist()
		layer['config']['MASKS'] = masks.tolist()
		layer['config']['NUM_INJECTION_SITES'] = data['injection_sites']
	
# Convert the modified model configuration back to JSON format
modified_config_json = json.dumps(new_config)

# Write the modified model configuration to the new model file
modified_model_file.attrs.create('model_config', modified_config_json.encode('utf-8'))

# Iterate over the original model file and copy each layer to the new model file
for layer_name in original_model_file.keys():
    original_model_file.copy(layer_name, modified_model_file)

# Close both model files
original_model_file.close()
modified_model_file.close()

