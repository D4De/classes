from src.error_simulator_keras import create_injection_sites_layer_simulator
import json
import numpy as np

with open('config.json', 'r') as f:
	data = json.load(f)
	

available_injection_sites, masks = create_injection_sites_layer_simulator(
	num_requested_injection_sites = data['injection_sites'] * 5,
	layer_type = data['layer_type'], 
    layer_output_shape_cf = data['layer_output_shape_cf'],
    layer_output_shape_cl = data['layer_output_shape_cl'], 
    models_folder = data['models_folder'], 
    range_min = data['range_min'], 
    range_max = data['range_max']
)

np.save('injections.npy', available_injection_sites)
np.save('masks.npy', masks)