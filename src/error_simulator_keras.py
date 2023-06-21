from typing import Optional
import tensorflow as tf
import numpy as np
from .injection_sites_generator import InjectableSite, InjectionSitesGenerator, InjectionValue
from enum import IntEnum
import sys

from src.loggers import get_logger

logger = get_logger("ErrorSimulator")

def create_injection_sites_layer_simulator(num_requested_injection_sites : int, layer_type : str, layer_output_shape_cf : str,
                                           layer_output_shape_cl : str, models_folder : str, range_min : float = None, range_max : float = None, fixed_spatial_class : Optional[str] = None, fixed_domain_class : Optional[dict] = None):
    def __generate_injection_sites(sites_count : int, layer_type : str, size : int):

        injection_site = InjectableSite(layer_type, size)

        injection_sites = InjectionSitesGenerator([injection_site], models_folder, fixed_spatial_class, fixed_domain_class).generate_random_injection_sites(sites_count)

        return injection_sites

    if range_min is None or range_max is None:
        range_min = -30.0
        range_max = 30.0
        logger.warn(f"""range_min and/or range_max are not specified, so their default values will be kept ({range_min=}, {range_max=}). 
                        You may want to change the defaults by calculating the average range_min and range_max average, 
                        and specify them as inputs of this function. """)


    available_injection_sites = []
    masks = []

    for _ in range(num_requested_injection_sites):
        curr_injection_sites = __generate_injection_sites(1, layer_type, layer_output_shape_cf)
        shape = eval(layer_output_shape_cl.replace('None', '1'))
        curr_inj_nump = np.zeros(shape=shape[1:])
        curr_mask = np.ones(shape=shape[1:])

        if len(curr_injection_sites) > 0:
            for idx, value in curr_injection_sites[0].get_indexes_values():
                channel_last_idx = (idx[0], idx[2], idx[3], idx[1])
                curr_mask[channel_last_idx[1:]] = 0
                curr_inj_nump[channel_last_idx[1:]] += value.get_value(range_min, range_max)

            available_injection_sites.append(curr_inj_nump)
            masks.append(curr_mask)

    return available_injection_sites, masks

class ErrorSimulatorMode(IntEnum):
    disabled = 1,
    enabled  = 2

'''
Inject Different fault for each image inside a batch
'''
@tf.function
def fault_injection_batch(inputs,__num_inj_sites,__available_injection_sites,__masks):
    #Extract batch Size
    shape       = tf.shape(inputs)
    batch_size  = shape[0]

    #Compute fault on first image
    init = fault_injection(inputs[0],__num_inj_sites,__available_injection_sites,__masks)
    i=tf.constant(1)

    '''
    While exit condition
    '''
    def condition(i,_):
        return i<batch_size
    
    '''
    While body functiom, execute fault injection on each element of the batch
    '''
    def iteration(i,outputs):
        tmp = fault_injection(inputs[i],__num_inj_sites,__available_injection_sites,__masks)
        outputs = tf.concat([outputs,tmp],0)
        i+=1
        return [i,outputs]
    
    #Execute a while loop of batch size to apply fault on full batch
    i, outputs = tf.while_loop(condition, iteration,
                                [i, init],
                                [i.get_shape(), tf.TensorShape([None,init.get_shape()[1],init.get_shape()[2],init.get_shape()[3]])])
    
    #tf.print(outputs)
    return outputs
    

    '''
    Inject a Fault from the generated injection sites on the selected input
    '''
def fault_injection(inputs,__num_inj_sites,__available_injection_sites,__masks):
    random_index = tf.random.uniform(
        shape=[1], minval=0,
        maxval=__num_inj_sites, dtype=tf.int32, seed=22)

    #print(f"Fault from {self.name}")
    random_tensor   = tf.gather(__available_injection_sites, random_index)
    random_mask     = tf.gather(__masks, random_index)

    #return [inputs[i] * random_mask + random_tensor, random_tensor, random_mask]
    return inputs * random_mask + random_tensor
 
class ErrorSimulator(tf.keras.layers.Layer):

    def __init__(self, available_injection_sites, masks, num_inj_sites, **kwargs):

        super(ErrorSimulator, self).__init__(**kwargs)
        self.__num_inj_sites = num_inj_sites
        self.__available_injection_sites = []
        self.__masks = []

        #Parameter to chose between enable/disable faults
        self.mode = tf.Variable([[int(ErrorSimulatorMode.enabled)]],shape=tf.TensorShape((1,1)),trainable=False) 
        
        for inj_site in available_injection_sites:
            self.__available_injection_sites.append(tf.convert_to_tensor(inj_site, dtype=tf.float32))
        for mask in masks:
            self.__masks.append(tf.convert_to_tensor(mask, dtype=tf.float32))

    '''
    Allow to enable or disable the Fault Layer
    '''
    def set_mode(self, mode:ErrorSimulatorMode):
        self.mode.assign([[int(mode)]])
    
    def call(self, inputs):
        #tf.print("MODE LAYER :", self.mode, tf.constant([[int(ErrorSimulatorMode.disabled)]]), output_stream=sys.stdout)
        #TF operator to check which mode is active
        #If Disabled => Return Vanilla output
        #If Enabled  => Return Faulty  output
        return tf.cond(self.mode == tf.constant([[int(ErrorSimulatorMode.disabled)]]),
                       true_fn=lambda: inputs,
                       false_fn=lambda: fault_injection_batch(inputs,self.__num_inj_sites,self.__available_injection_sites,self.__masks))
 