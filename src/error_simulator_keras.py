import tensorflow as tf
import numpy as np
from src.injection_sites_generator import InjectableSite, InjectionSitesGenerator


def create_injection_sites_layer_simulator(num_requested_injection_sites, layer_type, layer_output_shape_cf,
                                           layer_output_shape_cl, models_folder):
    def __generate_injection_sites(sites_count, layer_type, size, models_mode=''):

        injection_site = InjectableSite(layer_type, '', size)
        try:
            injection_sites, cardinality, pattern = InjectionSitesGenerator([injection_site],
                                                                            models_mode, models_folder)\
                .generate_random_injection_sites(sites_count)
        except:
            return []
        return injection_sites, cardinality, pattern

    available_injection_sites = []
    masks = []

    for _ in range(num_requested_injection_sites):
        curr_injection_sites, cardinality, pattern = __generate_injection_sites(1, layer_type, layer_output_shape_cf)
        shape = eval(layer_output_shape_cl.replace('None', '1'))
        curr_inj_nump = np.zeros(shape=shape[1:])
        curr_mask = np.ones(shape=shape[1:])

        if len(curr_injection_sites) > 0:
            for idx, value in curr_injection_sites[0].get_indexes_values():
                channel_last_idx = (idx[0], idx[2], idx[3], idx[1])
                if value.value_type == '[-1,1]':
                    curr_inj_nump[channel_last_idx[1:]] += value.raw_value
                else:
                    curr_mask[channel_last_idx[1:]] = 0
                    curr_inj_nump[channel_last_idx[1:]] += value.raw_value

            available_injection_sites.append(curr_inj_nump)
            masks.append(curr_mask)

    return available_injection_sites, masks


class ErrorSimulator(tf.keras.layers.Layer):

    def __init__(self, available_injection_sites, masks, num_inj_sites, **kwargs):

        super(ErrorSimulator, self).__init__(**kwargs)
        self.__num_inj_sites = num_inj_sites
        self.__available_injection_sites = []
        self.__masks = []
        self.__cardinalities = []
        self.__patterns = []

        for inj_site in available_injection_sites:
            self.__available_injection_sites.append(tf.convert_to_tensor(inj_site, dtype=tf.float32))
        for mask in masks:
            self.__masks.append(tf.convert_to_tensor(mask, dtype=tf.float32))

    def call(self, inputs):
        random_index = tf.random.uniform(
            shape=[1], minval=0,
            maxval=self.__num_inj_sites, dtype=tf.int32, seed=22)

        random_tensor = tf.gather(self.__available_injection_sites, random_index)
        random_mask = tf.gather(self.__masks, random_index)
        return [inputs * random_mask + random_tensor, random_tensor, random_mask]
