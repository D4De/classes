import tensorflow as tf
from injection_sites_generator import *
from random import random


class FaultInjector(tf.keras.layers.Layer):

    def __init__(self, available_injection_sites, masks, num_inj_sites, **kwargs):

        super(FaultInjector, self).__init__(**kwargs)
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
        return [inputs * random_mask + random_tensor, random_tensor, random_mask,]

