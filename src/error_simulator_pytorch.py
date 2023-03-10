from src.injection_sites_generator import *
import torch.nn as nn


class Simulator(nn.Module):
    def __init__(self, layer_type, size):
        super().__init__()
        self.sites_count = 1
        self.layer_type = layer_type
        self.size = size
        self.models_mode = ''

    def __generate_injection_sites(self, sites_count, layer_type, size):
        injection_site = InjectableSite(layer_type, '', size)

        try:
            injection_sites, _, _ = InjectionSitesGenerator([injection_site], '') \
                .generate_random_injection_sites(sites_count)
        except:
            return []

        return injection_sites

    def forward(self, x):
        injection_site = self.__generate_injection_sites(1, self.layer_type, self.size)

        if len(injection_site) > 0:
            for idx, value in injection_site[0].get_indexes_values():
                if value.value_type == '[-1,1]':
                    x[idx] += value.raw_value
                else:
                    x[idx] = value.raw_value
        return x
