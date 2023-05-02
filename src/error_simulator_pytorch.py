from .injection_sites_generator import *
import torch.nn as nn
import torch


class Simulator(nn.Module):
    def __init__(self, layer_type, size, models_folder):
        super().__init__()
        self.sites_count = 1
        self.layer_type = layer_type
        self.size = size
        self.models_mode = ''
        self.models_folder = models_folder

    def __generate_injection_sites(self, sites_count, layer_type, size, models_folder):
        injection_site = InjectableSite(layer_type, '', size)

        try:
            injection_sites, _, _ = InjectionSitesGenerator([injection_site], '', models_folder) \
                .generate_random_injection_sites(sites_count)
        except:
            return []

        return injection_sites

    def forward(self, x):
        injection_site = self.__generate_injection_sites(1, self.layer_type, self.size, self.models_folder)

        if len(injection_site) > 0:
            for idx, value in injection_site[0].get_indexes_values():
                if value.value_type == '[-1,1]':
                    x[idx] += value.raw_value
                else:
                    x[idx] = torch.from_numpy(np.asarray(value.raw_value))
        return x
