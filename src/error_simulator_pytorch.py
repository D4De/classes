from .injection_sites_generator import *
import torch.nn as nn
import torch

from src.loggers import get_logger

logger = get_logger("ErrorSimulator")

class Simulator(nn.Module):
    def __init__(self, layer_type, size, models_folder, fixed_spatial_class = None, fixed_domain_class = None):
        super().__init__()
        self.sites_count = 1
        self.layer_type = layer_type
        self.size = size
        self.models_mode = ''
        self.models_folder = models_folder
        self.fixed_spatial_class = fixed_spatial_class
        self.fixed_domain_class = fixed_domain_class

    def __generate_injection_sites(self, sites_count, layer_type, size, models_folder) -> List[InjectionSite]:
        injection_site = InjectableSite(layer_type, size)

        injection_sites = InjectionSitesGenerator([injection_site], models_folder, fixed_spatial_class=self.fixed_spatial_class, fixed_domain_class=self.fixed_domain_class) \
                .generate_random_injection_sites(sites_count)

        return injection_sites

    def forward(self, x):
        injection_site = self.__generate_injection_sites(1, self.layer_type, self.size, self.models_folder)
        range_min = torch.min(x)
        range_max = torch.max(x)
        if len(injection_site) > 0  and len(injection_site[0]) > 0:
            for idx, value in injection_site[0].get_indexes_values():
                x[idx] = float(value.get_value(range_min, range_max))
        else:
            raise RuntimeError("No injection happened")
        return x
