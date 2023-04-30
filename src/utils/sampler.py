import torch
import numpy as np
from torch.utils.data.sampler import Sampler

class BatchSampler(Sampler):
    def __init__(self, config, classes, data_set_len):
        self.config = config
        self.n_way = config.train.n_way
        self.n_support = config.train.n_support
        self.n_query = config.train.n_query
        self.class_list = list(classes)
        self.n_episode = data_set_len//(self.n_way*(self.n_support+self.n_query))
    def __iter__(self):
        for _ in range(self.n_episode):
            # randomly select n_way classes
            selected_classes = np.random.choice(self.class_list, self.n_way, replace=False)
            yield np.tile(selected_classes, self.n_support +self.n_query)
            
    def __len__(self):
        return self.n_episode