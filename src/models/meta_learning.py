import torch
import torch.nn as nn
from src.models.ResNet import *
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.n_way = config.train.n_way
        self.n_support = config.train.n_support
        self.n_query = config.train.n_query
        self.loss_fn = nn.CrossEntropyLoss()
        self.sr = config.features.sr
        self.fps = self.sr / config.features.hop_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if config.train.backbone == 'resnet':
            self.feature_extractor = ResNet(config.features.inchannel, ResBlock)
        # elif config.feature_extractor == 'densenet':
        #     pass
        else:
            raise ValueError('Unsupported feature extractor: {}'.format(config.feature_extractor))
    
    def forward(self,x):
        out  = self.feature_extractor(x)
        return out
    
    @abstractmethod
    def inner_loop(self, support, query):
        pass
    
    
    @abstractmethod
    def train_loop(self, data_loader, optimizer):
        pass
  
            
            
    def test_loop(self, test_loader):
        pass
    
    def split_support_query(self, data):
        pass