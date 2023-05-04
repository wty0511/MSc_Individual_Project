import torch
import torch.nn as nn
from src.models.ResNet import *
from src.models.ResNet2 import *
from abc import abstractmethod
from src.models.ConvNet import *
import numpy as np
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
        elif config.train.backbone == 'resnet2':
            self.feature_extractor  = ResNet2()
        elif config.train.backbone == 'convnet':
            self.feature_extractor = ConvNet()
        else:
            raise ValueError('Unsupported feature extractor: {}'.format(config.train.backbone))
        
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
    
    
    def split2nway_1d(self, data):
        
        
        
        class_data, data, label = data
        class_data = np.array(class_data)
        class_data = class_data.reshape(-1, self.n_way)
        label = label.view(-1, self.n_way)
        return class_data, data, label
    
    def split_2d(self, data):
        if torch.is_tensor(data):
            data = data.view(-1, self.n_way, data.size(-2), data.size(-1))
            return data
        else:
            raise ValueError('Unsupported data type: {}'.format(type(data)))
    
    def split_1d(self, data):
        if torch.is_tensor(data):
            return data.view(-1, self.n_way, data.size(-1))
        elif isinstance(data, np.ndarray):
            return data.reshape(-1, self.n_way, data.shape[-1])
        else:
            raise ValueError('Unsupported data type: {}'.format(type(data)))



    def split_support_query_data(self, data_pos, data_neg):
        if self.config.train.neg_prototype:
            data_pos = self.split_2d(data_pos)
            data_neg = self.split_2d(data_neg)
            data_all = torch.cat([data_pos, data_neg], dim=1)
            data_support = data_all[:self.n_support, :, :, :]
            data_query = data_all[self.n_support:, :, :, :]
        else:
            data_pos = self.split_2d(data_pos)
            data_support = data_pos[:self.n_support, :, :, :]
            data_query = data_pos[self.n_support:, :, :, :]
        data_support = data_support.view(-1, data_support.size(-2), data_support.size(-1))
        data_query = data_query.view(-1, data_query.size(-2), data_query.size(-1))
        return data_support, data_query
    
    def split_support_query_feature(self, pos_input, neg_input, is_data, model = None):
        
        if self.config.train.neg_prototype:
            if is_data:
                if model is not None:
                    feature_pos = model.forward(pos_input)
                    feature_neg = model.forward(neg_input)
                else:
                    feature_pos = self.forward(pos_input)
                    feature_neg = self.forward(neg_input)
            
            else:
                feature_pos = pos_input
                feature_neg = neg_input
            
            feature_pos = feature_pos.view(-1, self.n_way, feature_pos.size(-1))
            feature_neg = feature_neg.view(-1, self.n_way, feature_neg.size(-1))
            feature_all = torch.cat([feature_pos, feature_neg], dim=1)
            
            feature_support = feature_all[:self.n_support, :, :]
            feature_query = feature_all[self.n_support:, :, :]
            
        else:
            if is_data:
                if model is not None:
                    feature_pos = model.forward(pos_input)
                else:
                    feature_pos = self.forward(pos_input)
                
            else:
                feature_pos = pos_input
            
            feature_pos = feature_pos.view(-1, self.n_way, feature_pos.size(-1))
            feature_support = feature_pos[:self.n_support, :, :]
            feature_query = feature_pos[self.n_support:, :, :]
        
        return feature_support, feature_query
        