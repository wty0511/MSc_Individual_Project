import torch
import torch.nn as nn
from src.models.ResNet import *
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
        elif config.train.backbone == 'convnet':
            self.feature_extractor = ConvNet()
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
    

    def split_support_query(self, data, feature = True):
        pos_data, neg_data = data        
        class_pos, data_pos, label_pos =pos_data
        class_neg, data_neg, label_neg =neg_data
        class_pos = np.array(class_pos)
        class_neg = np.array(class_neg)

        class_pos = class_pos.reshape(-1, self.n_way)
        class_neg = class_neg.reshape(-1, self.n_way)

        class_all = np.concatenate([class_pos, class_neg], axis=1)
        
        
        label_pos = label_pos.view(-1, self.n_way)
        label_neg = label_neg.view(-1, self.n_way)
        label_all = torch.cat([label_pos, label_neg], dim=1)

        
        feature_all = self.forward(torch.cat([data_pos, data_neg], dim=0))
        feature_pos, feature_neg = torch.split(feature_all, [data_pos.size(0), data_neg.size(0)], dim=0)
        
        feature_pos = feature_pos.view(-1, self.n_way, feature_pos.size(-1))
        feature_neg = feature_neg.view(-1, self.n_way, feature_neg.size(-1))
        feature_all = torch.cat([feature_pos, feature_neg], dim=1)
        
        data_pos = data_pos.view(-1, self.n_way, data_pos.size(-2), data_pos.size(-1))
        data_neg = data_neg.view(-1, self.n_way, data_pos.size(-2), data_pos.size(-1))
        data_all = torch.cat([data_pos, data_neg], dim=1)
        
        data_support = data_all[: self.n_support, :, :, :]
        data_query = data_pos[self.n_support :, :, :, :]
        

        class_support = class_all[: self.n_support, :]
        feature_support = feature_all[: self.n_support, :, :]
        label_support = label_all[: self.n_support, :]

        class_query = class_pos[self.n_support :, :]
        feature_query = feature_pos[self.n_support :, :, :]
        label_query = label_pos[self.n_support :, :]
        if feature:
            return {'class':class_support, 'feature':feature_support, 'label':label_support}, {'class':class_query, 'feature':feature_query, 'label':label_query}
        
        else:
            return {'class':class_support, 'data':data_support, 'label':label_support}, {'class':class_query, 'data':data_query, 'label':label_query}
        