import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.models.meta_learning import BaseModel
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class ProtoNet(BaseModel):
    def __init__(self, config):
        super(ProtoNet, self).__init__(config)
        self.test_loop_batch_size = config.val.test_loop_batch_size


    def inner_loop(self,support, query):
        prototype     = support['feature'].view(self.n_support, self.n_way, -1 ).mean(0).squeeze()
        dists = self.euclidean_dist(query['feature'], prototype)
        scores = -dists
        y_query = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query))
        return self.loss_fn(scores, y_query )

    @abstractmethod
    def train_loop(self, data_loader, optimizer):
        for i, data in enumerate(data_loader):
            support_data, query_data = self.split_support_query(data)
            optimizer.zero_grad()
            loss = self.inner_loop(query_data, support_data)
            loss.backward()
            optimizer.step()
  
            
            
    def test_loop(self, test_loader):
        all_prob = []
        for i, (pos_sup, neg_sup, query, seg_len, seg_hop, query_start) in enumerate(test_loader):
            pos_dataset = TensorDataset(pos_sup, None)
            neg_dataset = TensorDataset(neg_sup, None)
            query_dataset = TensorDataset(query, None)
            pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            pos_feat = []
            for batch in pos_loader:
                pos_data, _ = batch
                feat = self.forward(pos_data)
                pos_feat.append(feat.mean(0))
            pos_feat = torch.stack(pos_feat, dim=0).mean(0)

            neg_feat = []
            for batch in neg_loader:
                neg_data, _ = batch
                feat = self.forward(neg_data)
                neg_feat.append(feat.mean(0))
            neg_feat = torch.stack(neg_feat, dim=0).mean(0)
            
            query_feat = []
            for batch in query_loader:
                query_data, _ = batch
                feat = self.forward(query_data)
                query_feat.append(feat)
            query_feat = torch.cat(query_feat, dim=0)
            pos_feat = torch.stack([pos_feat,neg_feat], dim=0)
            dist = self.euclidean_dist(query_feat, pos_feat)
            scores = -dist
            prob = F.softmax(scores, dim=1)
            prob = prob[:,0].detach().cpu().numpy()
            prob = np.where(prob>self.config.val.threshold, 1, 0)
            all_prob.extend(prob)
        
        on_set = np.flatnonzero(np.diff(np.concatenate(np.zeros(1),all_prob))==1)
        off_set = np.flatnonzero(np.diff(np.concatenate(all_prob,np.zeros(1)))==-1) + 1 #off_set is the index of the first 0 after 1
        query_start_time = query_start/self.fps
        
        on_set_time = on_set*seg_hop/self.fps + query_start_time
        off_set_time = off_set*seg_hop/self.fps + query_start_time
        
        return on_set_time, off_set_time
            
    
    def split_support_query(self, data):
        pos_data, neg_data = data
        class_pos, feature_pos, label_pos =pos_data
        class_neg, feature_neg, label_neg =neg_data
        feature_all = self.forward(feature_pos.cat(feature_neg, dim=0))
        feature_pos, feature_neg = torch.split(feature_all, [feature_pos.size(0), feature_neg.size(0)], dim=0)
        
        class_support = list(class_pos[:self.n_way*self.n_support]).extend(list(class_neg[:self.n_way*self.n_support]))
        feature_support = feature_pos[:self.n_way*self.n_support].cat(feature_neg[:self.n_way*self.n_support], dim=0)
        label_support = label_pos[:self.n_way*self.n_support].cat(label_neg[:self.n_way*self.n_support], dim=0)
        
        class_query = list(class_pos[self.n_way*self.n_support:]).extend(list(class_neg[self.n_way*self.n_support:]))
        feature_query = feature_pos[self.n_way*self.n_support:].cat(feature_neg[self.n_way*self.n_support:], dim=0)
        label_query = label_pos[self.n_way*self.n_support:].cat(label_neg[self.n_way*self.n_support:], dim=0)
        
        return {'class':class_support, 'feature':feature_support, 'label':label_support}, {'class':class_query, 'feature':feature_query, 'label':label_query}
                



    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)
        return torch.pow(query - support, 2).sum(2)
