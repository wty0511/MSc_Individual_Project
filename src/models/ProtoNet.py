import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.models.meta_learning import BaseModel
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
from src.evaluation_metrics.evaluation import *
from src.utils.feature_extractor import *


class ProtoNet(BaseModel):
    def __init__(self, config):
        super(ProtoNet, self).__init__(config)
        
        self.test_loop_batch_size = config.val.test_loop_batch_size


    def inner_loop(self,support, query):
        # print(support['label'].view(self.n_support, self.n_way * 2, -1 ))
        prototype     = support['feature'].mean(0).squeeze()
        query['label'] = query['label'].to(self.device)
        dists = self.euclidean_dist(query['feature'].view(-1, query['feature'].shape[-1]), prototype)
        pred = dists.argmin(-1)
        acc = torch.eq(pred, query['label'].view(-1)).float().mean()
        scores = -dists
        y_query = torch.from_numpy(np.tile(np.arange(self.n_way * 2),self.n_query)).long().to(self.device)
        # print(y_query)
        # print(pred)
        return self.loss_fn(scores, y_query ), acc


    def train_loop(self, data_loader, optimizer):
        avg_loss = 0
        for i, data in enumerate(data_loader):
            #print('waiting for data')
            support_data, query_data = self.split_support_query(data)
            #print('data split')
            optimizer.zero_grad()

            loss, acc = self.inner_loop(support_data, query_data)
            loss.backward()
            optimizer.step()
            if i % 5 == 0:
                print('Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(i+1, len(data_loader), loss.item(), acc.item()))
            avg_loss = avg_loss + loss.item()

        avg_loss = avg_loss / len(data_loader)
        return avg_loss
            
    def test_loop(self, test_loader):
        all_prob = {}
        all_time = {'Audiofilename':[], 'Starttime':[], 'Endtime':[]}
        for i, (pos_sup, neg_sup, query, seg_len, seg_hop, query_start) in enumerate(test_loader):
            seg_hop = seg_hop.item()
            query_start = query_start.item()

            # print(pos_sup[1].squeeze().shape)
            # print(neg_sup[1].squeeze().shape)
            # print(query.shape)
            wav_file= pos_sup[0][0].split('&')[1]
            pos_dataset = TensorDataset(pos_sup[1].squeeze(), torch.zeros(pos_sup[1].squeeze().shape[0]))
            neg_dataset = TensorDataset(neg_sup[1].squeeze(), torch.zeros(neg_sup[1].squeeze().shape[0]))

            query_dataset = TensorDataset(query.squeeze(), torch.zeros(query.squeeze().shape[0]))

            pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            # print(len(pos_dataset))
            # print(len(neg_dataset))
            # print(len(query_dataset))
            pos_feat = []
            for batch in pos_loader:
                pos_data, _ = batch
                feat = self.forward(pos_data)
                # print(feat.shape)
                pos_feat.append(feat.mean(0))
            pos_feat = torch.stack(pos_feat, dim=0).mean(0)

            neg_feat = []
            for batch in neg_loader:
                neg_data, _ = batch
                # print(neg_data.shape)
                feat = self.forward(neg_data)
                # print(feat.shape)
                neg_feat.append(feat.mean(0))
            neg_feat = torch.stack(neg_feat, dim=0).mean(0)
            
            proto = torch.stack([pos_feat,neg_feat], dim=0)
            
            prob_all = []
            for batch in query_loader:
                query_data, _ = batch
                feat = self.forward(query_data)
                dist = self.euclidean_dist(feat, proto)
                scores = -dist
                prob = F.softmax(scores, dim=1)
                prob_all.append(prob.detach().cpu().numpy())

            prob_all = np.concatenate(prob_all, axis=0)
            prob_all = prob_all[:,0]
            prob_all = np.where(prob_all>self.config.val.threshold, 1, 0)

            all_prob[wav_file] = prob_all
            
        for wav_file in all_prob.keys():
            prob = all_prob[wav_file]
            on_set = np.flatnonzero(np.diff(np.concatenate(([0],prob), axis=0))==1)
            off_set = np.flatnonzero(np.diff(np.concatenate((prob,[0]), axis=0))==-1) + 1 #off_set is the index of the first 0 after 1
            query_start_time = query_start/self.fps
            
            on_set_time = on_set*seg_hop/self.fps + query_start_time
            off_set_time = off_set*seg_hop/self.fps + query_start_time
            all_time['Audiofilename'].extend([os.path.basename(wav_file)]*len(on_set_time))
            all_time['Starttime'].extend(on_set_time)
            all_time['Endtime'].extend(off_set_time)
            # print(len(on_set_time))
            # print(len(off_set_time))
            # print('~~~~~~~~~~~~')
            
            
        df_all_time = pd.DataFrame(all_time)
        
        pred_path = normalize_path(self.config.val.pred_dir)
        if not os.path.dirname(pred_path):
            os.makedirs(os.path.dirname(pred_path))
        df_all_time.to_csv(pred_path, index=False)

        ref_files_path = normalize_path(self.config.path.val_dir)
        report_dir = normalize_path(self.config.val.report_dir)
        report = evaluate(pred_path, ref_files_path, self.config.team_name, self.config.dataset, report_dir)
        return df_all_time, report
            
    
    def split_support_query(self, data):
        pos_data, neg_data = data        
        class_pos, feature_pos, label_pos =pos_data
        class_neg, feature_neg, label_neg =neg_data
        class_pos = np.array(class_pos)
        class_neg = np.array(class_neg)

        class_pos = class_pos.reshape(-1, self.n_way)
        class_neg = class_neg.reshape(-1, self.n_way)
        class_all = np.concatenate([class_pos, class_neg], axis=1)
        
        
        label_pos = label_pos.view(-1, self.n_way)
        label_neg = label_neg.view(-1, self.n_way)
        label_all = torch.cat([label_pos, label_neg], dim=1)
        
        feature_all = self.forward(torch.cat([feature_pos, feature_neg], dim=0))
        feature_pos, feature_neg = torch.split(feature_all, [feature_pos.size(0), feature_neg.size(0)], dim=0)
        
        feature_pos = feature_pos.view(-1, self.n_way, feature_pos.size(-1))
        feature_neg = feature_neg.view(-1, self.n_way, feature_neg.size(-1))
        
        feature_all = torch.cat([feature_pos, feature_neg], dim=1)
        

        

        class_support = class_all[: self.n_support, :]
        feature_support = feature_all[: self.n_support, :, :]
        label_support = label_all[: self.n_support, :]

        class_query = class_all[self.n_support :, :]
        feature_query = feature_all[self.n_support :, :, :]
        label_query = label_all[self.n_support :, :]
        # print('label')
        # print(label_support)
        # print(label_query)
        
        
        return {'class':class_support, 'feature':feature_support, 'label':label_support}, {'class':class_query, 'feature':feature_query, 'label':label_query}
                



    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)

        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)

        return torch.pow(query - support, 2).sum(2)
