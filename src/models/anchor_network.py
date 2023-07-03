# This code is modified from https://github.com/haoheliu/DCASE_2022_Task_5

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
import time
from sklearn.metrics import classification_report, f1_score
from src.utils.post_processing import *
from src.evaluation_metrics.evaluation_confidence_intervals import *
from copy import deepcopy
import random
import torch.optim as optim
import h5py
from pytorch_metric_learning import  losses,reducers
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from src.utils.distance import NormMinusLpDistance


class AnchorNet(BaseModel):
    def __init__(self, config):
        super(AnchorNet, self).__init__(config)
        
        self.test_loop_batch_size = config.val.test_loop_batch_size
        # self.loss_func = losses.ProxyAnchorLoss(num_classes=19, embedding_size=512,   alpha = 32, margin =0.1, distance = NormMinusLpDistance(power = 2, p =2, normalize_embeddings = False)).to(torch.device('cuda'))
        self.loss_func = losses.ProxyAnchorLoss(num_classes=19, embedding_size=512,   alpha = 32, margin =0.1).to(torch.device('cuda'))
    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)
        return torch.pow(query - support, 2).sum(2)

    
    def feed_forward_test(self, prototype, query_data):
        # Execute a model with given output layer weights and inputs

        query_feat = self.feature_extractor(query_data)
        dists = self.cos_dist(query_feat, prototype)
        # dists = -self.euclidean_dist(query_feat, prototype)
        dists = dists
        preds = F.softmax(dists, dim = 1)
        preds = preds.detach().cpu().numpy()
        return preds, query_feat
    

    def cos_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        query = query.unsqueeze(1).expand(n, m, -1)
        query = F.normalize(query, p=2, dim=2)
        support = support.unsqueeze(0).expand(n, m, -1)
        support = F.normalize(support, p=2, dim=2)
        sim = F.cosine_similarity(query, support, dim=2)
        # sim /= 0.1
        return sim

        
                    
                    
    def train_loop(self, data_loader, loss_optimizer, model_optimizer):
        self.feature_extractor.train()
        loss_all = []
        for batch in data_loader:
            data, label = batch
            # print(data.shape)
            # print(label)
            feat = self.feature_extractor(data)
            # print(feat.shape)
            # print(label.shape)
            loss = self.loss_func(feat, label)
            loss.backward()
            # for i in self.parameters():
            #     print(i.grad)
            loss_optimizer.step()
            loss_optimizer.zero_grad()
            model_optimizer.step()
            model_optimizer.zero_grad()
            loss_all.append(loss.item())
        #     print('loss', loss.item())
        # print('loss', np.mean(loss_all))

    def test_loop(self, test_loader , fix_shreshold = None, mode = 'test'):
        self.feature_extractor.eval()
        all_prob = {}
        all_meta = {}
        for i, (pos_sup, neg_sup, query, seg_len, seg_hop, query_start, query_end, label) in enumerate(test_loader):
            seg_hop = seg_hop.item()
            query_start = query_start.item()

            # print(pos_sup[1].squeeze().shape)
            # print(neg_sup[1].squeeze().shape)
            # print(query.shape)
            wav_file= pos_sup[0][0].split('&')[1]
            all_meta[wav_file]={}
            all_meta[wav_file]['start'] = query_start

            all_meta[wav_file]['end'] = query_end
            all_meta[wav_file]['seg_hop'] = seg_hop
            all_meta[wav_file]['seg_len'] = seg_len
            
            all_meta[wav_file]['label'] = label[0]

            feat_file = os.path.splitext(os.path.basename(wav_file))[0] + '.hdf5'
            feat_file = os.path.join('/root/task5_2023/latent_feature/TNN_noMAML', feat_file)
            if os.path.isfile(feat_file):
                os.remove(feat_file)
        
            
            directory = os.path.dirname(feat_file)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # print(wav_file)
            # print(query_start)
            pos_data = pos_sup[1].squeeze()
            query = query.squeeze()
            query_dataset = TensorDataset(query, torch.zeros(query.shape[0]))
            query_loader = DataLoader(query_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
            pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            
            prob_mean = []
            for i in range(5):
                test_loop_neg_sample = self.config.val.test_loop_neg_sample
                neg_sup[1] = neg_sup[1].squeeze() 
                if neg_sup[1].shape[0] > test_loop_neg_sample:
                    neg_indices = torch.randperm(neg_sup[1].shape[0])[:test_loop_neg_sample]
                    neg_seg_sample = neg_sup[1][neg_indices]
                else:
                    neg_seg_sample = neg_sup[1]
                # neg_seg_sample = neg_sup[1]
                neg_dataset = TensorDataset(neg_seg_sample, torch.zeros(neg_seg_sample.shape[0]))
                neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                
                support_data = torch.cat([pos_data, neg_seg_sample], dim=0)
                
                #  support_data = pos_data
                m = pos_data.shape[0]
                n = neg_seg_sample.shape[0]
                support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))
                support_label = torch.from_numpy(support_label).long().to(self.device)
                
                
                support_feats = self.feature_extractor(support_data)
                # with h5py.File(feat_file, 'w') as f:
                #     f.create_dataset("features", (0, 512), maxshape=(None, 512))
                #     f.create_dataset("labels", data=label.squeeze(0).numpy())
                #     f.create_dataset("features_t", data = support_feats.detach().cpu().numpy())
                #     f.create_dataset("labels_t", data=support_label.cpu().numpy())
                            
                
                
      
                
                pos_feat  = []
                for batch in pos_loader:
                    p_data, _ = batch
                    feat = self.feature_extractor.forward(p_data)
                    # print(feat.shape)
                    pos_feat.append(feat.mean(0))
                pos_feat = torch.stack(pos_feat, dim=0).mean(0)
                neg_feat = []

                with torch.no_grad():
                    for batch in neg_loader:
                        n_data, _ = batch
                        # print(neg_data.shape)
                        feat = self.feature_extractor.forward(n_data)
                        # print(feat.shape)
                        neg_feat.append(feat.mean(0))
                    neg_feat = torch.stack(neg_feat, dim=0).mean(0) 
                    proto = torch.stack([pos_feat,neg_feat], dim=0)
                    
                    
                
                prob_all = []
                for batch in query_loader:
                    query_data, _ = batch
                    prob, feats  = self.feed_forward_test(proto, query_data)
                    prob_all.append(prob)
                    # with h5py.File(feat_file, 'a') as f:
                            
                    #         size = f['features'].shape[0]
                    #         nwe_size = f['features'].shape[0] + feats.shape[0]

                    #         f['features'].resize((nwe_size, 512))

                    #         f['features'][size:nwe_size] = feats.detach().cpu().numpy()
                prob_all = np.concatenate(prob_all, axis=0)
                #########################################################################
                  
                prob_all = prob_all[:,0]
                # prob_all = np.where(prob_all>self.config.val.threshold, 1, 0)
                prob_mean.append(prob_all)
            prob_mean = np.stack(prob_mean, axis=0).mean(0)
            all_prob[wav_file] = prob_mean
        
        best_res = None
        best_f1 = 0
        best_report = {}
        best_threshold = 0
        for threshold in np.arange(0.5, 1, 0.1):
            if fix_shreshold is not None:
                threshold = fix_shreshold
            report_f1 = {}
            all_time = {'Audiofilename':[], 'Starttime':[], 'Endtime':[]}
            for wav_file in all_prob.keys():
                
                prob = np.where(all_prob[wav_file]>threshold, 1, 0)

                # acc = np.sum(prob^1 == np.array(all_meta[wav_file]['label']))/len(prob)
                # 计算分类报告
                y_pred = prob^1
                y_true =  np.array(all_meta[wav_file]['label'])
 
                # print(all_meta[wav_file]['seg_hop'])
                # print(all_meta[wav_file]['seg_len'])
                # 输出分类报告

                report_f1[os.path.basename(wav_file)] = classification_report(y_true, y_pred,zero_division=0, digits=5)
                # 计算各个类别的F1分数
                # f1_scores = f1_score(y_true, y_pred, average=None)

                # 输出各个类别的F1分数
                # print("F1 scores for each class:")
                # print(f1_scores)
                # print(len(prob))
                # print(np.sum(prob))
                # print(np.sum(prob)/len(prob))


                on_set = np.flatnonzero(np.diff(np.concatenate(([0],prob), axis=0))==1)
                off_set = np.flatnonzero(np.diff(np.concatenate((prob,[0]), axis=0))==-1) + 1 #off_set is the index of the first 0 after 1
                # for i, j in zip(on_set, off_set):
                #     print(i,j)
                
                
                on_set_time = on_set*all_meta[wav_file]['seg_hop']/self.fps + all_meta[wav_file]['start']
                off_set_time = off_set*all_meta[wav_file]['seg_hop']/self.fps + all_meta[wav_file]['start']
                all_time['Audiofilename'].extend([os.path.basename(wav_file)]*len(on_set_time))
                all_time['Starttime'].extend(on_set_time)
                all_time['Endtime'].extend(off_set_time)
                # print(wav_file)
                # print(on_set_time[:5])
                # print('query_start', all_meta[wav_file]['start'])
                for i in range(len(off_set_time)):
                    if off_set_time[i] > all_meta[wav_file]['end']:
                        raise ValueError('off_set_time is larger than query_end')

            df_all_time = pd.DataFrame(all_time)
            df_all_time = post_processing(df_all_time, self.config, mode )
            df_all_time = df_all_time.astype('str')
            pred_path = normalize_path(self.config.checkpoint.pred_dir)
            pred_path = os.path.join(pred_path, 'pred_{:.2f}.csv'.format(threshold))
            if not os.path.exists(os.path.dirname(pred_path)):
                os.makedirs(os.path.dirname(pred_path))
            df_all_time.to_csv(pred_path, index=False)
            
            ref_files_path = test_loader.dataset.val_dir
            report_dir = normalize_path(self.config.checkpoint.report_dir)
            report = evaluate(df_all_time, ref_files_path, self.config.team_name, self.config.dataset, report_dir)
            if report['overall_scores']['fmeasure (percentage)'] > best_f1:
                best_f1 = report['overall_scores']['fmeasure (percentage)']
                best_res = report
                best_report = report_f1
                best_threshold = threshold
            if fix_shreshold is not None:
                break
        for i in best_report.keys():
            print(i)
            print(best_report[i])
            print('~~~~~~~~~~~~~~~')
        print(best_res)
        print('best_threshold', best_threshold)
        print('~~~~~~~~~~~~~~~')
        return df_all_time, best_res, best_threshold