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
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.PairwiseDistance(p=2)  # 欧氏距离计算
        distances = euclidean_distance(output1, output2)
        losses = 0.5 * (1 - label) * torch.pow(distances, 2) + \
                 0.5 * label * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        loss = torch.mean(losses)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # print(anchor.shape)
        # print(positive.shape)
        # print(negative.shape)
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        # print(distance_positive)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        # print(distance_negative)
        # print('~~~~~~~~~~~~~')
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TriNet(BaseModel):
    def __init__(self, config):
        super(TriNet, self).__init__(config)
        
        self.test_loop_batch_size = config.val.test_loop_batch_size
        self.loss_fn = TripletLoss(margin=10.0)
        self.approx = True
        self.ce = nn.CrossEntropyLoss()
        self.cosloss = nn.CosineEmbeddingLoss(margin= 0.95)
    # def inner_loop(self, support_data, support_label = None, mode = 'train'):
    #     local_model = deepcopy(self.feature_extractor)
    #     local_model.train()
    #     local_optim = optim.SGD(local_model.parameters(), self.config.train.lr_inner, momentum = self.config.train.momentum, weight_decay=self.config.train.weight_decay) 
    #     local_optim.zero_grad()
    #     fast_parameters = list(local_model.parameters())
    #     if mode == 'test':
    #         support_label = support_label.cpu().numpy()
    #     for i in range(100):
    #         if mode == 'train':
    #             label = random.randint(0, 1)
    #             # same class
    #             if label == 0:               
    #                 class1 = random.randint(0, self.n_way - 1)
    #                 indices1 = [i for i in range(len(support_data)) if i % self.n_way == class1]
    #                 index1, index2 = random.sample(indices1, 2)

    #                 sampl1 = support_data[index1]
    #                 sampl2 = support_data[index2]

    #             else:
    #                 class1, class2 = random.sample(range(self.n_way), 2)
                    
    #                 indices1 = [i for i in range(len(support_data)) if i % self.n_way == class1]
    #                 indices2 = [i for i in range(len(support_data)) if i % self.n_way == class2]
                    
    #                 # Randomly select two different indices
    #                 index1 = random.sample(indices1, 1)
    #                 index2 = random.sample(indices2, 1)
                    
    #                 sampl1 = support_data[index1[0]]
    #                 sampl2 = support_data[index2[0]]


    #         else:


    #             label = random.randint(0, 1)
    #             # same class
    #             if label == 0:               
    #                 class1 = random.randint(0, 1)
    #                 indices1 = np.where(support_label == class1)[0].tolist()
    #                 index1, index2 = random.sample(indices1, 2)
                    
    #                 sampl1 = support_data[index1]
    #                 sampl2 = support_data[index2]
    #             else:
    #                 class1 = 0
    #                 class2 = 1
                                        
    #                 indices1 = np.where(support_label == class1)[0].tolist()

    #                 indices2 = np.where(support_label == class2)[0].tolist()

                    
    #                 # Randomly select two different indices
    #                 index1 = random.sample(indices1, 1)
    #                 index2 = random.sample(indices2, 1)
                    
    #                 sampl1 = support_data[index1[0]]
    #                 sampl2 = support_data[index2[0]]
    #         sampl1 = sampl1.unsqueeze(0)
    #         sampl2 = sampl2.unsqueeze(0)
    #         # print('label:{}'.format(label))
    #         # print(sampl1.shape)
    #         # print(sampl2.shape)
    #         # print('~~~~~~~~~~~')
    #         feat1 = local_model(sampl1)
    #         feat2 = local_model(sampl2)
    #         loss = self.loss_fn(feat1, feat2, label)
    #         if self.approx:
    #             grad = torch.autograd.grad(loss, fast_parameters)
    #         else:
    #             grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
    #         for k, weight in enumerate(local_model.parameters()):
    #             # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
    #             weight.grad = grad[k]
    #         local_optim.step()
    #         local_optim.zero_grad()
    #         loss = loss.detach()
    #     if mode != 'train':
    #         print('inner loop: loss:{:.3f}'.format(loss.item()))
    #     if mode != 'train':    
    #         print('!!!!!!!!!')
    #     return local_model 
    
    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)

        return torch.pow(query - support, 2).sum(2)

    def cossim(self, query, support):
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        n = query.size(0)
        m = support.size(0)
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)
        return cos_sim(query, support)

    def feed_forward(self, local_model, support_data, query_data):
        # # Execute a model with given output layer weights and inputs
        # support_feat = local_model(support_data)
        # # nway 不是2nway要注意
        # support_feat = self.split_1d(support_feat)
        # prototype = support_feat.mean(0)
        # query_feat = local_model(query_data)
        # dists = self.euclidean_dist(query_feat, prototype)

        # scores = -dists

        # preds = scores.argmax(dim=1)
        # y_query = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
        # acc = torch.eq(preds, y_query).float().mean()

        
        # loss = self.ce(scores, y_query)
        # y_query = y_query.cpu().numpy()
        # preds = preds.detach().cpu().numpy()
        # report = classification_report(y_query, preds,zero_division=0, digits=3)
        # print(report)
        loss = 0
        for i in range(100):
            label = random.randint(0, 1)
            # same class
            if label == 0:               
                class1 = random.randint(0, self.n_way - 1)
                indices1 = [i for i in range(len(query_data)) if i % self.n_way == class1]
                index1, index2 = random.sample(indices1, 2)

                sampl1 = query_data[index1]
                sampl2 = query_data[index2]

            else:
                class1, class2 = random.sample(range(self.n_way), 2)
                
                indices1 = [i for i in range(len(query_data)) if i % self.n_way == class1]
                indices2 = [i for i in range(len(query_data)) if i % self.n_way == class2]
                
                # Randomly select two different indices
                index1 = random.sample(indices1, 1)
                index2 = random.sample(indices2, 1)
                
                sampl1 = query_data[index1[0]]
                sampl2 = query_data[index2[0]]
            sampl1 = sampl1.unsqueeze(0)
            sampl2 = sampl2.unsqueeze(0)
            # print('label:{}'.format(label))
            # print(sampl1.shape)
            # print(sampl2.shape)
            # print('~~~~~~~~~~~')
            feat1 = local_model(sampl1)
            feat2 = local_model(sampl2)
            loss += self.loss_fn(feat1, feat2, label)
        loss /= 100
        report = None
        acc = torch.tensor(0.0).to(self.device)
        return loss, report, acc
    
    
    def feed_forward_test(self, prototype, query_data):
        # Execute a model with given output layer weights and inputs

        query_feat = self.feature_extractor(query_data)
        dists = self.euclidean_dist(query_feat, prototype)
        # dists = self.cossim(query_feat, prototype)

        pred = dists.argmin(-1)
        
        scores = -dists
        preds = F.softmax(scores, dim = 1)
        preds = preds.detach().cpu().numpy()

        return preds
    



        
                    
                    
    def train_loop(self, data_loader, optimizer):
        self.feature_extractor.train()
        for i, pair_batch in tqdm(enumerate(data_loader)):
            data, labels = pair_batch
            anchor, pos, neg = data
            self.feature_extractor.zero_grad()
            anchor_feat = self.feature_extractor(anchor)
            pos_feat = self.feature_extractor(pos)
            neg_feat = self.feature_extractor(neg)
            # loss = self.loss_fn(anchor_feat, pos_feat, torch.zeros(anchor.shape[0]).to(self.device))
            # loss = loss + self.loss_fn(anchor_feat, neg_feat, torch.ones(anchor.shape[0]).to(self.device))
            # loss = self.cosloss(anchor_feat, pos_feat, torch.ones(anchor.shape[0]).to(self.device)) + self.cosloss(anchor_feat, neg_feat, -torch.ones(anchor.shape[0]).to(self.device))
            loss = self.loss_fn(anchor_feat, pos_feat, neg_feat)
            loss.backward()
            optimizer.step()
            print('loss:{:.3f}'.format(loss.item()))

    def test_loop(self, test_loader , fix_shreshold = None):
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
            # print(wav_file)
            # print(query_start)
            pos_data = pos_sup[1].squeeze()
            query = query.squeeze()
            query_dataset = TensorDataset(query, torch.zeros(query.shape[0]))
            query_loader = DataLoader(query_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
            pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            
            prob_mean = []
            for i in range(1):
                test_loop_neg_sample = self.config.val.test_loop_neg_sample
                neg_sup[1] = neg_sup[1].squeeze() 
                
                # if neg_sup[1].shape[0] > test_loop_neg_sample:
                #     neg_indices = torch.randperm(neg_sup[1].shape[0])[:test_loop_neg_sample]
                #     neg_seg_sample = neg_sup[1][neg_indices]
                # else:
                #     neg_seg_sample = neg_sup[1]
                neg_seg_sample = neg_sup[1]
                neg_dataset = TensorDataset(neg_seg_sample, torch.zeros(neg_seg_sample.shape[0]))
                neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                

                
                support_data = torch.cat([pos_data, neg_seg_sample], dim=0)
                # support_data = pos_data
                m = pos_data.shape[0]
                n = neg_seg_sample.shape[0]
                support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))
                support_label = torch.from_numpy(support_label).long().to(self.device)
                
                
                pos_feat  = []
                for batch in pos_loader:
                    p_data, _ = batch
                    feat = self.feature_extractor.forward(p_data)
                    # print(feat.shape)
                    pos_feat.append(feat.mean(0))
                pos_feat = torch.stack(pos_feat, dim=0).mean(0)
            
                neg_feat = []
                for batch in neg_loader:
                    n_data, _ = batch
                    # print(neg_data.shape)
                    feat = self.feature_extractor.forward(n_data)
                    # print(feat.shape)
                    neg_feat.append(feat.mean(0))
                neg_feat = torch.stack(neg_feat, dim=0).mean(0) 
                proto = torch.stack([pos_feat,neg_feat], dim=0)
                
                
                
                prob_all = []
                for batch in tqdm(query_loader):
                    query_data, _ = batch
                    prob = self.feed_forward_test(proto, query_data)
                    prob_all.append(prob)
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
        for threshold in np.arange(0.5, 1, 0.05):
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
            df_all_time = post_processing(df_all_time)
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
    
    
    
    
    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)

        return torch.pow(query - support, 2).sum(2)

    
    # def euclidean_dist(self,query, support):
    #     init_weight = 2 * support

    #     init_bias = -torch.norm(support, dim=1)**2
        # print(init_weight.shape)
        # print(init_bias.shape)
        
        # output_weight = init_weight.detach()
        # output_bias = init_bias.detach()
        # score2 = query.matmul(init_weight.t()) + init_bias-(torch.norm(query, dim=1)**2).unsqueeze(1)
        
        # score2 = F.linear(query, init_weight, init_bias)
        # score2 = score2-(torch.norm(query, dim=1)**2).unsqueeze(1)

        # n = query.size(0)
        # m = support.size(0)
        
        # query = query.unsqueeze(1).expand(n, m, -1)
        # support = support.unsqueeze(0).expand(n, m, -1)
        # score = torch.pow(query - support, 2).sum(2)

        # prob2 = F.softmax(score2, dim=1)
        # prob1 = F.softmax(-score, dim=1)

        # print(torch.argmax(prob1, dim=1))
        # print(torch.argmax(prob2, dim=1))
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # return -score2
    
    