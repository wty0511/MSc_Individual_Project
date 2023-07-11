# This code is modified from https://github.com/haoheliu/DCASE_2022_Task_5
# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
# This code is modified from https://github.com/phlippe/uvadlc_notebooks
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
from src.utils.class_pair_dataset import *
from pytorch_metric_learning import losses

from copy import deepcopy
import random
import torch.optim as optim
from src.utils.sampler import *
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2, dim=1)
        # print(cos_sim)
        
        # print(torch.sum(label==0))
        losses =(1 - label) * torch.pow((1-cos_sim),2.0) + \
                 label * torch.pow(cos_sim, 2) * (cos_sim > self.margin).float()
                
        loss = torch.mean(losses)
        # print('loss', loss)
        return loss
    
    
class SNNMAML(BaseModel):
    def __init__(self, config):
        super(SNNMAML, self).__init__(config)
        
        self.test_loop_batch_size = config.val.test_loop_batch_size
        self.loss_fn = ContrastiveLoss(margin = 0.3)
        # self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        self.approx = True
        self.ce = nn.CrossEntropyLoss()
    def get_contrastive_pairs(self, data, labels):
        
        data1 = []
        data2 = []
        pair_labels = []
        # print(labels)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    data1.append(data[i])
                    data2.append(data[j])
                    pair_labels.append(0)
                else:
                    data1.append(data[i])
                    data2.append(data[j])
                    pair_labels.append(1)
                    # print(0)
        
        data1 = torch.stack(data1, dim=0)
        data2 = torch.stack(data2, dim=0)
        pair_labels = torch.tensor(pair_labels).long().to(self.device)
        return data1, data2, pair_labels

    def inner_loop(self, support_data, support_label = None, mode = 'train'):

        fast_parameters = list(self.feature_extractor.parameters())
        for weight in self.feature_extractor.parameters():
            weight.fast = None
        self.feature_extractor.zero_grad()
        
        sampler = IntClassSampler(self.config, support_label, 10)
        dataset =  PairDataset(self.config, support_data, support_label, debug = False)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size = 10)
        # self.config.train.lr_inner = 0.01
        for i in range(5):
            for batch in dataloader:
                data, lable = batch
                anchor, pos, neg = data
                anchor_feat = self.feature_extractor(anchor)
                pos_feat = self.feature_extractor(pos)
                neg_feat = self.feature_extractor(neg)
                anchor = F.normalize(anchor_feat, dim=1)
                pos_feat = F.normalize(pos_feat, dim=1)
                neg_feat = F.normalize(neg_feat, dim=1)
                
                loss = self.loss_fn(anchor_feat, pos_feat, torch.zeros(anchor_feat.shape[0]).long().to(self.device))
                # print('inner loop: loss:{:.3f}'.format(loss.item()))
                loss += self.loss_fn(anchor_feat, neg_feat, torch.ones(anchor_feat.shape[0]).long().to(self.device))
                # print(loss.item())
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                if self.approx:
                    grad = [ g.detach()  for g in grad ] 
                fast_parameters = []
                
                for k, weight in enumerate(self.parameters()):
                    #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                    if weight.fast is None:
                        weight.fast = weight - self.config.train.lr_inner * grad[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - self.config.train.lr_inner * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                    fast_parameters.append(weight.fast) # update the fast_parameters
                loss = loss.detach()
                # print(loss)
                # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

        # print('~~~~~~~~~~~~~~~')
        # loss = torch.stack(losses, dim=0).mean()
        # print('inner loop: loss:{:.3f}'.format(loss.item()))
        # print('~~~~~~~~~~~~~~~')
        
        # data1, data2, pair_labels= self.get_contrastive_pairs(support_data, support_label)
        # data_all = torch.stack([data1, data2], dim=1)
        # # print(data_all.shape)
        # dataset = TensorDataset(data_all, pair_labels)
        # data_loader = DataLoader(dataset, batch_size= 256, shuffle=False)
        # for i in range(self.config.train.inner_step):
        #     for batch in data_loader:
        #         data, label = batch
        #         data1 = data[:, 0, :, :]
        #         data2 = data[:, 1, :, :]
        #         # print(data1.shape)
        #         # print(data2.shape)
        #         feat1 = self.feature_extractor(data1)
        #         feat2 = self.feature_extractor(data2)
        #         # print(len(support_data))
        #         feat1 = F.normalize(feat1, dim=1)
        #         feat2 = F.normalize(feat2, dim=1)
        #         # print(len(pair_labels))
        #         loss = self.loss_fn(feat1, feat2, label)
        #         # print('loss', loss)

        #         grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
        #         if self.approx:
        #             grad = [ g.detach()  for g in grad ] 
        #         fast_parameters = []
                
        #         for k, weight in enumerate(self.parameters()):
        #             # print(grad[k])
        #             #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
        #             if weight.fast is None:
        #                 weight.fast = weight - self.config.train.lr_inner * grad[k] #create weight.fast 
        #             else:
        #                 weight.fast = weight.fast - self.config.train.lr_inner * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
        #             fast_parameters.append(weight.fast) # update the fast_parameters
        #         loss = loss.detach()
        #     # print(loss)
        #     # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        
        # # print('~~~~~~~~~~~~~~~')
        if mode != 'train':
            print('inner loop: loss:{:.3f}'.format(loss.item()))
        if mode != 'train':    
            print('!!!!!!!!!')
        return 
     
    # def inner_loop_test(self, support_data, neg_loader, support_label = None,  mode = 'train'):
    #     fast_parameters = list(self.feature_extractor.parameters())
    #     for weight in self.feature_extractor.parameters():
    #         weight.fast = None
    #     self.feature_extractor.zero_grad()
        
    #     sampler = IntClassSampler(self.config, support_label, 50, mode = 'test')
    #     dataset =  PairDataset(self.config, support_data, support_label, debug = False)
    #     dataloader = DataLoader(dataset, sampler=sampler, batch_size = 50)
    #     # self.config.train.lr_inner = 0.01
    #     for i in range(5):
    #         for batch in dataloader:
    #             neg_feat = []

    #             for b in neg_loader:
    #                 n_data, _ = b
    #                 # print(neg_data.shape)
    #                 feat = self.feature_extractor.forward(n_data)
    #                 # print(feat.shape)
    #                 neg_feat.append(feat)
    #             neg_feat = torch.cat(neg_feat, dim=0).mean(0)
                
    #             data, lable = batch
    #             anchor, pos, neg = data
    #             anchor_feat = self.feature_extractor(anchor)
    #             pos_feat = self.feature_extractor(pos)
    #             # neg_feat = self.feature_extractor(neg)
    #             loss = self.loss_fn(anchor_feat, pos_feat, torch.zeros(anchor_feat.shape[0]).long().to(self.device))
    #             # print('inner loop: loss:{:.3f}'.format(loss.item()))
    #             # loss += self.loss_fn(anchor_feat, neg_feat, torch.ones(anchor_feat.shape[0]).long().to(self.device))
    #             # print(loss.item())
    #             grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
    #             if self.approx:
    #                 grad = [ g.detach()  for g in grad ] 
    #             fast_parameters = []
                
    #             for k, weight in enumerate(self.parameters()):
    #                 #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
    #                 if weight.fast is None:
    #                     weight.fast = weight - self.config.train.lr_inner * grad[k] #create weight.fast 
    #                 else:
    #                     weight.fast = weight.fast - self.config.train.lr_inner * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
    #                 fast_parameters.append(weight.fast) # update the fast_parameters
    #             loss = loss.detach()
    #             # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #     # print('~~~~~~~~~~~~~~~')
    #     # loss = torch.stack(losses, dim=0).mean()
    #     # print('inner loop: loss:{:.3f}'.format(loss.item()))
    #     # print('~~~~~~~~~~~~~~~')
    #     if mode != 'train':
    #         print('inner loop: loss:{:.3f}'.format(loss.item()))
    #     if mode != 'train':    
    #         print('!!!!!!!!!')
    #     return 
    
    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        query = F.normalize(query, dim=1)
        support = F.normalize(support, dim=1)
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)
        return F.cosine_similarity(query, support, dim=2)


    def feed_forward(self, support_data, query_data):
        # # Execute a model with given output layer weights and inputs
        support_feat = self.feature_extractor(support_data)
        # # nway 不是2nway要注意
        support_feat = self.split_1d(support_feat)
        prototype = support_feat.mean(0)
        query_feat = self.feature_extractor(query_data)
        
        dists = self.euclidean_dist(query_feat, prototype)
        # print(dists)
        scores = dists

        preds = scores.argmax(dim=1)
        y_query = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
        acc = torch.eq(preds, y_query).float().mean()
        # print(acc)
        
        loss = self.ce(scores, y_query)
        # print(loss)
        y_query = y_query.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        report = classification_report(y_query, preds,zero_division=0, digits=3)
        # print(report)
        
        # query_label = np.tile(np.arange(self.n_way),self.n_query)
        # sampler = IntClassSampler(self.config, query_label, 100)
        # dataset =  PairDataset(self.config, query_data, query_label, debug = False)
        # dataloader = DataLoader(dataset, sampler=sampler, batch_size = 32)
        # loss = []
        # for batch in dataloader:
        #     data, lable = batch
        #     anchor, pos, neg = data
        #     # print(anchor.shape)
        #     # print(pos.shape)
        #     # print(neg.shape)
        #     anchor_feat = self.feature_extractor(anchor)
        #     pos_feat = self.feature_extractor(pos)
        #     neg_feat = self.feature_extractor(neg)
        #     loss_temp= self.loss_fn(anchor_feat, pos_feat, torch.zeros(anchor_feat.shape[0]).long().to(self.device))
        #     # print('inner loop: loss:{:.3f}'.format(loss.item()))
        #     loss_temp += self.loss_fn(anchor_feat, neg_feat, torch.ones(anchor_feat.shape[0]).long().to(self.device))
 
        #     loss.append(loss_temp)
        #     # print('inner loop: loss:{:.3f}'.format(loss.item()))
        #     # loss += self.loss_fn(anchor_feat, neg_feat, torch.ones(anchor_feat.shape[0]).long().to(self.device))
        
        # loss = torch.mean(torch.stack(loss))

        # report = None
        # acc = torch.tensor(0.0).to(self.device)
        return loss, report, acc
        

    
    def feed_forward_test(self,  prototype, query_data):
        # Execute a model with given output layer weights and inputs

        query_feat = self.feature_extractor(query_data)
        dists = self.euclidean_dist(query_feat, prototype)

        pred = dists.argmin(-1)
        
        scores = dists
        preds = F.softmax(scores, dim = 1)
        preds = preds.detach().cpu().numpy()
        return preds
    
    def outer_loop(self, data_loader, mode = 'train', opt = None):
        loss_epoch = []
        for i, task_batch in tqdm(enumerate(data_loader)):
            accuracies = []
            losses = []
            self.feature_extractor.zero_grad()
            loss_all = []
            for task in task_batch:
                if not self.config.train.neg_prototype:
                    pos_data = task 
                    classes, data_pos, _ =pos_data
                    # support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
                    support_data, query_data = self.split_support_query_data(data_pos, None)
                    support_label = np.tile(np.arange(self.n_way),self.n_support)

                else:
                    raise ValueError('neg_prototype must be False')
                
                self.inner_loop(support_data, support_label, mode = 'train')
                
                # print(support_data.shape)
                # print(query_data.shape)
                loss, _, acc = self.feed_forward(support_data, query_data)
                loss_all.append(loss)
                opt.zero_grad()
            
            loss_q = torch.stack(loss_all).sum(0)
            loss_q.backward()
            # for name, param in self.feature_extractor.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad)
            opt.step()
            opt.zero_grad()
            print('outer loop: loss:{:.3f}'.format(loss_q.item()/len(task_batch)))
            loss_epoch.append(loss_q.item()/len(task_batch))
            
    
        return np.mean(loss_epoch)       
    
    
    
    def train_loop(self, data_loader, optimizer):
        
        return self.outer_loop(data_loader, mode = 'train', opt = optimizer)
    
    
    def test_loop(self, test_loader ,fix_shreshold= None, mode = 'test'):
        best_res_all = []
        all_loss = []
        best_threshold_all = []
        for i in range(1):
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
                query_loader = DataLoader(query_dataset, batch_size=256, shuffle=False)
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
                    # support_data = pos_data
                    m = pos_data.shape[0]
                    n = neg_seg_sample.shape[0]
                    support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))
                    # support_label = torch.from_numpy(support_label).long().to(self.device)
                    # support_label = np.zeros((m,))
                    # support_label = torch.from_numpy(support_label).long().to(self.device)
                    self.inner_loop(support_data, support_label, mode = 'test')
                    pos_feat  = []
                    for batch in pos_loader:
                        p_data, _ = batch
                        feat = self.feature_extractor(p_data)
                        # print(feat.shape)
                        pos_feat.append(feat.mean(0))
                    pos_feat = torch.stack(pos_feat, dim=0).mean(0)
                
                    neg_feat = []
                    for batch in neg_loader:
                        n_data, _ = batch
                        # print(neg_data.shape)
                        feat = self.feature_extractor(n_data)
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
                    temp_prob = torch.from_numpy(prob_all).to(self.device)
                    # print(all_meta[wav_file]['label'])
                    pos_num =torch.sum(all_meta[wav_file]['label']==0)
                    neg_num = torch.sum(all_meta[wav_file]['label']==1)
                    
                    loss = F.cross_entropy(temp_prob,all_meta[wav_file]['label'].to(self.device),weight = torch.tensor([neg_num/pos_num, 1]).to(self.device)).to(self.device)
                    print('loss', loss.item())
                    all_loss.append(loss.detach().cpu().numpy())
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
                df_all_time = post_processing(df_all_time, self.config, mode)
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
            # for i in best_report.keys():
            #     print(i)
            #     print(best_report[i])
            #     print('~~~~~~~~~~~~~~~')
            # print(best_res)
            # print('best_threshold', best_threshold)
            # print('~~~~~~~~~~~~~~~')
            best_threshold_all.append(best_threshold)
            best_res_all.append(best_res)
        print('losses', np.mean(all_loss))
        print(self.average_res(best_res_all))
        print(np.mean(best_threshold_all))
        return df_all_time, self.average_res(best_res_all), np.mean(best_threshold_all) , np.mean(all_loss)
    
    
    
    

    
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
    
    def average_res (self, res_list):
        avg = deepcopy(res_list[0])
    
        for key in avg.keys():
            if key == 'team_name' or key == 'dataset' or key == 'set_name' or key == 'report_date':
                continue
            if key == 'overall_scores':
                for key2 in avg[key].keys():
                    avg[key][key2] = 0
            if key == 'scores_per_subset':
                for subset in avg[key].keys():
                    for key2 in avg[key][subset].keys():
                        avg[key][subset][key2] = 0
            if key == 'scores_per_audiofile':
                for file in avg[key].keys():
                    for key2 in avg[key][file].keys():
                        avg[key][file][key2] = 0
        
        
        for res in res_list:
            for key in res.keys():
                if key == 'team_name' or key == 'dataset' or key == 'set_name' or key == 'report_date':
                    continue
                if key == 'overall_scores':
                    for key2 in res[key].keys():
                        avg[key][key2] += res[key][key2]/len(res_list)
                if key == 'scores_per_subset':
                    for subset in res[key].keys():
                        for key2 in res[key][subset].keys():
                            avg[key][subset][key2] += res[key][subset][key2]/len(res_list)
                if key == 'scores_per_audiofile':
                    for file in res[key].keys():
                        for key2 in res[key][file].keys():
                            avg[key][file][key2] += res[key][file][key2]/len(res_list)
        return avg
            