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
from copy import deepcopy
import random
import torch.optim as optim
from src.utils.sampler import *
import h5py
from pytorch_metric_learning import  losses,reducers

from pytorch_metric_learning.distances import LpDistance
from src.models.triplet_loss import *


    
class TNNMAML(BaseModel):
    def __init__(self, config):
        super(TNNMAML, self).__init__(config)
        
        self.test_loop_batch_size = config.val.test_loop_batch_size
        # self.loss_fn = TripletLoss(margin=0.1)
        # self.loss_fn = losses.TripletMarginLoss(margin=0.2,
        #                 swap=False,
        #                 smooth_loss=False,
        #                 triplets_per_anchor= 'all',
        #                 distance = LpDistance(normalize_embeddings=True, p=2, power=2))
        # self.loss_fn = TripletLoss(margin= self.config.train.margin)
        self.loss_fn = TripletLossHard(margin= self.config.train.margin)
        self.approx = True
        self.ce = nn.CrossEntropyLoss()
    def inner_loop(self, support_data, support_label = None, mode = 'train'):
        
        # self.config.train.lr_inner = 0.01
        fast_parameters = list(self.feature_extractor.parameters())
        for weight in self.feature_extractor.parameters():
            weight.fast = None
        self.feature_extractor.zero_grad()
        support_label = torch.from_numpy(support_label).long().to(self.device)
        
        for i in range(self.config.train.inner_step):
            break
            feat = F.normalize(self.feature_extractor(support_data), dim=1)
            loss = self.loss_fn(feat, support_label)
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
                # print(len(fast_parameters))
            loss = loss.detach()
            # print('inner loop: loss:{:.3f}'.format(loss.item()))
                # print('inner loop: loss:{:.3f}'.format(loss.item()))
        # print('inner loop: loss:{:.3f}'.format(loss.item()))
        
        
        # print('!!!!!!!')
        # if mode != 'train':
        #     print('inner loop: loss:{:.3f}'.format(loss.item()))
        if mode != 'train':    
            print('!!!!!!!!!')
        return
    
    
    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)
        return torch.pow(query - support, 2).sum(2)



    def feed_forward(self, support_data, query_data):
        # # Execute a model with given output layer weights and inputs
        support_feat = self.feature_extractor(support_data)
        # nway 不是2nway要注意
        support_feat = self.split_1d(support_feat)
        prototype = support_feat.mean(0)
        query_feat = self.feature_extractor(query_data)
        dists = self.euclidean_dist(query_feat, prototype)

        scores = -dists
        
        preds = scores.argmax(dim=1)
        # print(dists)
        y_query = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
        acc = torch.eq(preds, y_query).float().mean()
        # print(acc)
        loss = self.ce(scores, y_query)
        # print(loss)
        y_query = y_query.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        report = classification_report(y_query, preds,zero_division=0, digits=3)
        return loss, report, acc
        
    
    
    def feed_forward_test(self,  prototype, query_data):
        # Execute a model with given output layer weights and inputs
        
        query_feat = self.feature_extractor(query_data)
        query_feat = F.normalize(query_feat, dim=1)
        prototype = F.normalize(prototype, dim=1)
        dists = self.euclidean_dist(query_feat, prototype)

        pred = dists.argmin(-1)
        # print(dists)
        scores = -dists
        preds = F.softmax(scores, dim = 1)
        preds = preds.detach().cpu().numpy()
        return preds, query_feat
    

    
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
                    pos_data, neg_data = task 
                    classes, data_pos, _ =pos_data
                    _, data_neg, _ =neg_data
                    # support_feat, query_feat = self.split_support_query_feature(data_pos, data_neg, is_data = True)
                    support_data, query_data = self.split_support_query_data(data_pos, data_neg)
                    support_label = np.tile(np.arange(self.n_way*2),self.n_support)
                # print(support_data.shape)
                # print(support_label)
                self.inner_loop(support_data, support_label, mode = 'train')
                
                loss, _, acc = self.feed_forward(support_data, query_data)
                accuracies.append(acc)
                # print(acc)
                # print('test: loss:{:.3f}'.format(loss.item()))
                loss_all.append(loss)

                opt.zero_grad()
                # print('outer loop: loss:{:.3f}'.format(loss.item()))
                
            loss_q = torch.stack(loss_all).sum(0)
            loss_q.backward()
            loss_epoch.append(loss_q.item()/len(task_batch))
            print('outer loop: acc:{:.3f}'.format(torch.stack(accuracies).mean().item()))
            # for i in self.parameters():
            #     print(i.grad)
            # for name, param in self.feature_extractor.named_parameters():
            #     print(name, param.grad)
            opt.step()
            opt.zero_grad()
            print('outer loop: loss:{:.3f}'.format(loss_q.item()/len(task_batch)))
        return np.mean(loss_epoch)
            
    def train_loop(self, data_loader, optimizer):
        
        return self.outer_loop(data_loader, mode = 'train', opt = optimizer)

    def get_topk_sim(self, pos, neg):
        # 最不相似的k个
        
        distances = torch.sqrt(torch.sum((neg - pos.mean(dim=0).unsqueeze(0))**2, dim=(-1,-2)))
        similarity_scores = 1.0 / (1.0 + distances)  # 加1是为了防止除以零
        k= np.min([50, neg.size(0)])
        _, indices = torch.topk(similarity_scores, k= k)
        
        # indices[torch.linspace(0, indices.size(0) - 1, k).long()]
        # indices[-k:]
        # indices[:k]
        return  indices[torch.linspace(0, indices.size(0) - 1, k).long()]
    
    def test_loop(self, test_loader, fix_shreshold=None, mode = 'test'): 
        best_res_all = []
        best_threshold_all = []
        
        all_loss = []
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
                query_loader = DataLoader(query_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
                pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                
                prob_mean = []
                for i in range(3):
                    test_loop_neg_sample = self.config.val.test_loop_neg_sample
                    # test_loop_neg_sample = 200
                    neg_sup[1] = neg_sup[1].squeeze() 
                    
                    if neg_sup[1].shape[0] > test_loop_neg_sample:
                        neg_indices = torch.randperm(neg_sup[1].shape[0])[:test_loop_neg_sample]
                        neg_seg_sample = neg_sup[1][neg_indices]
                    else:
                        neg_seg_sample = neg_sup[1]
                    # neg_seg_sample = neg_sup[1]
                    # print(neg_seg_sample.shape)
                    neg_dataset = TensorDataset(neg_seg_sample, torch.zeros(neg_seg_sample.shape[0]))
                    neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
      
                    
                    
                    feat_file = os.path.splitext(os.path.basename(wav_file))[0] + '.hdf5'
                    feat_file = os.path.join('/root/task5_2023/latent_feature/TNN', feat_file)
                    if os.path.isfile(feat_file):
                        os.remove(feat_file)
                    dims = (len(query_dataset), 512)
                    
                    directory = os.path.dirname(feat_file)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                    

                    # pos_feat  = []
                    # for batch in pos_loader:
                    #     p_data, _ = batch
                    #     feat = self.feature_extractor(p_data)
                    #     # print(feat.shape)
                    #     pos_feat.append(feat)
                    # pos_feat = torch.cat(pos_feat, dim=0)
                    # neg_feat = []
                    # with torch.no_grad():
                    #     for batch in neg_loader:
                    #         n_data, _ = batch
                    #         # print(neg_data.shape)
                    #         feat = self.feature_extractor.forward(n_data)
                    #         # print(feat.shape)
                    #         neg_feat.append(feat)

                    # neg_feat = torch.cat(neg_feat, dim=0)
                    # print('pos',len(pos_feat))
                    # print('neg',len(neg_feat))
                                        
                    # print('pos',pos_feat.shape)
                    # print('neg',neg_feat.shape)
 
                    
                    neg_seg_sample_index = self.get_topk_sim(pos_data, neg_seg_sample)



                    # neg_seg_sample = neg_seg_sample[neg_seg_sample_index]
                    
                    
                    # pos_data = pos_data[:5]
                    # neg_seg_sample = neg_seg_sample[:5]
                    support_data = torch.cat([pos_data, neg_seg_sample], dim=0)
                    # support_dataset = TensorDataset(support_data, torch.zeros(support_data.shape[0]))
                    # support_loader = DataLoader(support_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                    
                    
                
                    support_feats = self.feature_extractor(support_data)
                    # support_data = pos_data
                    m = pos_data.shape[0]
                    n = neg_seg_sample.shape[0]
                    support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))
                    # support_label = torch.from_numpy(support_label).long().to(self.device)
                    # support_label = np.zeros((m,))
                    # support_label = torch.from_numpy(support_label).long().to(self.device)
                    
                    with h5py.File(feat_file, 'w') as f:
                        f.create_dataset("features", (0, 512), maxshape=(None, 512))
                        f.create_dataset("labels", data=label.squeeze(0).numpy())
                        f.create_dataset("features_t", data = support_feats.detach().cpu().numpy())
                        f.create_dataset("labels_t", data=support_label)
                        # support_data
                        
                    self.inner_loop(support_data, support_label, mode = 'test')

                    pos_feat  = []
                    for batch in pos_loader:
                        p_data, _ = batch
                        feat = self.feature_extractor(p_data)
                        # print(feat.shape)
                        pos_feat.append(feat.mean(0))
                    # pos = torch.cat(pos_feat, dim=0)
                    # print('pos',pos.shape)
                    pos_feat = torch.stack(pos_feat, dim=0).mean(0)
                    neg_feat = []
                    with torch.no_grad():
                        for batch in neg_loader:
                            n_data, _ = batch
                            # print(neg_data.shape)
                            feat = self.feature_extractor.forward(n_data)
                            # print(feat.shape)
                            neg_feat.append(feat.mean(0))
                    # neg = torch.cat(neg_feat, dim=0)
                    # print('neg',neg.shape)
                    neg_feat = torch.stack(neg_feat, dim=0).mean(0) 
                    proto = torch.stack([pos_feat,neg_feat], dim=0)
                    
                    
                    
                    prob_all = []
                    for batch in tqdm(query_loader):
                        query_data, _ = batch
                        prob, feats = self.feed_forward_test(proto, query_data)
                        prob_all.append(prob)
                        feats = feats.detach().cpu().numpy()
                        with h5py.File(feat_file, 'a') as f:
                            
                            size = f['features'].shape[0]
                            nwe_size = f['features'].shape[0] + feats.shape[0]

                            f['features'].resize((nwe_size, 512))

                            f['features'][size:nwe_size] = feats
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
                    
                    # if '21-ML_176488' not in wav_file:
                    #     continue
                    # print(wav_file)
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
                    # print(report_f1[os.path.basename(wav_file)])
                    # if 'DCASE2021-ML_190099' in wav_file:
                    #     print(all_prob[wav_file])
                    #     print(prob)
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
                print('pred_path')
                
                pred_path = os.path.join(pred_path, 'pred_test{:.2f}.csv'.format(threshold))
                print(pred_path)
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
            best_threshold_all.append(best_threshold)
            best_res_all.append(best_res)
            # for i in best_report.keys():
            #     print(i)
            #     print(best_report[i])
            #     print('~~~~~~~~~~~~~~~')
            # print(best_res)
            # print('best_threshold', best_threshold)
            # print('~~~~~~~~~~~~~~~')
        print(self.average_res(best_res_all))
        print(np.mean(best_threshold_all))
        print('losses', np.mean(all_loss))
        
        return df_all_time, self.average_res(best_res_all), np.mean(best_threshold_all), np.mean(all_loss)
    
    

    


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
            