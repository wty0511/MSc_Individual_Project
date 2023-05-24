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

class ProtoNet(BaseModel):
    def __init__(self, config):
        super(ProtoNet, self).__init__(config)
        
        self.test_loop_batch_size = config.val.test_loop_batch_size


    def inner_loop(self,support, query):
        # print(support['label'].view(self.n_support, self.n_way * 2, -1 ))
        prototype     = support.mean(0).squeeze()
        # query['label'] = query['label'].to(self.device)
        dists = self.euclidean_dist(query.view(-1, query.shape[-1]), prototype)
        # print(dists)
        pred = dists.argmin(-1)
        
        scores = -dists
        # print(scores)
        # print(pred)
        # print(query['label'].view(-1))
        y_query = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
        acc = torch.eq(pred, y_query).float().mean()
        # print(y_query)
        # print(pred)
        return self.loss_fn(scores, y_query ), acc


    def train_loop(self, data_loader, optimizer):
        avg_loss = 0

        
        for i, data in enumerate(data_loader):
            if i != 0:
                end = time.time()
                elapsed_time = end - start
                # print(f"代码运行时间：{elapsed_time:.2f} 秒")


            #print('waiting for data')
            if self.config.train.neg_prototype:
                data = data[0]
                pos_data, neg_data = data 
                _, data_pos, _ =pos_data
                _, data_neg, _ =neg_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, data_neg, is_data = True)
            else:
                pos_data = data[0]
                _, data_pos, _ =pos_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
            optimizer.zero_grad()
            loss, acc = self.inner_loop(support_feat, query_feat)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(i+1, len(data_loader), loss.item(), acc.item()))
            
            start = time.time()
            
            
            avg_loss = avg_loss + loss.item()

        avg_loss = avg_loss / len(data_loader)
        return avg_loss
    
    def test_loop(self, test_loader):
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
            all_meta[wav_file]['label'] = label[0]
            pos_data = pos_sup[1].squeeze()
            query = query.squeeze()
            pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
            query_dataset = TensorDataset(query, torch.zeros(query.shape[0]))

            pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
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

            prob_mean = []
            for i in range(10):
                neg_sup[1] = neg_sup[1].squeeze() 
                
                if neg_sup[1].shape[0] > self.config.val.test_loop_neg_sample:
                    neg_indices = torch.randperm(neg_sup[1].shape[0])[: self.config.val.test_loop_neg_sample]
                    neg_seg_sample = neg_sup[1][neg_indices]
                else:
                    neg_seg_sample = neg_sup[1]
                neg_dataset = TensorDataset(neg_seg_sample, torch.zeros(neg_seg_sample.shape[0]))
                neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                neg_feat = []
                for batch in neg_loader:
                    neg_data, _ = batch
                    # print(neg_data.shape)
                    feat = self.forward(neg_data)
                    # print(feat.shape)
                    neg_feat.append(feat.mean(0))
                neg_feat = torch.stack(neg_feat, dim=0).mean(0)
                ##############################################################################
                
                proto = torch.stack([pos_feat,neg_feat], dim=0)
                prob_all = []
                for batch in query_loader:
                    query_data, _ = batch
                    query_feat = self.forward(query_data)
                    dist = self.euclidean_dist(query_feat, proto)
                    scores = -dist
                    prob = F.softmax(scores, dim=1)
                    prob_all.append(prob.detach().cpu().numpy())
                ##############################################################################
                
                prob_all = np.concatenate(prob_all, axis=0)
                # print(prob_all)
                prob_all = prob_all[:,0]
                # prob_all = np.where(prob_all>self.config.val.threshold, 1, 0)
                prob_mean.append(prob_all)
            prob_mean = np.stack(prob_mean, axis=0).mean(0)
            all_prob[wav_file] = prob_mean
        
        best_res = None
        best_f1 = 0
        best_threshold = 0
        for threshold in np.arange(0.4, 1.0, 0.05):
            all_time = {'Audiofilename':[], 'Starttime':[], 'Endtime':[]}
            for wav_file in all_prob.keys():
                prob = np.where(all_prob[wav_file]>threshold, 1, 0)
                y_pred = prob^1
                y_true =  np.array(all_meta[wav_file]['label'])
                report = classification_report(y_true, y_pred,zero_division=0, digits=5)
                # print(os.path.basename(wav_file))
                # # 输出分类报告
                # print("Classification report:")
                # print(report)
                
                on_set = np.flatnonzero(np.diff(np.concatenate(([0],prob), axis=0))==1)
                
                off_set = np.flatnonzero(np.diff(np.concatenate((prob,[0]), axis=0))==-1) + 1 #off_set is the index of the first 0 after 1
                


                on_set_time = on_set*all_meta[wav_file]['seg_hop']/self.fps + all_meta[wav_file]['start']
                off_set_time = off_set*all_meta[wav_file]['seg_hop']/self.fps + all_meta[wav_file]['start']
                all_time['Audiofilename'].extend([os.path.basename(wav_file)]*len(on_set_time))
                all_time['Starttime'].extend(on_set_time)
                all_time['Endtime'].extend(off_set_time)
                
            
            df_all_time = pd.DataFrame(all_time)
            df_all_time = post_processing(df_all_time)
            df_all_time = df_all_time.astype('str')
            pred_path = normalize_path(self.config.checkpoint.pred_dir)
            pred_path = os.path.join(pred_path, 'pred_{}.csv'.format(threshold))
            if not os.path.exists(os.path.dirname(pred_path)):
                os.makedirs(os.path.dirname(pred_path))
            print(os.path.dirname(pred_path))
            df_all_time.to_csv(pred_path, index=False)

            ref_files_path = test_loader.dataset.val_dir
            print(ref_files_path)
            report_dir = normalize_path(self.config.checkpoint.report_dir)
            # evaluate_bootstrapped
            # reports = evaluate_bootstrapped(df_all_time, ref_files_path, self.config.team_name, self.config.dataset, report_dir)
            # print(reports)
            # if reports['fmeasure']['mean'] > best_f1:
            #     best_f1 = reports['overall_scores']['fmeasure (percentage)']
            #     best_res = reports
            
            report = evaluate(df_all_time, ref_files_path, self.config.team_name, self.config.dataset, report_dir)
            if report['overall_scores']['fmeasure (percentage)'] > best_f1:
                best_f1 = report['overall_scores']['fmeasure (percentage)']
                best_res = report
                best_threshold = threshold
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(best_res)
        print('best_threshold:{}'.format(best_threshold))
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
    
    