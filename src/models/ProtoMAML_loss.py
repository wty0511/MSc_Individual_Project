# This code is modified from https://github.com/KevinMusgrave/pytorch-adapt
# This code is modified from https://github.com/haoheliu/DCASE_2022_Task_5
# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
# This code is modified from https://github.com/phlippe/uvadlc_notebooks
from src.models.meta_learning import *
import torch
import numpy as np
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from src.utils.feature_extractor import *
from src.evaluation_metrics.evaluation import *
from src.utils.post_processing import *
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import silhouette_score
# from pytorch_adapt.layers.silhouette_score import get_silhouette_score


def get_silhouette_score(feats, labels):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")
    scores = []
    for L in unique_labels:
        curr_cluster = feats[labels == L]
        num_elements = len(curr_cluster)
        if num_elements > 1:
            intra_cluster_dists = torch.cdist(curr_cluster, curr_cluster)
            mean_intra_dists = torch.sum(intra_cluster_dists, dim=1) / (
                num_elements - 1
            )  # minus 1 to exclude self distance
            dists_to_other_clusters = []
            for otherL in unique_labels:
                if otherL != L:
                    other_cluster = feats[labels == otherL]
                    inter_cluster_dists = torch.cdist(curr_cluster, other_cluster)
                    mean_inter_dists = torch.sum(inter_cluster_dists, dim=1) / (
                        len(other_cluster)
                    )
                    dists_to_other_clusters.append(mean_inter_dists)
            dists_to_other_clusters = torch.stack(dists_to_other_clusters, dim=1)
            min_dists, _ = torch.min(dists_to_other_clusters, dim=1)
            curr_scores = (min_dists - mean_intra_dists) / (
                torch.maximum(min_dists, mean_intra_dists)
            )
        else:
            curr_scores = torch.tensor([0], device=device, dtype=dtype)

        scores.append(curr_scores)

    scores = torch.cat(scores, dim=0)
    if len(scores) != num_samples:
        raise ValueError(
            f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})"
        )
    return torch.mean(scores)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, labels):
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(embeddings1 - embeddings2, 2), dim=0))
        loss = (labels * torch.pow(euclidean_distance, 2) +
                (1 - labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return torch.mean(loss) * 0.5




class ProtoMAML(BaseModel):
    
    def __init__(self, config):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            num_inner_steps - Number of inner loop updates to perform
        """
        
        super(ProtoMAML, self).__init__(config)
        self.config = config
        self.approx = True
        self.test_loop_batch_size = config.val.test_loop_batch_size
        self.contrastive_loss = ContrastiveLoss(20)

    
    
    def inner_loop(self, support_data, support_feature, support_label, mode = 'train', query_data = None):
        
        fast_parameters = list(self.feature_extractor.parameters())
        for weight in self.feature_extractor.parameters():
            weight.fast = None

        # Optimize inner loop model on support set
        for i in range(100):
            # Determine loss on the support set
            loss, preds, acc = self.feed_forward(support_data, support_label, mode = mode, query_data = query_data)
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

        
        # if query_data is not None:

        #     dataset_q = TensorDataset(query_data, torch.zeros(query_data.shape[0]))
        #     data_loader_q = DataLoader(dataset_q, batch_size=self.test_loop_batch_size, shuffle=False)
        #     for i in range(5):
        #         for batch in data_loader_q:
        #             data, _ = batch
        #             feat = local_model(data)
        #             preds = F.linear(feat, output_weight, output_bias)
        #             preds = F.softmax(preds, dim = 1)
        #             print(preds)
        #             margin_prob = torch.sum(preds, dim=0) / preds.shape[0]
        #             term1 = torch.sum(-margin_prob * torch.log(margin_prob))
        #             term2 = torch.sum(preds) / preds.shape[0]
        #             # element_size = preds.element_size()  # 每个元素的字节数
        #             # total_size = preds.numel() * element_size  # 总占用空间大小
        #             # # 打印结果
        #             # print(f"Tensor Size: {total_size / 1024**2:.2f} MB")
        #             # allocated_memory = torch.cuda.memory_allocated(device=self.device)
        #             # print(f"Allocated Memory: {allocated_memory / 1024**3:.2f} GB")
        #             loss =  term1 + term2
        #             if self.approx:
        #                 grad = torch.autograd.grad(loss, fast_parameters, create_graph=False)
        #             else:
        #                 grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)


        #             output_weight.data = output_weight.data - self.config.train.lr_inner * grad[-2]
        #             output_bias.data = output_bias.data - self.config.train.lr_inner * grad[-1]
        #             for k, weight in enumerate(local_model.parameters()):
        #                 # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
        #                 weight.grad = grad[k]
                    
        #             local_optim.step()
        #             local_optim.zero_grad()
            
        if mode != 'train':
            print('inner loop: loss:{:.3f}'.format(loss.item()))
        if mode != 'train':    
            print('!!!!!!!!!')

        return 

    # def silhouette(self, inputs, targets):
    #     # Convert tensors to numpy arrays
    #     inputs = inputs.detach().cpu().numpy()
    #     # targets = targets.argmax(dim=1).detach().cpu().numpy()
    #     targets = targets.detach().cpu().numpy()
    #     # Compute silhouette score
    #     try:
    #         score = silhouette_score(inputs, targets)
            
    #     except:
    #         print('silhouette error')
    #         score = -1.0
    #     # score = silhouette_score(inputs, targets)

    #     # Convert score to tensor
        
    #     score = torch.tensor(score, requires_grad=True).to(self.device)
    #     score -= 1
    #     # print(-score)
    #     # Since we want to maximize silhouette score, we minimize its negative
    #     return - score
    
    # def silhouette_score_torch(X, labels):
    #     def pairwise_distance(X):
    #         n = X.size(0)
    #         dist = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(n, n)
    #         dist = dist + dist.t()
    #         dist.addmm_(1, -2, X, X.t())
    #         dist = dist.clamp(min=0)
    #         return torch.sqrt(dist)

    #     def compute_a(X, labels, dists):
    #         mask = labels.expand(len(labels), len(labels)).eq(labels.expand(len(labels), len(labels)).t())
    #         mask.fill_diagonal_(0)
    #         a = torch.where(mask, dists, torch.zeros_like(dists)).sum(1) / mask.sum(1).float()
    #         return a

    #     def compute_b(X, labels, dists):
    #         n = len(labels)
    #         min_dists = torch.ones(n).to(X.device) * float('inf')
    #         for label in labels.unique():
    #             mask = labels.eq(label)
    #             _min_dists = dists.masked_fill(mask.unsqueeze(-1), float('inf')).min(1)[0]
    #             min_dists = torch.where(_min_dists < min_dists, _min_dists, min_dists)
    #         return min_dists

    #     dists = pairwise_distance(X)
    #     a = compute_a(X, labels, dists)
    #     b = compute_b(X, labels, dists)
    #     return (b - a).clamp(min=0).mean()


    def feed_forward(self, data, labels, mode, query_data = None):
        feats = self.feature_extractor(data)
        
        loss = - get_silhouette_score(feats, labels) +1.0
        report = None
        acc = 0
        return loss, report, acc
    
    def feed_forward_test(self,  prototype, query_data):
        # Execute a model with given output layer weights and inputs
        prototype = prototype.squeeze()
        query_feat = self.feature_extractor(query_data)
        dists = self.euclidean_dist(query_feat, prototype)

        pred = dists.argmin(-1)
        
        scores = -dists
        preds = F.softmax(scores, dim = 1)
        preds = preds.detach().cpu().numpy()
        return preds
    
    def outer_loop(self, data_loader, mode = 'train', opt = None):


        for i, task_batch in tqdm(enumerate(data_loader)):
            accuracies = []
            losses = []
            self.feature_extractor.zero_grad()
            loss_all = []
            for task in task_batch:
                if self.config.train.neg_prototype:
                    pos_data, neg_data = task 
                    classes, data_pos, _ =pos_data
                    _, data_neg, _ =neg_data
                    support_feat, query_feat = self.split_support_query_feature(data_pos, data_neg, is_data = True)
                    support_data, query_data = self.split_support_query_data(data_pos, data_neg)
                    support_label = torch.from_numpy(np.tile(np.arange(self.n_way*2),self.n_support)).long().to(self.device)
                else:
                    pos_data = task 
                    classes, data_pos, _ =pos_data
                    support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
                    support_data, query_data = self.split_support_query_data(data_pos, None)
                    support_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_support)).long().to(self.device)

                # print(len(set(list(classes))))
                # Perform inner loop adaptation
                # print(data_pos.shape)
                # print(data_neg.shape)

                self.inner_loop(support_data, support_feat, support_label, mode = 'train', query_data = None)
                
                query_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
                loss, _, acc = self.feed_forward(query_data , query_label, mode = 'train')
                loss_all.append(loss)
                opt.zero_grad()
                

            loss_q = torch.stack(loss_all).mean(0)
            loss_q.backward()
            opt.step()
            opt.zero_grad()
            print('outer loop: loss:{:.3f}'.format(loss_q.item()))


    def train_loop(self, data_loader, optimizer):
        
        self.outer_loop(data_loader, mode = 'train', opt = optimizer)

    def test_loop(self, test_loader,fix_shreshold = None):
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
            pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
            query_dataset = TensorDataset(query, torch.zeros(query.shape[0]))

            pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
            # print(len(pos_dataset))
            # print(len(neg_dataset))
            # print(len(query_dataset))
            pos_feat = []
            for batch in pos_loader:
                p_data, _ = batch
                feat = self.forward(p_data)
                # print(feat.shape)
                pos_feat.append(feat.mean(0))
            pos_feat = torch.stack(pos_feat, dim=0).mean(0)

            prob_mean = []
            for i in range(1):
                test_loop_neg_sample = self.config.val.test_loop_neg_sample
                neg_sup[1] = neg_sup[1].squeeze() 
                
                if neg_sup[1].shape[0] > test_loop_neg_sample:
                    neg_indices = torch.randperm(neg_sup[1].shape[0])[:test_loop_neg_sample]
                    neg_seg_sample = neg_sup[1][neg_indices]
                else:
                    neg_seg_sample = neg_sup[1]

                neg_dataset = TensorDataset(neg_seg_sample, torch.zeros(neg_seg_sample.shape[0]))
                neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                neg_feat = []
                for batch in neg_loader:
                    n_data, _ = batch
                    # print(neg_data.shape)
                    feat = self.forward(n_data)
                    # print(feat.shape)
                    neg_feat.append(feat.mean(0))
                neg_feat = torch.stack(neg_feat, dim=0).mean(0)
                #########################################################################
                # use less unsqueeze, used to match the dimension of inner loop
                proto = torch.stack([pos_feat,neg_feat], dim=0).unsqueeze(0)
                
                support_data = torch.cat([pos_data, neg_seg_sample], dim=0)
                # support_data = pos_data
                m = pos_data.shape[0]
                n = neg_seg_sample.shape[0]
                
                support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))
                support_label = torch.from_numpy(support_label).long().to(self.device)
                # support_label = np.zeros((m,))
                # support_label = torch.from_numpy(support_label).long().to(self.device)

                self.inner_loop(support_data, proto, support_label, mode = 'test', query_data = None)
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
        
        # print(query.shape)
        # print(support.shape)
        n = query.size(0)
        m = support.size(0)
        
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)

        return torch.pow(query - support, 2).sum(2)
    
    def test_loop_task(self, test_loader):
        self.outer_loop( test_loader, mode = 'test')
    