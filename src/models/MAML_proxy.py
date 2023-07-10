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
from pytorch_metric_learning import losses,reducers
from src.utils.distance import NormMinusLpDistance

class MAML_proxy(BaseModel):
    
    def __init__(self, config):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            num_inner_steps - Number of inner loop updates to perform
        """
        
        super(MAML_proxy, self).__init__(config)
        self.config = config
        self.approx = config.train.frist_order
        self.test_loop_batch_size = config.val.test_loop_batch_size
        # self.test_loop_batch_size = 128
        self.use_cosine_similarity = True
        
        
    def inner_loop(self, support_data, support_feature, support_label, mode = 'train'):
        dim = support_feature.shape[-1]
        
        if not self.use_cosine_similarity:
            if mode == 'train':
                loss_func = losses.ProxyAnchorLoss(num_classes=self.n_way, embedding_size=dim,   alpha = 4, margin =0.1, distance = NormMinusLpDistance(power = 2, p =2, normalize_embeddings = False)).to(torch.device('cuda'))
            else:
                loss_func = losses.ProxyAnchorLoss(num_classes=2, embedding_size=dim, alpha =4, margin = 0.1, distance = NormMinusLpDistance(power = 2, p =2, normalize_embeddings = False)).to(torch.device('cuda'))
                
        if self.use_cosine_similarity:
            if mode == 'train':
                loss_func = losses.ProxyAnchorLoss(num_classes=self.n_way, embedding_size=dim, alpha = self.config.train.alpha, margin = self.config.train.margin).to(torch.device('cuda'))
            else:
                loss_func = losses.ProxyAnchorLoss(num_classes=2, embedding_size=dim, alpha = self.config.train.alpha, margin = self.config.train.margin).to(torch.device('cuda'))
        
        
        prototypes  = support_feature.mean(0).squeeze()
        
        norms = torch.norm(prototypes, dim=1, keepdim=True)
        expanded_norms = norms.expand_as(prototypes)
        if self.use_cosine_similarity:   
            prototypes = prototypes / expanded_norms
        # prototypes = self.intializer(prototypes)
        init_weight =  prototypes
        output_weight = init_weight.detach().requires_grad_()
        loss_func.proxies = nn.parameter.Parameter(output_weight)
        
        # loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=self.config.train.lr_inner *self.config.train.scale)
        loss_func.train()
        
        # loss_optimizer.zero_grad()
        for weight in self.feature_extractor.parameters():
            weight.fast = None
        self.feature_extractor.zero_grad()

        
        # self.config.train.lr_inner = 0.01
        # Create output layer weights 
        # Optimize inner loop model on support set
        
        fast_parameters = list(self.feature_extractor.parameters())
        for weight in self.feature_extractor.parameters():
            weight.fast = None
        self.feature_extractor.zero_grad()
        
        for i in range(self.config.train.inner_step):
            # Determine loss on the support set
            proxies = loss_func.proxies
            
            loss, preds, acc = self.feed_forward(loss_func, support_data, support_label, mode = mode)
            

            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            grad_loss = torch.autograd.grad(loss, loss_func.parameters(), create_graph=True)
            if self.approx:
                grad = [ g.detach()  for g in grad ]
                grad_loss = [ g.detach()  for g in grad_loss ]
            
            fast_parameters = []
            
            for k, weight in enumerate(self.feature_extractor.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.config.train.lr_inner * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.config.train.lr_inner * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) # update the fast_parameters

            for k, weight in enumerate(self.loss_fn.parameters()):
                weight = weight - self.config.train.lr_inner * grad_loss[k]

        #     print('inner loop: loss:{:.3f} acc:{:.3f}'.format(loss.item(), torch.mean(acc).item()))
        # print('~~~~~~~~~~~')
        # print(preds)
        if mode != 'train':
            print('inner loop: loss:{:.3f} acc:{:.3f}'.format(loss.item(), torch.mean(acc).item())) 
            # print(preds)
        if mode != 'train':    
            print('!!!!!!!!!')
    
        return loss_func
   
    
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
    
    def feed_forward(self, loss_func, data, labels, mode):

        # Execute a model with given output layer weights and inputs
        
        feat_dataset = TensorDataset(data, torch.zeros(data.shape[0]))
        feat_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=False)
        feats = []
        for batch in feat_data_loader:
            data, _ = batch
            feats.append(self.feature_extractor(data))
        feats = torch.cat(feats, dim=0)
        
        proxies = loss_func.proxies
        
        if self.use_cosine_similarity:
            preds = self.cos_dist(feats, proxies)
            # print(preds)
            preds = preds *  self.config.train.temperature
        else:
            preds = - self.euclidean_dist(feats, proxies)
        
        # print(F.softmax(preds, dim = 1))
        
        # print(preds)
        # print(labels)
        # feats_pos = torch.mean(feats[(labels == 0)], dim=0)
        # for i in feats[(labels == 1)]:
        #     c_loss+=self.contrastive_loss(feats_pos, i, torch.tensor(0).to(self.device))
        # c_loss = c_loss/feats[(labels == 1)].shape[0]
        # half_size = 1
        # preds = preds / 2.0
        # if self.config.train.neg_prototype or mode == 'test':
        #     weights = torch.cat((torch.full((half_size,), 3, dtype=torch.float), torch.full((half_size,), 1, dtype=torch.float))).to(self.device)
        #     loss = F.cross_entropy(preds, labels, weight=weights)
        # else:
        # preds = preds * 2.0
        pos_num = preds[(labels == 0)].shape[0]
        neg_num = preds[(labels == 1)].shape[0]
        
        if mode == 'train':
            loss1 = F.cross_entropy(preds, labels)
        else:
            loss1 = F.cross_entropy(preds, labels, weight=torch.tensor([neg_num/pos_num, 1.0]).to(self.device))
        loss2= loss_func(feats, labels)
        loss = loss1 * 0.0 + loss2
        # print(F.softmax(preds, dim = 1))
        acc = (preds.argmax(dim=1) == labels).float()
        labels = labels.cpu().numpy()
        preds = preds.argmax(dim=1).detach().cpu().numpy()
        report = classification_report(labels, preds,zero_division=0, digits=3)
        # print(report)
        return loss, report, acc
    
    def euclidean_dist(self,query, support):
        n = query.size(0)
        m = support.size(0)
        query = query.unsqueeze(1).expand(n, m, -1)
        support = support.unsqueeze(0).expand(n, m, -1)
        return torch.pow(query - support, 2).sum(2)
    
    def feed_forward2(self, loss_func, data, labels, mode):
        # Execute a model with given output layer weights and inputs
        # print('feed_forward')
        # print(len(data))
        
        # print('data', data.shape)
        feat_dataset = TensorDataset(data, torch.zeros(data.shape[0]))
        feat_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=False)
        
        feats = []
        for batch in feat_data_loader:
            data, _ = batch
            feats.append(self.feature_extractor(data))
        
        feats = torch.cat(feats, dim=0)
        pos_num = feats[(labels == 0)].shape[0]
        neg_num = feats[(labels == 1)].shape[0]
        
        proxies = loss_func.proxies
        if self.use_cosine_similarity:
            preds = self.cos_dist(feats, proxies)
            preds = preds  *  self.config.train.temperature
        else:
            preds = - self.euclidean_dist(feats, proxies)
        preds = preds  *  self.config.train.temperature
        
        # preds = self.cos_dist(feats, proxies)
        #
        # print('preds', F.softmax(preds, dim = 1))
        if mode == 'train':
            loss1 = F.cross_entropy(preds, labels)
        else:
            loss1 = F.cross_entropy(preds, labels, weight=torch.tensor([neg_num/pos_num, 1.0]).to(self.device))
        
        # acc = (preds.argmax(dim=1) == labels).float().mean()
        # print('acc', acc)
        loss2= loss_func(feats, labels)
        # print('loss', loss2)
        loss = loss1   +  loss2 * 0.0
        # loss = loss1
        # print('loss2', loss2)
        # print(get_silhouette_score(feats, labels))
        # pos_num = feats[(labels == 0)].shape[0]
        # neg_num = feats[(labels == 1)].shape[0]
        
        # c_loss = torch.tensor(0.0).to(self.device)
        # feats_pos = torch.mean(feats[(labels == 0)], dim=0)
        # for i in feats[(labels == 1)]:
        #     c_loss+=self.contrastive_loss(feats_pos, i, torch.tensor(0).to(self.device))
        # c_loss = c_loss/feats[(labels == 1)].shape[0]

        # dataset = TensorDataset(feats, torch.zeros(feats.shape[0]))
        # data_loader = DataLoader(dataset, batch_size=32, shuffle=False)


        # print('loss', loss)
        # print(preds.argmax(dim=1))
        # print(labels)
        # print('!!!!!!')
        acc = (preds.argmax(dim=1) == labels).float()
        # aux_loss = (get_silhouette_score(feats, labels) + torch.tensor(1.).to(self.device)) / 2
        # print('aux_loss', aux_loss)
        # aux_loss = get_silhouette_score(feats, labels) + torch.tensor(1.).to(self.device)) / 2
        # loss -= torch.log(aux_loss)
        # loss += torch.tensor(1.).to(self.device)
        # loss -= get_silhouette_score(feats, labels)
        labels = labels.cpu().numpy()
        preds = preds.argmax(dim=1).detach().cpu().numpy()
        report = classification_report(labels, preds,zero_division=0, digits=3)
        # print(report)
        # report = None
        # acc = torch.tensor(0.0).to(self.device)
        return loss, report, acc

    
    def feed_forward_test(self, proxies, data):
        # Execute a model with given output layer weights and inputs
        
        feats = self.feature_extractor(data)
        if self.use_cosine_similarity:
            preds = self.cos_dist(feats, proxies)
            preds = preds *  self.config.train.temperature
        else:
            preds = - self.euclidean_dist(feats, proxies)
            
        if self.use_cosine_similarity:
            preds = self.cos_dist(feats, proxies)
            preds = preds *  self.config.train.temperature
        else:
            preds = - self.euclidean_dist(feats, proxies)
        # print('preds', -preds)
        # preds = preds/0.6
        preds = F.softmax(preds, dim = 1)
        # print('preds', preds)
        preds = preds.detach().cpu().numpy()
        return preds
    
    def outer_loop(self, data_loader, mode = 'train', opt = None):
        loss_epoch = []
        for i, task_batch in tqdm(enumerate(data_loader)):
            loss_all = []
            acc_all = []
            self.feature_extractor.zero_grad()
            for task in task_batch:
                for weight in self.feature_extractor.parameters():
                    weight.fast = None
                if self.config.train.neg_prototype:
                    pos_data, neg_data = task 
                    classes, data_pos, _ =pos_data
                    _, data_neg, _ =neg_data
                    # support_feat, query_feat = self.split_support_query_feature(data_pos, data_neg, is_data = True)
                    support_data, query_data = self.split_support_query_data(data_pos, data_neg)
                    support_label = torch.from_numpy(np.tile(np.arange(self.n_way*2),self.n_support)).long().to(self.device)
                else:
                    pos_data = task 
                    classes, data_pos, _ =pos_data
                    # support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
                    support_data, query_data = self.split_support_query_data(data_pos, None)
                    support_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_support)).long().to(self.device)

                # print(len(set(list(classes))))
                # Perform inner loop adaptation
                # print(data_pos.shape)
                # print(data_neg.shape)
                for weight in self.feature_extractor.parameters():
                    weight.fast = None
                
                support_feat = self.feature_extractor(support_data)
                support_label_unique = torch.unique(support_label)
                support_feat = [torch.mean(support_feat[(support_label == label)], dim=0) for label in support_label_unique]
                support_feat = torch.stack(support_feat, dim=0).unsqueeze(0)
                
                loss_func = self.inner_loop(support_data, support_feat, support_label, mode = 'train')
                
                query_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
                loss, report, acc = self.feed_forward2(loss_func, query_data , query_label, mode = 'train')
                # print(report)
                # print('~~~~')
                # print(loss.item())
                loss_all.append(loss)
                opt.zero_grad()
                # print(report)
                acc_all.append(acc)
            loss_q = torch.stack(loss_all).sum(0)
            loss_q.backward()
            # for i in self.parameters():
            #     print(i.grad)
            opt.step()
            opt.zero_grad()
            print('acc:{:.3f}'.format(torch.cat(acc_all).mean().item()))
            print('outer loop: loss:{:.3f}'.format(loss_q.item()/len(task_batch)))
        return np.mean(loss_epoch)

    def train_loop(self, data_loader, optimizer):
        
        return self.outer_loop(data_loader, mode = 'train', opt = optimizer)

    def test_loop(self, test_loader,fix_shreshold = None, mode = 'test'):
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
                # pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
                query_dataset = TensorDataset(query, torch.zeros(query.shape[0]))

                # pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                query_loader = DataLoader(query_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                # print(len(pos_dataset))
                # print(len(neg_dataset))
                # print(len(query_dataset))
                pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
                pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                for weight in self.feature_extractor.parameters():
                    weight.fast = None
                pos_feat = []
                for batch in pos_loader:
                    p_data, _ = batch
                    feat = self.forward(p_data)
                    # print(feat.shape)
                    pos_feat.append(feat.mean(0))
                pos_feat = torch.stack(pos_feat, dim=0).mean(0)

                prob_mean = []
                for i in range(5):
                    test_loop_neg_sample = self.config.val.test_loop_neg_sample
                    neg_sup[1] = neg_sup[1].squeeze() 
                    
                    if neg_sup[1].shape[0] > test_loop_neg_sample:
                        neg_indices = torch.randperm(neg_sup[1].shape[0])[:test_loop_neg_sample]
                        neg_seg_sample = neg_sup[1][neg_indices]
                    else:
                        neg_seg_sample = neg_sup[1]
                    for weight in self.feature_extractor.parameters():
                        weight.fast = None
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
                    proto = torch.stack([pos_feat,neg_feat], dim=0).unsqueeze(0)
                    
                    #########################################################################
                    # use less unsqueeze, used to match the dimension of inner loop
                    # proto = torch.stack([pos_feat,neg_feat], dim=0).unsqueeze(0)
                    # pos_data = pos_data[:5]
                    # neg_seg_sample = neg_seg_sample[:5]
                    support_data = torch.cat([pos_data, neg_seg_sample], dim=0)
                    # support_data = pos_data
                    m = pos_data.shape[0]
                    n = neg_seg_sample.shape[0]
                    
                    support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))

                    support_label = torch.from_numpy(support_label).long().to(self.device)
                    loss_func = self.inner_loop(support_data, proto, support_label, mode = 'test')
                    # support_label = np.zeros((m,))
                    # support_label = torch.from_numpy(support_label).long().to(self.device)
                    
                    
                    prob_all = []
                    for batch in tqdm(query_loader):
                        query_data, _ = batch
                        prob = self.feed_forward_test(loss_func.proxies, query_data)
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
                df_all_time = post_processing(df_all_time, self.config, mode = mode)
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
        # print(all_loss)
        return df_all_time, self.average_res(best_res_all), np.mean(best_threshold_all) , np.mean(all_loss)
    
    
    
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
    
    def test_loop_task(self, test_loader):
        self.outer_loop( test_loader, mode = 'test')
    