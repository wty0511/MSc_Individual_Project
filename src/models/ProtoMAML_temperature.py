# This code is modified from https://github.com/KevinMusgrave/pytorch-adapt
# This code is modified from https://github.com/haoheliu/DCASE_2022_Task_5

# This code is modified from  DCASE 2022 challenge https://github.com/c4dm/dcase-few-shot-bioacoustic

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
import h5py
from src.utils.sampler import *
from src.utils.class_pair_dataset import *



class ProtoMAML_temp(BaseModel):
    
    def __init__(self, config):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            num_inner_steps - Number of inner loop updates to perform
        """
        
        super(ProtoMAML_temp, self).__init__(config)
        self.config = config
        self.test_loop_batch_size = config.val.test_loop_batch_size
        self.regularizer = nn.Sequential(
                nn.Linear(8, 16),
                nn.Sigmoid(),
                # nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus(),
                # nn.Hardtanh(0.5, 50)
            ).to(device=self.device)

    def feed_forward2(self, support_data, support_label):
        # # Execute a model with given output layer weights and inputs
        # support_feat = self.feature_extractor(support_data)
        # # nway 不是2nway要注意
        # support_feat = self.split_1d(support_feat)
        # prototype = support_feat.mean(0)
        # query_feat = self.feature_extractor(query_data)
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
        support_label = support_label.cpu().numpy()
        sampler = IntClassSampler(self.config, support_label, 100)
        dataset =  PairDataset(self.config, support_data, support_label, debug = False)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size = 10)
        loss = []
        for batch in dataloader:
            data, lable = batch
            anchor, pos, neg = data
            # print(anchor.shape)
            # print(pos.shape)
            # print(neg.shape)
            anchor_feat = self.feature_extractor(anchor)
            pos_feat = self.feature_extractor(pos)
            neg_feat = self.feature_extractor(neg)
            loss.append(self.tri_loss(anchor_feat, pos_feat, neg_feat))
            # print('inner loop: loss:{:.3f}'.format(loss.item()))
            # loss += self.loss_fn(anchor_feat, neg_feat, torch.ones(anchor_feat.shape[0]).long().to(self.device))
        
        loss = torch.sum(torch.stack(loss))
        # print(loss)
        report = None
        acc = torch.tensor(0.0).to(self.device)
        return loss, report, acc
    
    def inner_loop(self, support_data, support_feature, support_label, mode = 'train'):
        
        
        prototypes     = support_feature.mean(0).squeeze()
        norms = torch.norm(prototypes, dim=1, keepdim=True)
        expanded_norms = norms.expand_as(prototypes)
        # prototypes = prototypes / expanded_norms
        
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.feature_extractor)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), self.config.train.lr_inner, momentum = self.config.train.momentum, weight_decay=self.config.train.weight_decay)
        # local_optim = optim.SGD(local_model.parameters(), 0.0, momentum = self.config.train.momentum, weight_decay=self.config.train.weight_decay) 
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization
        # init_weight = 2 * prototypes
        # init_bias = -torch.norm(prototypes, dim=1)**2
        init_weight = 2 * prototypes
        # init_weight =  prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2

        

        
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # print('inner loop')
        # Optimize inner loop model on support set
        for i in range(15):
            # Determine loss on the support set

            loss, preds, acc = self.feed_forward(local_model, output_weight, output_bias, support_data, support_label, mode = mode)
            
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            
            output_weight.data -= self.config.train.lr_inner * output_weight.grad
            output_bias.data -= self.config.train.lr_inner * output_bias.grad
            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)
            loss = loss.detach()
            # print('loss', loss)
            acc = torch.mean(acc).detach()
        # print('~~~~~~~')
        if mode != 'train':
            print('inner loop: loss:{:.3f} acc:{:.3f}'.format(loss.item(), torch.mean(acc).item())) 
            print(preds)
        if mode != 'train':    
            print('!!!!!!!!!')
        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        # support_feat = local_model(support_data)
        support_dataset = TensorDataset(support_data, torch.zeros(support_data.shape[0]))
        support_data_loader = DataLoader(support_dataset, batch_size=1, shuffle=False)
        
        # support_feat = []
        # for batch in support_data_loader:
        #     data, _ = batch
        #     support_feat.append(local_model(data))
            
        # support_feat = torch.cat(support_feat, dim=0)
        support_feat = torch.empty(1, 512)
        # print('end inner loop')
        return local_model, output_weight, output_bias, support_feat
    
    
    def feed_forward(self, local_model, output_weight, output_bias, data, labels, mode):
        # Execute a model with given output layer weights and inputs
        # print('feed_forward')
        # print(len(data))
        
        # print('data', data.shape)
        feat_dataset = TensorDataset(data, torch.zeros(data.shape[0]))
        feat_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=False)
        
        feats = []
        for batch in feat_data_loader:
            data, _ = batch
            feats.append(local_model(data))
        feats = torch.cat(feats, dim=0)
        weight = []
        
        for name, param in local_model.named_parameters():
            if '0.weight' in name:
                weight.append(torch.mean(param))
            if '0.bias' in name:
                weight.append(torch.mean(param))
        weight = torch.stack(weight)
        
        
        pos_mean = torch.mean(feats[(labels == 0)], dim=0)
        neg_mean = torch.mean(feats[(labels == 1)], dim=0)
        euclidean_distance = torch.dist(pos_mean, neg_mean)
        cosine_similarity = F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0))

        # temperature_input = torch.cat((euclidean_distance), dim=0)
        # print(temperature_input)

        
        
        c_loss = torch.tensor(0.0).to(self.device)
        # feats_pos = torch.mean(feats[(labels == 0)], dim=0)
        # for i in feats[(labels == 1)]:
        #     c_loss+=self.contrastive_loss(feats_pos, i, torch.tensor(0).to(self.device))
        # c_loss = c_loss/feats[(labels == 1)].shape[0]
        pos_num = feats[(labels == 0)].shape[0]
        neg_num = feats[(labels == 1)].shape[0]
        

        dataset = TensorDataset(feats, torch.zeros(feats.shape[0]))
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        preds_all = []
        for batch in data_loader:
            data, _ = batch
            preds = F.linear(data, output_weight, output_bias)
            # preds = F.linear(data, output_weight, bias=None)
            preds_all.append(preds)
        preds = torch.cat(preds_all, dim=0)
        # print(preds)
        half_size = output_weight.shape[0] // 2
        temperature = 2.0
        # print('euclidean_distance', euclidean_distance)
        # temp = self.regularizer(euclidean_distance.unsqueeze(0))
        silhouette_score = get_silhouette_score(feats, labels).unsqueeze(0)
        # print('silhouette_score', silhouette_score)
        # print('silhouette_score', silhouette_score)
        preds_softmax = F.softmax(preds, dim=1)
        mean_probs = torch.mean(preds_softmax, dim=0) 
        condition_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9)) + torch.sum(preds_softmax * torch.log(preds_softmax + 1e-9))/preds_softmax.shape[0]
        print(torch.cat([silhouette_score,condition_entropy.unsqueeze(0), weight]))
        temp = self.regularizer(torch.cat([silhouette_score,condition_entropy.unsqueeze(0), weight]))
        
        # print('temp', temp)
        
        # print(preds)
        # print(temp)
        # print(temp.shape)
        preds = preds / temp
        # print(preds)
        if self.config.train.neg_prototype or mode == 'test':
            
            weights = torch.cat((torch.full((half_size,), max(neg_num/pos_num, 5), dtype=torch.float), torch.full((half_size,), 1, dtype=torch.float))).to(self.device)
            # weights = torch.cat((torch.full((half_size,), 1, dtype=torch.float), torch.full((half_size,), 1, dtype=torch.float))).to(self.device)
            loss = F.cross_entropy(preds, labels, weight=weights)
        else:
            loss = F.cross_entropy(preds, labels)
        # print(loss)
        # print(preds.argmax(dim=1))
        # print(labels)
        # print('!!!!!!')
        # print('pred', preds.shape)
        # print('loss', loss.unsqueeze(1).shape)
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
        loss+=c_loss
        return loss, report, acc

    
    
    def feed_forward_test(self, local_model, output_weight, output_bias, data):
        # Execute a model with given output layer weights and inputs
        local_model.eval()
        feats = local_model(data)
        preds = F.linear(feats, output_weight, output_bias)
        preds = F.softmax(preds, dim = 1)
        preds = preds.detach().cpu().numpy()
        return preds, feats
    
    def outer_loop(self, data_loader, mode = 'train', opt = None):

        
        for i, task_batch in tqdm(enumerate(data_loader)):
            accuracies = []
            losses = []
            self.feature_extractor.zero_grad()
            # for g in self.feature_extractor.parameters():
            #     print(g.grad)
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

                local_model, output_weight, output_bias, support_feats = self.inner_loop(support_data, support_feat, support_label)
                
                query_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
                # print('after inner loop')
                loss, _, acc = self.feed_forward(local_model, output_weight, output_bias, query_data , query_label, mode = 'train')
                
                
                
                if mode == 'train':

                    
                    # for g in self.feature_extractor.parameters():
                    #     print(g.grad)
                    # return
                    loss.backward()
                    # print('loss', loss)
                    # for i in self.regularizer.parameters():
                    #     print(i.grad)
                    #     print(i.data)
                    for p_global, p_local in zip(self.feature_extractor.parameters(), local_model.parameters()):
                        
                        if p_global.grad is None or p_local.grad is None:
                            print('None')
                            continue
                        # print(p_global.grad)
                        p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model
                loss = loss.detach().cpu().item()
                acc = acc.mean().detach().cpu().item()
                # print("loss: ", loss, "acc: ", acc)
                accuracies.append(acc)
                losses.append(loss)
            if i % 1 == 0:
                print("loss: ", np.mean(losses), "acc: ", np.mean(accuracies))
            if mode == "train":
                opt.step()
                opt.zero_grad()

    def train_loop(self, data_loader, optimizer):
        
        self.outer_loop(data_loader, mode = 'train', opt = optimizer)

    def test_loop(self, test_loader,fix_shreshold = None):
        best_res_all = []
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
                pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
                query_dataset = TensorDataset(query, torch.zeros(query.shape[0]))

                pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                query_loader = DataLoader(query_dataset, batch_size=16, shuffle=False)
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
                    feat_file = os.path.splitext(os.path.basename(wav_file))[0] + '.hdf5'
                    feat_file = os.path.join('/root/task5_2023/latent_feature/protoMAML', feat_file)
                    if os.path.isfile(feat_file):
                        os.remove(feat_file)
                    dims = (len(query_dataset), 512)
                    
                    directory = os.path.dirname(feat_file)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                        
                        
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

                    print("Current GPU Memory Usage By PyTorch: {} GB".format(torch.cuda.memory_allocated(self.device) / 1e9))
                    local_model, output_weight, output_bias, support_feats = self.inner_loop(support_data, proto, support_label, mode = 'test')
                    print("Current GPU Memory Usage By PyTorch: {} GB".format(torch.cuda.memory_allocated(self.device) / 1e9))
                    # with h5py.File(feat_file, 'w') as f:
                    #     f.create_dataset("features", (0, 512), maxshape=(None, 512))
                    #     f.create_dataset("labels", data=label.squeeze().cpu().numpy())
                    #     f.create_dataset("features_t", data = support_feats.detach().cpu().numpy())
                    #     f.create_dataset("labels_t", data=support_label.cpu().numpy())
                    prob_all = []
                    for batch in tqdm(query_loader):
                        query_data, _ = batch
                        prob, feats = self.feed_forward_test(local_model, output_weight, output_bias, query_data)
                        feats = feats.detach().cpu().numpy()
                        # with h5py.File(feat_file, 'a') as f:
                            
                        #     size = f['features'].shape[0]
                        #     nwe_size = f['features'].shape[0] + feats.shape[0]

                        #     f['features'].resize((nwe_size, 512))

                        #     f['features'][size:nwe_size] = feats
                        prob_all.append(prob)
                    prob_all = np.concatenate(prob_all, axis=0)
                    #########################################################################
                    
                    prob_all = prob_all[:,0]
                    print(np.sum(prob_all>0.9))
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
            # for i in best_report.keys():
            #     print(i)
            #     print(best_report[i])
            #     print('~~~~~~~~~~~~~~~')
            # print(best_res)
            best_threshold_all.append(best_threshold)
            best_res_all.append(best_res)
            # print('best_threshold', best_threshold)
            # print('~~~~~~~~~~~~~~~')
        print(self.average_res(best_res_all))
        print(np.mean(best_threshold_all))
        return df_all_time, self.average_res(best_res_all), np.mean(best_threshold_all)
    
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
    