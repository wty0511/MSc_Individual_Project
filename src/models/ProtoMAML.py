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
        self.test_loop_batch_size = config.val.test_loop_batch_size
    
    def inner_loop(self, support_data, support_feature, support_label):
        

        prototypes     = support_feature.mean(0).squeeze()
        
        
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.feature_extractor)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), self.config.train.lr_inner, momentum = self.config.train.momentum, weight_decay=self.config.train.weight_decay) 
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()
        
        # Optimize inner loop model on support set
        for i in range(200):
            # Determine loss on the support set

            loss, _, acc = self.feed_forward(local_model, output_weight, output_bias, support_data, support_label)
            
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            output_weight.data -= 0.005 * output_weight.grad
            output_bias.data -= 0.005 * output_bias.grad
            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)
            # loss = loss.detach()
            # acc = torch.mean(acc).detach()
            # print('inner loop: loss:{:.3f} acc:{:.3f}'.format(loss.item(), torch.mean(acc).item())) 
        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias
        
        return local_model, output_weight, output_bias
    
    
    def feed_forward(self, local_model, output_weight, output_bias, data, labels):
        # Execute a model with given output layer weights and inputs
        feats = local_model(data)
        dataset = TensorDataset(feats, torch.zeros(feats.shape[0]))
        data_loader = DataLoader(dataset, batch_size=self.test_loop_batch_size, shuffle=False)
        preds_all = []
        for batch in data_loader:
            data, _ = batch
            preds = F.linear(data, output_weight, output_bias)
            preds_all.append(preds)
        preds = torch.cat(preds_all, dim=0)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc

    
    
    def feed_forward_test(self, local_model, output_weight, output_bias, data):
        # Execute a model with given output layer weights and inputs
        feats = local_model(data)
        preds = F.linear(feats, output_weight, output_bias)
        preds = F.softmax(preds, dim = 1)
        preds = preds.detach().cpu().numpy()
        return preds
    
    def outer_loop(self, data_loader, mode = 'train', opt = None):
        accuracies = []
        losses = []
        
        
        for i, data in tqdm(enumerate(data_loader)):
            if i %1 ==0:
                self.feature_extractor.zero_grad()
            if self.config.train.neg_prototype:
                pos_data, neg_data = data 
                classes, data_pos, _ =pos_data
                _, data_neg, _ =neg_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, data_neg, is_data = True)
                support_data, query_data = self.split_support_query_data(data_pos, data_neg)
                support_label = torch.from_numpy(np.tile(np.arange(self.n_way*2),self.n_support)).long().to(self.device)
            else:
                pos_data = data 
                classes, data_pos, _ =pos_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
                support_data, query_data = self.split_support_query_data(data_pos, None)
                support_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_support)).long().to(self.device)
            print(set(list(classes)))
            # Perform inner loop adaptation
            local_model, output_weight, output_bias = self.inner_loop(support_data, support_feat, support_label)
            
            query_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
            
            loss, _, acc = self.feed_forward(local_model, output_weight, output_bias, query_data , query_label)
            
            if mode == 'train':
                loss.backward()
                for p_global, p_local in zip(self.feature_extractor.parameters(), local_model.parameters()):
                    if p_global.grad is None or p_local.grad is None:
                        continue
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model
            
            print("loss: ", loss.detach().cpu().item(), "acc: ", acc.mean().detach().cpu().item())
            accuracies.append(acc.mean().detach())
            losses.append(loss.detach())
            
            if mode == "train" and i %1 ==0:
                opt.step()
                opt.zero_grad()

    def train_loop(self, data_loader, optimizer):
        
        self.outer_loop(data_loader, mode = 'train', opt = optimizer)

    def test_loop(self, test_loader):
        all_prob = {}
        all_meta = {}
        for i, (pos_sup, neg_sup, query, seg_len, seg_hop, query_start, query_end) in enumerate(test_loader):
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
            print(wav_file)
            print(query_start)
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
                #########################################################################
                # use less unsqueeze, used to match the dimension of inner loop
                proto = torch.stack([pos_feat,neg_feat], dim=0).unsqueeze(0)
                support_data = torch.cat([pos_data, neg_seg_sample], dim=0)
                m = pos_data.shape[0]
                n = neg_seg_sample.shape[0]
                support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))
                support_label = torch.from_numpy(support_label).long().to(self.device)
                local_model, output_weight, output_bias = self.inner_loop(support_data, proto, support_label)
                prob_all = []
                for batch in tqdm(query_loader):
                    query_data, _ = batch
                    prob = self.feed_forward_test(local_model, output_weight, output_bias, query_data)
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
        for threshold in np.arange(0.5, 1, 0.05):
            all_time = {'Audiofilename':[], 'Starttime':[], 'Endtime':[]}
            for wav_file in all_prob.keys():
                prob = np.where(all_prob[wav_file]>threshold, 1, 0)
                # print(len(prob))
                # print(np.sum(prob))
                # print(np.sum(prob)/len(prob))
                # print('~~~~~~~~')

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
            pred_path = normalize_path(self.config.val.pred_dir)
            pred_path = os.path.join(pred_path, 'pred_{}.csv'.format(threshold))

            if not os.path.dirname(pred_path):
                os.makedirs(os.path.dirname(pred_path))
            df_all_time.to_csv(pred_path, index=False)
            
            ref_files_path = normalize_path(test_loader.dataset.val_dir)
            report_dir = normalize_path(self.config.val.report_dir)
            report = evaluate(df_all_time, ref_files_path, self.config.team_name, self.config.dataset, report_dir)
            if report['overall_scores']['fmeasure (percentage)'] > best_f1:
                best_f1 = report['overall_scores']['fmeasure (percentage)']
                best_res = report
        print(best_res)
        return df_all_time, best_res
    
    
    
    
    def test_loop_task(self, test_loader):
        self.outer_loop( test_loader, mode = 'test')