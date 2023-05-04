from src.models.meta_learning import *
import torch
import numpy as np
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
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
    
    def inner_loop(self, support_data, support_feature):
        

        prototypes     = support_feature.mean(0).squeeze()
        
        if self.config.train.neg_prototype:
            support_label = torch.from_numpy(np.tile(np.arange(self.n_way*2),self.n_support)).long().to(self.device)
        else:
            support_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_support)).long().to(self.device)
        
        
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
        for i in range(100):
            # Determine loss on the support set

            loss, _, acc = self.feed_forward(local_model, output_weight, output_bias, support_data, support_label)
            
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            output_weight.data -= 0.01 * output_weight.grad
            output_bias.data -= 0.01 * output_bias.grad
            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)
        loss = loss.detach()
        acc = torch.mean(acc).detach()
        # print('inner loop: loss:{:.3f} acc:{:.3f}'.format(loss.item(), torch.mean(acc).item())) 
        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias
        
        return local_model, output_weight, output_bias
    
    
    def feed_forward(self, local_model, output_weight, output_bias, data, labels):
        # Execute a model with given output layer weights and inputs
        feats = local_model(data)
        preds = F.linear(feats, output_weight, output_bias)
        
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc
    
    
    def outer_loop(self, data_loader, mode = 'train', opt = None):
        accuracies = []
        losses = []
        self.feature_extractor.zero_grad()
        
        for i, data in tqdm(enumerate(data_loader)):
            if self.config.train.neg_prototype:
                pos_data, neg_data = data 
                _, data_pos, _ =pos_data
                _, data_neg, _ =neg_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, data_neg, is_data = True)
                support_data, query_data = self.split_support_query_data(data_pos, data_neg)
            else:
                pos_data = data 
                _, data_pos, _ =pos_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
                support_data, query_data = self.split_support_query_data(data_pos, None)
            # Perform inner loop adaptation
            local_model, output_weight, output_bias = self.inner_loop(support_data, support_feat)
            
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
            
        if mode == "train":
            opt.step()
            opt.zero_grad()

    def train_loop(self, data_loader, optimizer):
        
        self.outer_loop(data_loader, mode = 'train', opt = optimizer)
        
    def test_loop(self, test_loader):
        self.outer_loop(test_loader, mode = 'test')
