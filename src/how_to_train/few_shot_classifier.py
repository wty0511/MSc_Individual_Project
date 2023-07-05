# code modified how to train your maml
# https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
import os
from src.models.meta_learning import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.feature_extractor import *
import torch.optim as optim
import itertools
from sklearn.metrics import classification_report, f1_score
from src.evaluation_metrics.evaluation import *
from src.utils.post_processing import *
from copy import deepcopy
from src.how_to_train.meta_optimizer import LSLRGradientDescentLearningRule
from src.how_to_train.meta_neural_net_work_architectures import Convnet

# def set_torch_seed(seed):
#     """
#     Sets the pytorch seeds for current experiment run
#     :param seed: The seed (int)
#     :return: A random number generator to use
#     """
#     rng = np.random.RandomState(seed=seed)
#     torch_seed = rng.randint(0, 999999)
#     torch.manual_seed(seed=torch_seed)

#     return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, cfg):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.config = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.batch_size = cfg.train.task_batch_size
        self.use_cuda = True
        self.current_epoch = 0
        # self.classifier_head = nn.Linear(512, cfg.train.n_way).to(device=self.device)
        self.classifier = Convnet(cfg = cfg, meta_classifier=True).to(device=self.device)
        self.task_learning_rate = cfg.train.lr_inner
        self.n_way = cfg.train.n_way
        self.n_support = cfg.train.n_support
        self.n_query = cfg.train.n_query
        # 可训练的模块类似于META SGD
        self.test_loop_batch_size = cfg.val.test_loop_batch_size
        # params_head = self.classifier_head.named_parameters()
        # params_classifier = self.classifier.named_parameters()
        output_weight = torch.zeros(self.n_way, 512).to(device=self.device).requires_grad_()
        output_bias = torch.zeros(self.n_way).to(device=self.device).requires_grad_()
        params_head = {'weight': output_weight, 'bias': output_bias}.items()
        all_params = itertools.chain(self.classifier.named_parameters(), params_head)
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=self.device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps= cfg.train.inner_step,
                                                                    use_learnable_learning_rates= True)
        self.sr = cfg.features.sr
        self.fps = self.sr / cfg.features.hop_length
        
        t = self.get_inner_loop_parameter_dict(params= all_params)
        # print(t.keys())
        self.inner_loop_optimizer.initialise(
            names_weights_dict=t)
        # self.inner_loop_optimizer.initialise(
            # names_weights_dict=self.get_inner_loop_parameter_dict(params=self.named_parameters()))
        # print("Inner Loop parameters")
        # for key, value in self.inner_loop_optimizer.named_parameters():
        #     print(key, value.shape)

      
        self.to(self.device)
        # print("Outer Loop parameters")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape, param.device, param.requires_grad)


        self.optimizer = optim.Adam(self.trainable_parameters(), lr= cfg.train.lr, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.config.train.epoches,
                                                              eta_min=self.config.train.min_lr)

        # self.device = torch.device('cpu')
        # if torch.cuda.is_available():
        #     if torch.cuda.device_count() > 1:
        #         self.to(torch.cuda.current_device())
        #         self.classifier = nn.DataParallel(module=self.classifier)
        #     else:
        #         self.to(torch.cuda.current_device())

        #     self.device = torch.cuda.current_device()

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        # 开始的时候，每个step的loss权重都是一样的
        loss_weights = np.ones(shape=(self.config.train.inner_step)) * (
                1.0 / self.config.train.inner_step)
        # 每个epoch decay一次，decay_rate是一个step的权重，multi_step_loss_num_epochs 是使用MSL的epoch数
        decay_rate = 1.0 / self.config.train.inner_step / self.config.train.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.config.train.inner_step
        for i in range(len(loss_weights) - 1):
            # 除了最后一个step，其他step的loss权重都会decay，但是不能小于min_value_for_non_final_losses
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        # 最后一个step的loss权重会增加，但是不能大于 所有之前setp的权重都是最小的情况。保证他们的和为1
        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.config.train.inner_step - 1) * decay_rate),
            1.0 - ((self.config.train.inner_step - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        # 必须是可以计算梯度的参数，如果bn层的参数不可训练，那么就不会被加入到inner loop的参数中
        params_dict = {}
         
        for name, param in params:
            if param.requires_grad and (
                not self.config.train.enable_inner_loop_optimizable_bn_params
                and "norm_layer" not in name
                or self.config.train.enable_inner_loop_optimizable_bn_params
            ):
                params_dict[name] = param.to(device=self.device)
                
            
        # params_dict = {
        #     name: param.to(device=self.device)
        #     for name, param in params
        #     if param.requires_grad
        #     and (
        #         not self.config.train.enable_inner_loop_optimizable_bn_params
        #         and "norm_layer" not in name
        #         or self.config.train.enable_inner_loop_optimizable_bn_params
        #     )
        # }
        # print('inner loop params', params.keys())
        # print('inner loop params', params_dict.keys())
        return params_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        # for name, param in names_weights_copy.items():
        #     print(name, param.grad_fn)
        self.classifier.zero_grad(params=names_weights_copy)
        # self.classifier_head.zero_grad()
        # print(names_weights_copy.keys())
        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=False)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        # print('grads',names_grads_copy)
        #不是多卡，所以不需要这个
        # names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        # for key, grad in names_grads_copy.items():
        #     print(key)
        #     print(grad.shape)
        
        # for key, grad in names_weights_copy.items():
        #     print(key)
        #     print(grad.shape)
        
        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)
        
        # num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # names_weights_copy = {
        #     name.replace('module.', ''): value.unsqueeze(0).repeat(
        #         [num_devices] + [1 for i in range(len(value.shape))]) for
        #     name, value in names_weights_copy.items()}


        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = {'loss': torch.mean(torch.stack(total_losses))}

        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        # x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        # [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = self.config.train.n_way

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(data_batch))]
        self.classifier.zero_grad()
        
        task_accuracies = []
        # for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_support_set,
        #                       y_support_set,
        #                       x_target_set,
        #                       y_target_set)):
        for task_id, task in enumerate(data_batch):
            if self.config.train.neg_prototype:
                pos_data, neg_data = task 
                classes, data_pos, _ =pos_data
                _, data_neg, _ =neg_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, data_neg, is_data = True)
                support_data, query_data = self.split_support_query_data(data_pos, data_neg)
                support_label = torch.from_numpy(np.tile(np.arange(self.n_way*2),self.n_support)).long().to(self.device)
                query_label = torch.from_numpy(np.tile(np.arange(self.n_way*2),self.n_query)).long().to(self.device)
            else:
                pos_data = task 
                classes, data_pos, _ =pos_data
                support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
                support_data, query_data = self.split_support_query_data(data_pos, None)
                support_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_support)).long().to(self.device)
                query_label = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
            
            prototypes     = support_feat.mean(0).squeeze()
            norms = torch.norm(prototypes, dim=1, keepdim=True)
            expanded_norms = norms.expand_as(prototypes)
            prototypes = prototypes / expanded_norms
            init_weight = 2 * prototypes
            init_bias = -torch.norm(prototypes, dim=1)**2
            output_weight = init_weight.detach().requires_grad_()
            output_bias = init_bias.detach().requires_grad_()
            # self.classifier_head = nn.Linear(512, self.n_way).to(device=self.device)
            # self.classifier_head.weight= nn.Parameter(output_weight, requires_grad=True)
            # self.classifier_head.bias = nn.Parameter(output_bias, requires_grad=True)
            # self.classifier_head.zero_grad()
            task_losses = []
            

            
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            # names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
            # params_head = self.classifier_head.named_parameters()
            params_head = {'weight': output_weight, 'bias': output_bias}.items()
            
            params_classifier = self.classifier.named_parameters()
            all_params = itertools.chain(params_head, params_classifier)
            names_weights_copy = self.get_inner_loop_parameter_dict(all_params)
            # for key, value in names_weights_copy.items():
            #     print(key, value.shape)
            # print('~~~~')
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            # names_weights_copy = {
            #     name.replace('module.', ''): value.unsqueeze(0).repeat(
            #         [num_devices] + [1 for i in range(len(value.shape))]) for
            #     name, value in names_weights_copy.items()}

            # n, s, c, h, w = x_target_set_task.shape

            # x_support_set_task = x_support_set_task.view(-1, c, h, w)
            # y_support_set_task = y_support_set_task.view(-1)
            # x_target_set_task = x_target_set_task.view(-1, c, h, w)
            # y_target_set_task = y_target_set_task.view(-1)
            # print('start inner loop')
            for num_step in range(num_steps):

                support_loss, support_preds = self.net_forward(
                    x=support_data,
                    y=support_label,
                    weights=names_weights_copy,
                    backup_running_statistics=num_step == 0,
                    training=True,
                    num_step=num_step,
                )

                # print('first layer before update',names_weights_copy['layer_dict.conv0.conv.weight'].shape)
                # print('step',num_step)
                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)
                # print('first layer after update',names_weights_copy['layer_dict.conv0.conv.weight'].shape)
                # print('~~~~')
                if use_multi_step_loss_optimization and training_phase and epoch < self.config.train.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=query_data,
                                                                 y=query_label, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                
                elif num_step == (self.config.train.inner_step - 1):
                    target_loss, target_preds = self.net_forward(x=query_data,
                                                                 y=query_label, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)
                    task_losses.append(target_loss)
                
            # print('end inner loop')
            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(query_label.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)
            if not training_phase:
                self.classifier.restore_backup_stats()
            output_weight = (output_weight - init_weight).detach() + init_weight
            output_bias = (output_bias - init_bias).detach() + init_bias

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds



    def forward_test(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase, fix_shreshold = None, mode = 'test'):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        # x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        # [b, ncs, spc] = y_support_set.shape
        fix_shreshold = 0.5
        best_res_all = []
        best_threshold_all = []
        for i in range(1):
            all_prob = {}
            all_meta = {}
            for i, (pos_sup, neg_sup, query, seg_len, seg_hop, query_start, query_end, label) in enumerate(data_batch):
                seg_hop = seg_hop.item()
                query_start = query_start.item()
                wav_file= pos_sup[0][0].split('&')[1]
                
                all_meta[wav_file]={}
                all_meta[wav_file]['start'] = query_start

                all_meta[wav_file]['end'] = query_end
                all_meta[wav_file]['seg_hop'] = seg_hop
                all_meta[wav_file]['seg_len'] = seg_len
                
                all_meta[wav_file]['label'] = label[0]
                pos_data = pos_sup[1].squeeze()
                query = query.squeeze()
                pos_dataset = TensorDataset(pos_data,  torch.zeros(pos_data.shape[0]))
                query_dataset = TensorDataset(query, torch.zeros(query.shape[0]))
                pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                query_loader = DataLoader(query_dataset, batch_size=16, shuffle=False)
                pos_feat = []
                for batch in pos_loader:
                    p_data, _ = batch
                    feat = self.classifier(p_data, num_step=0, training=False, backup_running_statistics=False, output_features=False)
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
                        feat = self.classifier(n_data, num_step=0, training=False, backup_running_statistics=False, output_features=False)
                        # print(feat.shape)
                        neg_feat.append(feat.mean(0))
                    neg_feat = torch.stack(neg_feat, dim=0).mean(0)


                    self.num_classes_per_set = self.config.train.n_way

                    total_losses = []
                    total_accuracies = []
                    per_task_target_preds = [[] for i in range(len(data_batch))]
                    self.classifier.zero_grad()
                    print('~~~~')
                    task_accuracies = []
                    # for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_support_set,
                    #                       y_support_set,
                    #                       x_target_set,
                    #                       y_target_set)):

                    prototypes = torch.stack([pos_feat,neg_feat], dim=0)

                    norms = torch.norm(prototypes, dim=1, keepdim=True)
                    expanded_norms = norms.expand_as(prototypes)
                    prototypes = prototypes / expanded_norms
                    init_weight = 2 * prototypes
                    init_bias = -torch.norm(prototypes, dim=1)**2
                    output_weight = init_weight.detach().requires_grad_()
                    output_bias = init_bias.detach().requires_grad_()
                    # self.classifier_head = nn.Linear(512, 2).to(device=self.device)
                    # self.classifier_head.weight = nn.Parameter(output_weight)
                    # self.classifier_head.bias = nn.Parameter(output_bias)
                    task_losses = []
                    support_data = torch.cat([pos_data, neg_seg_sample], dim=0)
                    # support_data = pos_data
                    m = pos_data.shape[0]
                    n = neg_seg_sample.shape[0]
                    
                    support_label = np.concatenate((np.zeros((m,)), np.ones((n,))))
                    support_label = torch.from_numpy(support_label).long().to(self.device)
                    per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
                    
                    params_head = {'weight': output_weight, 'bias': output_bias}.items()
                    params_classifier = self.classifier.named_parameters()
                    all_params = itertools.chain(params_head, params_classifier)
                    names_weights_copy = self.get_inner_loop_parameter_dict(all_params)
                    self.classifier.zero_grad()
                    # self.classifier_head.zero_grad()
                    for num_step in range(num_steps):

                        support_loss, support_preds = self.net_forward(
                            x=support_data,
                            y=support_label,
                            weights=names_weights_copy,
                            backup_running_statistics=num_step == 0,
                            training=True,
                            num_step=num_step,
                            test = True
                        )

                        # print('first layer before update',names_weights_copy['layer_dict.conv0.conv.weight'].shape)
                        # print(names_weights_copy['weight'])
                        names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                        names_weights_copy=names_weights_copy,
                                                                        use_second_order=use_second_order,
                                                                        current_step_idx=num_step)
                        # print(names_weights_copy['weight'])
                        # print('~~~~')
                        preds_feat = self.classifier(support_data, num_step=self.config.train.inner_step - 1, training=True, backup_running_statistics=True, output_features=False)
                        head_weight = names_weights_copy['weight']
                        head_bias = names_weights_copy['bias']
                        preds = F.linear(preds_feat, head_weight, head_bias)
                        preds = F.softmax(preds, dim = 1)
                        preds = preds.argmax(dim=1).detach().cpu().numpy()
    
                        print(classification_report(support_label.detach().cpu().numpy(), preds,zero_division=0, digits=3))
                        
                        
                    with torch.no_grad():
                        prob_all = []
                        for batch in query_loader:
                            query_data, _ = batch
                            preds_feat = self.classifier.forward(x=query_data, params=names_weights_copy,
                                            training=True,
                                            backup_running_statistics=True, num_step=self.config.train.inner_step - 1)
                            head_weight = names_weights_copy['weight']
                            head_bias = names_weights_copy['bias']
                            preds = F.linear(preds_feat, head_weight, head_bias)
                            preds = F.softmax(preds, dim = 1)
                            preds = preds.detach().cpu().numpy()
                            # with h5py.File(feat_file, 'a') as f:
                                
                            #     size = f['features'].shape[0]
                            #     nwe_size = f['features'].shape[0] + feats.shape[0]

                            #     f['features'].resize((nwe_size, 512))

                            #     f['features'][size:nwe_size] = feats
                            prob_all.append(preds)
                        prob_all = np.concatenate(prob_all, axis=0)
                        prob_all = prob_all[:,0]
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
                
                ref_files_path = data_batch.dataset.val_dir
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
            
    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, test = False):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds_feat = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step)
        head_weight = weights['weight']
        head_bias = weights['bias']
        preds = F.linear(preds_feat, head_weight, head_bias)
        # preds = self.classifier_head.forward(preds_feat)
        # print(preds)
        pos_num = x[(y == 0)].shape[0]
        neg_num = x[(y == 1)].shape[0]
        
        if test:
            weights = torch.cat((torch.full((1,), max(neg_num/pos_num, 1), dtype=torch.float), torch.full((1,), 1, dtype=torch.float))).to(self.device)
            loss = F.cross_entropy(input=preds, target=y,weight=weights)
        else:
            preds = preds
            loss = F.cross_entropy(input=preds, target=y)

        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        # head_name = [name for name in self.classifier_head.named_parameters()]
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.config.train.second_order and
                                                                      epoch > self.config.train.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.config.train.use_multi_step_loss_optimization,
                                                     num_steps=self.config.train.inner_step,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        pred_df, best_res, threshold = self.forward_test(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.config.train.inner_step,
                                                     training_phase=False)
        
        return pred_df, best_res, threshold

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is None:
        #             print('None gradient for outer loop parameter', name)
        #         else:
        #             print('Outer Loop Parameter', name, param.shape)
        # if 'imagenet' in self.args.dataset_name:
        #     for name, param in self.classifier.named_parameters():
        #         if param.requires_grad:
        #             param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()
        
        # x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        # x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        # x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        # y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        # y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        # data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.scheduler.step(epoch=epoch)
        self.optimizer.zero_grad()
        # self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        # x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        # x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        # x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        # y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        # y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        # data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        pred_df, best_res, threshold = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return pred_df, best_res, threshold

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.optimizer.load_state_dict(state['optimizer'])
        self.load_state_dict(state_dict=state_dict_loaded)
        return state


    def split2nway_1d(self, data):
        
        class_data, data, label = data
        class_data = np.array(class_data)
        class_data = class_data.reshape(-1, self.n_way)
        label = label.view(-1, self.n_way)
        return class_data, data, label
    
    def split_2d(self, data):
        if torch.is_tensor(data):
            data = data.view(-1, self.n_way, data.size(-2), data.size(-1))
            return data
        else:
            raise ValueError('Unsupported data type: {}'.format(type(data)))
    
    def split_1d(self, data):
        if torch.is_tensor(data):
            return data.view(-1, self.n_way, data.size(-1))
        elif isinstance(data, np.ndarray):
            return data.reshape(-1, self.n_way, data.shape[-1])
        else:
            raise ValueError('Unsupported data type: {}'.format(type(data)))



    def split_support_query_data(self, data_pos, data_neg):
        if self.config.train.neg_prototype:
            data_pos = self.split_2d(data_pos)
            data_neg = self.split_2d(data_neg)
            data_all = torch.cat([data_pos, data_neg], dim=1)
            data_support = data_all[:self.n_support, :, :, :]
            data_query = data_pos[self.n_support:, :, :, :]
        else:
            data_pos = self.split_2d(data_pos)
            data_support = data_pos[:self.n_support, :, :, :]
            data_query = data_pos[self.n_support:, :, :, :]
        data_support = data_support.view(-1, data_support.size(-2), data_support.size(-1))
        data_query = data_query.view(-1, data_query.size(-2), data_query.size(-1))
        return data_support, data_query
    
    def split_support_query_feature(self, pos_input, neg_input, is_data, model = None):
        
        if self.config.train.neg_prototype:
            if is_data:
                pos_dataset = TensorDataset(pos_input, torch.zeros(pos_input.shape[0]))
                neg_dataset = TensorDataset(neg_input, torch.zeros(neg_input.shape[0]))
                pos_loader = DataLoader(pos_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                neg_loader = DataLoader(neg_dataset, batch_size=self.test_loop_batch_size, shuffle=False)
                pos_all = []
                for batch in pos_loader:
                    data, _ = batch
                    if model is not None:
                        feature_pos = model.forward(data)
                    else:
                        feature_pos = self.classifier(data, num_step=0, training=True, backup_running_statistics=False, output_features=False)
                    pos_all.append(feature_pos)
                feature_pos = torch.cat(pos_all, dim=0)
                
                neg_all = []
                for batch in neg_loader:
                    data, _ = batch
                    if model is not None:
                        feature_neg = model.forward(data)
                    else:
                        feature_neg = self.classifier(data, num_step=0, training=True, backup_running_statistics=False, output_features=False)
                    neg_all.append(feature_neg)
                feature_neg = torch.cat(neg_all, dim=0)
            else:
                feature_pos = pos_input
                feature_neg = neg_input
            
            feature_pos = feature_pos.view(-1, self.n_way, feature_pos.size(-1))
            feature_neg = feature_neg.view(-1, self.n_way, feature_neg.size(-1))
            feature_all = torch.cat([feature_pos, feature_neg], dim=1)
            
            feature_support = feature_all[:self.n_support, :, :]
            feature_query = feature_pos[self.n_support:, :, :]
            
        else:
            if is_data:
                if model is not None:
                    feature_pos = model.forward(pos_input)
                else:
                    feature_pos = self.classifier(pos_input, num_step=0, training=True, backup_running_statistics=False, output_features=False)
                
            else:
                feature_pos = pos_input
            
            feature_pos = feature_pos.view(-1, self.n_way, feature_pos.size(-1))
            feature_support = feature_pos[:self.n_support, :, :]
            feature_query = feature_pos[self.n_support:, :, :]
        
        return feature_support, feature_query
        
        