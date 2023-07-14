#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Example implementation of MAML++ on miniImageNet.
"""
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
print(src_dir)
sys.path.append(src_dir)
import torch.nn.functional as F


import learn2learn as l2l
import numpy as np
import random
import torch
from src.utils.class_dataset import *
from src.utils.file_dataset import *
from src.utils.sampler import *
from collections import namedtuple
from typing import Tuple
from tqdm import tqdm

# from learn2earn_pp.mamlpp.cnn4_bnrs import CNN4_BNRS
# from learn2learn_pp.mamlpp.MAMLpp import MAMLpp
from cnn4_bnrs import CNN4_BNRS
from MAMLpp import MAMLpp


MetaBatch = namedtuple("MetaBatch", "support query")

train_samples, val_samples, test_samples = 38400, 9600, 12000  # Is that correct?
tasks = 600


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class MAMLppTrainer:
    def __init__(
        self,
        ways=5,
        k_shots=10,
        n_queries=30,
        steps=5,
        msl_epochs=25,
        DA_epochs=50,
        use_cuda=True,
        seed=42,
    ):
        self._use_cuda = use_cuda
        self._device = torch.device("cpu")
        if self._use_cuda and torch.cuda.device_count():
            torch.cuda.manual_seed(seed)
            self._device = torch.device("cuda")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Dataset
        # print("[*] Loading mini-ImageNet...")
        # (
        #     self._train_tasks,
        #     self._valid_tasks,
        #     self._test_tasks,
        # ) = l2l.vision.benchmarks.get_tasksets(
        #     "mini-imagenet",
        #     train_samples=k_shots,
        #     train_ways=ways,
        #     test_samples=n_queries,
        #     test_ways=ways,
        #     root="~/data",
        # )
        if not GlobalHydra().is_initialized():
            initialize(config_path="../")
        cfg = compose(config_name="config.yaml")
        self.config = cfg
        self.n_way = self.config.train.n_way
        self.n_support = self.config.train.n_support
        self.n_query = self.config.train.n_query
        train_dataset = ClassDataset(cfg, mode = 'train', same_class_in_different_file=True, debug= False)
        batch_sampler = TaskBatchSampler(cfg, train_dataset.classes, len(train_dataset))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = DataLoader(train_dataset, batch_sampler= batch_sampler,collate_fn=batch_sampler.get_collate_fn())

        # Model
        self._model = CNN4_BNRS(ways, adaptation_steps=steps)
        if self._use_cuda:
            self._model.cuda()

        # Meta-Learning related
        self._steps = steps
        self._k_shots = k_shots
        self._n_queries = n_queries
        self._inner_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        # Multi-Step Loss
        self._msl_epochs = msl_epochs
        self._step_weights = torch.ones(steps, device=self._device) * (1.0 / steps)
        self._msl_decay_rate = 1.0 / steps / msl_epochs
        self._msl_min_value_for_non_final_losses = torch.tensor(0.03 / steps)
        self._msl_max_value_for_final_loss = 1.0 - (
            (steps - 1) * self._msl_min_value_for_non_final_losses
        )

        # Derivative-Order Annealing (when to start using second-order opt)
        self._derivative_order_annealing_from_epoch = DA_epochs

    def _anneal_step_weights(self):
        self._step_weights[:-1] = torch.max(
            self._step_weights[:-1] - self._msl_decay_rate,
            self._msl_min_value_for_non_final_losses,
        )
        self._step_weights[-1] = torch.min(
            self._step_weights[-1] + ((self._steps - 1) * self._msl_decay_rate),
            self._msl_max_value_for_final_loss,
        )

    def _split_batch(self, batch: tuple) -> MetaBatch:
        """
        Separate data batch into adaptation/evalutation sets.
        """
        images, labels = batch
        batch_size = self._k_shots + self._n_queries
        assert batch_size <= images.shape[0], "K+N are greater than the batch size!"
        indices = torch.randperm(batch_size)
        support_indices = indices[: self._k_shots]
        query_indices = indices[self._k_shots :]
        return MetaBatch(
            (
                images[support_indices],
                labels[support_indices],
            ),
            (images[query_indices], labels[query_indices]),
        )
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
        
    def split_2d(self, data):
        if torch.is_tensor(data):
            data = data.view(-1, self.n_way, data.size(-2), data.size(-1))
            return data
        else:
            raise ValueError('Unsupported data type: {}'.format(type(data)))
    def _training_step(
        self,
        batch: MetaBatch,
        learner: MAMLpp,
        old_learner: MAMLpp,
        msl: bool = True,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, float]:
        
        
        pos_data = batch 
        classes, data_pos, _ =pos_data
        # support_feat, query_feat = self.split_support_query_feature(data_pos, None, is_data = True)
        s_inputs, q_inputs = self.split_support_query_data(data_pos, None)
        s_labels = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_support)).long().to(self.device)
        q_labels = torch.from_numpy(np.tile(np.arange(self.n_way),self.n_query)).long().to(self.device)
        
        support_feat = old_learner(s_inputs, step = 0)
        support_label_unique = torch.unique(s_labels)
        support_feat = [torch.mean(support_feat[(s_labels == label)], dim=0) for label in support_label_unique]
        support_feat = torch.stack(support_feat, dim=0).unsqueeze(0)
        prototypes     = support_feat.mean(0).squeeze()
        norms = torch.norm(prototypes, dim=1, keepdim=True)
        expanded_norms = norms.expand_as(prototypes)
        prototypes = prototypes / expanded_norms
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()
        classifier_head = [output_weight, output_bias]
        
        # s_inputs, s_labels = batch.support
        # q_inputs, q_labels = batch.query
        query_loss = torch.tensor(.0, device=self._device)

        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._device)
            s_labels = s_labels.cuda(device=self._device)
            q_inputs = q_inputs.float().cuda(device=self._device)
            q_labels = q_labels.cuda(device=self._device)
        
        
        # Derivative-Order Annealing
        second_order = True
        if epoch < self._derivative_order_annealing_from_epoch:
            second_order = False

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            pred = learner(s_inputs, step)
            pred = F.linear(pred, classifier_head[0], classifier_head[1])
            support_loss = self._inner_criterion(pred, s_labels)
            grad_head = torch.autograd.grad(support_loss, classifier_head, create_graph=True)
            learner.adapt(support_loss, first_order=not second_order, step=step)
            
            
            output_weight.data = output_weight.data - self.config.train.lr_inner * grad_head[0]
            output_bias.data = output_bias.data - self.config.train.lr_inner * grad_head[1]
            classifier_head = [output_weight, output_bias]
            # Multi-Step Loss
            if msl:
                q_pred = learner(q_inputs, step)
                query_loss += self._step_weights[step] * self._inner_criterion(
                    q_pred, q_labels
                )

        # Evaluate the adapted model on the query set
        if not msl:
            q_pred = learner(q_inputs, self._steps-1)
            query_loss = self._inner_criterion(q_pred, q_labels)
        acc = accuracy(q_pred, q_labels).detach()

        return query_loss, acc

    def _testing_step(
        self, batch: MetaBatch, learner: MAMLpp
    ) -> Tuple[torch.Tensor, float]:
        s_inputs, s_labels = batch.support
        q_inputs, q_labels = batch.query

        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._device)
            s_labels = s_labels.cuda(device=self._device)
            q_inputs = q_inputs.float().cuda(device=self._device)
            q_labels = q_labels.cuda(device=self._device)

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            pred = learner(s_inputs, step)
            support_loss = self._inner_criterion(pred, s_labels)
            learner.adapt(support_loss, step=step)

        # Evaluate the adapted model on the query set
        q_pred = learner(q_inputs, self._steps-1)
        query_loss = self._inner_criterion(q_pred, q_labels).detach()
        acc = accuracy(q_pred, q_labels)

        return query_loss, acc

    def train(
        self,
        meta_lr=0.001,
        fast_lr=0.01,
        meta_bsz=5,
        epochs=100,
        val_interval=1,
    ):
        print("[*] Training...")
        maml = MAMLpp(
            self._model,
            lr=fast_lr, # Initialisation LR for all layers and steps
            adaptation_steps=self._steps, # For LSLR
            first_order=False,
            allow_nograd=True, # For the parameters of the MetaBatchNorm layers
        )
        opt = torch.optim.AdamW(maml.parameters(), meta_lr, betas=(0, 0.999))

        iter_per_epoch = (
            train_samples // (meta_bsz * (self._k_shots + self._n_queries))
        ) + 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=epochs * iter_per_epoch,
            eta_min=0.00001,
        )

        for epoch in range(epochs):
            epoch_meta_train_loss, epoch_meta_train_acc = 0.0, 0.0
            for i, task_batch in tqdm(enumerate(self.train_loader)):
                opt.zero_grad()
                meta_train_losses, meta_train_accs = [], []

                for task in task_batch:
                    
                    
                    meta_loss, meta_acc = self._training_step(
                        task,
                        maml.clone(),
                        maml,
                        msl=(epoch < self._msl_epochs),
                        epoch=epoch,
                    )
                    meta_loss.backward()
                    meta_train_losses.append(meta_loss.detach())
                    meta_train_accs.append(meta_acc)

                epoch_meta_train_loss += torch.Tensor(meta_train_losses).mean().item()
                epoch_meta_train_acc += torch.Tensor(meta_train_accs).mean().item()

                # Average the accumulated gradients and optimize
                with torch.no_grad():
                    for p in maml.parameters():
                        # Remember the MetaBatchNorm layer has parameters that don't require grad!
                        if p.requires_grad:
                            p.grad.data.mul_(1.0 / meta_bsz)

                opt.step()
                scheduler.step()
                # Multi-Step Loss
                self._anneal_step_weights()

            epoch_meta_train_loss /= iter_per_epoch
            epoch_meta_train_acc /= iter_per_epoch
            print(f"==========[Epoch {epoch}]==========")
            print(f"Meta-training Loss: {epoch_meta_train_loss:.6f}")
            print(f"Meta-training Acc: {epoch_meta_train_acc:.6f}")

            # ======= Validation ========
            if (epoch + 1) % val_interval == 0:
                # Backup the BatchNorm layers' running statistics
                maml.backup_stats()

                # Compute the meta-validation loss
                # TODO: Go through the entire validation set, which shouldn't be shuffled, and
                # which tasks should not be continuously resampled from!
                meta_val_losses, meta_val_accs = [], []
                for _ in tqdm(range(val_samples // tasks)):
                    meta_batch = self._split_batch(self._valid_tasks.sample())
                    loss, acc = self._testing_step(meta_batch, maml.clone())
                    meta_val_losses.append(loss)
                    meta_val_accs.append(acc)
                meta_val_loss = float(torch.Tensor(meta_val_losses).mean().item())
                meta_val_acc = float(torch.Tensor(meta_val_accs).mean().item())
                print(f"Meta-validation Loss: {meta_val_loss:.6f}")
                print(f"Meta-validation Accuracy: {meta_val_acc:.6f}")
                # Restore the BatchNorm layers' running statistics
                maml.restore_backup_stats()
            print("============================================")

        return self._model.state_dict()

    def test(
        self,
        model_state_dict,
        meta_lr=0.001,
        fast_lr=0.01,
        meta_bsz=5,
    ):
        self._model.load_state_dict(model_state_dict)
        maml = MAMLpp(
            self._model,
            lr=fast_lr,
            adaptation_steps=self._steps,
            first_order=False,
            allow_nograd=True,
        )
        opt = torch.optim.AdamW(maml.parameters(), meta_lr, betas=(0, 0.999))

        meta_losses, meta_accs = [], []
        for _ in tqdm(range(test_samples // tasks)):
            meta_batch = self._split_batch(self._test_tasks.sample())
            loss, acc = self._testing_step(meta_batch, maml.clone())
            meta_losses.append(loss)
            meta_accs.append(acc)
        loss = float(torch.Tensor(meta_losses).mean().item())
        acc = float(torch.Tensor(meta_accs).mean().item())
        print(f"Meta-training Loss: {loss:.6f}")
        print(f"Meta-training Acc: {acc:.6f}")


if __name__ == "__main__":
    mamlPlusPlus = MAMLppTrainer()
    model = mamlPlusPlus.train()
    mamlPlusPlus.test(model)
