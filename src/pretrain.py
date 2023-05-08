import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import yaml
import argparse
import pandas as pd
import csv
import os
import pandas as pd
from src.utils.Feature_extract import feature_transform
from src.utils.Datagenerator import Datagen
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models.ProtoPretrain import Protonet
import torch
from tqdm import tqdm
from collections import Counter
from src.utils.batch_sampler import EpisodicBatchSampler
from torch.nn import functional as F
from glob import glob
import hydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf
import h5py
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from src.utils.trainer import Trainer




def train_protonet(model,train_loader,valid_loader,conf):
    arch = 'Protonet'
    alpha = 0.0  
    disable_tqdm = False 
    ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
    pretrain = False
    resume = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=0.5,
                                                   step_size=10)
    num_epochs = 14



    #cudnn.benchmark = True
    model.to(device) # cuda
    trainer = Trainer(device=device,num_class=19, train_loader=train_loader,val_loader=valid_loader,conf=conf)

    for epoch in range(num_epochs):
        trainer.do_epoch(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim)
        # Evaluation on validation set
        prec1 = trainer.meta_val(epoch=epoch,model=model, disable_tqdm=disable_tqdm)
        print('Meta Val {}: {}'.format(epoch, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))
        if is_best:
        # Save checkpoint
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':conf}, './Checkpoints/best.pth.tar')
        if lr_scheduler is not None:
            lr_scheduler.step()


if not GlobalHydra().is_initialized():
    initialize(config_path="../")
# Compose the configuration
cfg = compose(config_name="config.yaml")


gen_train = Datagen(cfg) 
X_train,Y_train,X_val,Y_val = gen_train.generate_train() 
X_tr = torch.tensor(X_train) 
Y_tr = torch.LongTensor(Y_train)
X_val = torch.tensor(X_val)
Y_val = torch.LongTensor(Y_val)


samples_per_cls =  cfg.train.n_support + cfg.train.n_query

batch_size_tr = samples_per_cls *  cfg.train.n_way
batch_size_vd = batch_size_tr

batch_size_tr = 64 # the batch size of training 
batch_size_vd = (cfg.train.n_support+cfg.train.n_query) * cfg.train.n_way

num_batches_vd = len(Y_val)//batch_size_vd
samplr_valid = EpisodicBatchSampler(Y_val,num_batches_vd,cfg.train.n_way,samples_per_cls)

train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr) # 利用torch 的 dataset,整合X,Y
valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size_tr,num_workers=0,pin_memory=True,shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,num_workers=0,pin_memory=True,shuffle=False)

model = Protonet()
train_protonet(model,train_loader,valid_loader,cfg)