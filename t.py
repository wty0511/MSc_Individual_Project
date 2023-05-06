from src.models.ProtoMAML import *
import os
import sys

import numpy as np
import torch
import torch.optim
from src.models.ProtoNet import ProtoNet
import random
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from src.utils.dataset_train import *
from src.utils.dataset_val import *
from src.training_pipeline import train



if not GlobalHydra().is_initialized():
    initialize(config_path="./")
# Compose the configuration
cfg = compose(config_name="config.yaml")
print('preparing training dataset')
train_dataset = ClassDataset(cfg)
print('preparing val dataset')
val_dataset = FileDataset(cfg)
batch_sampler = TaskBatchSampler(cfg, train_dataset.classes, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_sampler= batch_sampler,collate_fn=batch_sampler.get_collate_fn())
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
model = ProtoMAML(cfg)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
for epoch in range(100):
    model.train_loop(train_loader, optimizer)
    best_model_dir = cfg.checkpoint.best_model_dir
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    save_file = os.path.join(best_model_dir, '{:d}.pth'.format(epoch))
    torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg}, save_file)
    val_dataset = FileDataset(cfg,val=True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    df_all_time, report = model.test_loop(val_loader)