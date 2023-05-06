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
from src.utils.class_dataset import *
from src.utils.file_dataset import *
from src.training_pipeline import train



if not GlobalHydra().is_initialized():
    initialize(config_path="./")
# Compose the configuration
cfg = compose(config_name="config.yaml")
print('preparing training dataset')
train_dataset = ClassDataset(cfg, mode = 'train', same_class_in_different_file=True)
print(train_dataset.seg_meta.keys())
print('preparing val dataset')
val_dataset = FileDataset(cfg)
print(val_dataset.seg_meta.keys())

batch_sampler = TaskBatchSampler(cfg, train_dataset.classes, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_sampler= batch_sampler,collate_fn=batch_sampler.get_collate_fn())
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
model = ProtoMAML(cfg)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
for epoch in range(100):
    model.train_loop(train_loader, optimizer)
    model_dir = cfg.checkpoint.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_file = os.path.join(model_dir, '{:d}.pth'.format(epoch))
    torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg}, save_file)
    val_dataset = FileDataset(cfg,val=True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    df_all_time, report = model.test_loop(val_loader)