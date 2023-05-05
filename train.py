
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

    
if __name__ == "__main__":

    if not GlobalHydra().is_initialized():
        initialize(config_path="./")
    # Compose the configuration
    cfg = compose(config_name="config.yaml")
    print('preparing training dataset')
    train_dataset = ClassDataset(cfg)
    print('preparing val dataset')
    val_dataset = FileDataset(cfg)
    train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(cfg, train_dataset.classes, len(train_dataset)))
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    model = train(train_loader, val_loader, cfg)

