import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import numpy as np
import torch
import torch.optim
from src.models.ProtoNet import ProtoNet
import random
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from src.utils.dataset_train import *
from src.utils.dataset_val import *

# GPT
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

SEED = 42



# 基于colser look
def train(train_loader, val_loader, config):
    print('start training...')
    print('seed everything...')
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.train.model_type == 'protonet':
        model = ProtoNet(config)
    print('backbone:', config.train.backbone)
    print(model)
    print('device:', device)
    model = model.to(device)
    epoches = config.train.epoches
    if config.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    elif config.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.train.lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)
    else:
       raise ValueError('Unknown optimizer')

    print('optimizer:', config.train.optimizer)
    best_acc = 0       
    best_model_dir = config.checkpoint.best_model_dir
    for epoch in range(epoches):
        model.train()
        avg_loss = model.train_loop(train_loader,  optimizer )
        model.eval()

        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        df_all_time, report = model.test_loop(val_loader)
        print(report)
        acc = report['overall_scores']
        if acc > best_acc :
            print("best model! save...")
            best_acc = acc
            save_file = os.path.join(best_model_dir, 'best_model.pth')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':config}, save_file)

        if (epoch % config.checkpoint.save_freq==0) or (epoch==epoches-1):
            save_file = os.path.join(best_model_dir, '{:d}.pth'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':config}, save_file)
        print("Epoch [{}/{}], Train Loss: {:.4f},   Val F1: {:.4f}"
          .format(epoch+1, epoches, avg_loss, acc))
        break
    return model

    
    
if __name__ == "__main__":

    if not GlobalHydra().is_initialized():
        initialize(config_path="../")
    # Compose the configuration
    cfg = compose(config_name="config.yaml")
    print('preparing training dataset')
    train_dataset = TrainDataset(cfg)
    print('preparing val dataset')
    val_dataset = ValDataset(cfg)
    train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(cfg, train_dataset.classes, len(train_dataset)))
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    model = train(train_loader, val_loader, cfg)

