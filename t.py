# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
from src.models.ProtoMAML import *
# from src.models.ProtoMAMLcopy import *
# from src.models.ProtoMAML_loss import *
from src.models.ProtoMAML_temperature import *
from src.models.MAML import *
# from src.models.SiameseMAML import *
from src.models.SiameseMAML_copy2 import *
from src.models.ProtoMAML_proxy import *
# from src.models.SiameseMAML_sigmoid import *
from src.models.ProtoMAML_query import *
from src.models.ProtoMAML_grad import *
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
# from src.models.TrinetMAML import *
from src.models.ProtoMAMLfw import *
from src.models.TrinetMAML_copy import *
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
SEED = 42
set_seed(SEED)


debug = False
if not GlobalHydra().is_initialized():
    initialize(config_path="./")
# Compose the configuration
cfg = compose(config_name="config.yaml")
print('preparing training dataset')
train_dataset = ClassDataset(cfg, mode = 'train', same_class_in_different_file=True, debug= debug)
print(len(train_dataset))
print(train_dataset.seg_meta.keys())
print('preparing val dataset')
val_dataset = FileDataset(cfg,val=True, debug= debug)
print(val_dataset.seg_meta.keys())
best_f1 = 0
batch_sampler = TaskBatchSampler(cfg, train_dataset.classes, len(train_dataset))

train_loader = DataLoader(train_dataset, batch_sampler= batch_sampler,collate_fn=batch_sampler.get_collate_fn())
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
pretrain_model = torch.load('/root/task5_2023/Checkpoints/pretrain/Model/best_model.pth')
pretrain_dict = pretrain_model['state']
pretrain_dict = {f'encoder.{k}': v for k, v in pretrain_dict.items()}
# model = ProtoMAML_temp(cfg)
# model = ProtoMAML_grad(cfg)
# model = ProtoMAML_query(cfg)
# model = ProtoMAML(cfg)
# model = SNNMAML(cfg)
# model = MAML(cfg)
# model = TNNMAML(cfg)
model =ProtoMAML_proxy(cfg)
# model = ProtoMAMLfw(cfg)
print(pretrain_model['state'].keys())
# model.feature_extractor.load_state_dict(pretrain_dict)

print(len(train_loader))
model = model.cuda()
model_dir = cfg.checkpoint.model_dir
model_dir = normalize_path(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
model.train()

no_imporve = 0

for epoch in range(100//cfg.train.n_way):
    model.train()
    model.train_loop(train_loader, optimizer)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    save_file = os.path.join(model_dir, '{:d}.pth'.format(epoch))
    if epoch % 10 == 0:
        torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg}, save_file)
    model.eval()
    df_all_time, report, threshold = model.test_loop(val_loader)
    f1 = report['overall_scores']['fmeasure (percentage)']
    no_imporve +=1
    if no_imporve == 30:
        break
    if f1 > best_f1:
        no_imporve = 0
        best_f1 = f1
        save_file = os.path.join(model_dir, 'best_model.pth')
        print(save_file)
        torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg, 'f1':best_f1, 'model_name':'ProtoMAML', 'threshold' : threshold}, save_file)
        print("best model! save...")
        report_dir = normalize_path(cfg.checkpoint.report_dir)
        report_dir = os.path.join(report_dir,'val_report_best.json')
        if not os.path.exists(os.path.dirname(report_dir)):
            os.makedirs(os.path.dirname(report_dir))
        with open(report_dir, 'w') as outfile:
            json.dump(report, outfile)