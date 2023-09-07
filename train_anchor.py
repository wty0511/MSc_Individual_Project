# from src.models.ProtoMAML import *
# from src.models.ProtoMAMLcopy import *
# from src.models.ProtoMAML_loss import *

# This code is modified from https://github.com/wyharveychen/CloserLookFewShot

from src.models.SiameseNet import *
from src.models.TriNet import *
import os
import sys

import numpy as np
import torch
import torch.optim
from src.models.ProtoNet import ProtoNet
import random
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from src.utils.class_pair_dataset import *
from src.utils.file_dataset import *
from src.training_pipeline import train
from src.models.ConvNet import *
from src.models.anchor_network import AnchorNet
import omegaconf
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


SEED = 42
debug = False
best_f1 = 0
no_imporve = 0

set_seed(SEED)
if not GlobalHydra().is_initialized():
    initialize(config_path="./")
# Compose the configuration
cfg = compose(config_name="config_anchor.yaml")
print('preparing training dataset')
train_dataset = ClassDataset(cfg, mode = 'anchor', same_class_in_different_file=True, debug= debug)
# sample 5000 segments from training dataset
train_dataset.length = 5000
print(train_dataset.seg_meta.keys())
# select classes for training
class_sampler = ClassSampler(cfg, train_dataset.classes, len(train_dataset))
train_loader = DataLoader(train_dataset, sampler= class_sampler, batch_size = 512)
# each time sample one validation file
val_dataset = FileDataset(cfg,val=True, debug= debug)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
# proxy_anchor_loss
model = AnchorNet(cfg).cuda()
model_dir = cfg.checkpoint.model_dir
model_dir = normalize_path(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print('len',len(train_dataset))
model_optimizer = torch.optim.Adam(model.feature_extractor.parameters(), lr=cfg.train.lr)
loss_optimizer = torch.optim.Adam(model.loss_func.parameters(), lr=cfg.train.lr)
# for name in model.named_parameters():
#     print(name[0])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


config_dir = normalize_path(cfg.checkpoint.exp_dir)
if not os.path.exists(os.path.dirname(config_dir)):
    os.makedirs(os.path.dirname(config_dir))
config_dir = os.path.join(config_dir,'config.json')
with open(config_dir, 'w') as outfile:
    json.dump( omegaconf.OmegaConf.to_container(cfg, resolve=True), outfile,indent=2)
    
for epoch in range(100):
    model.train()
    model.train_loop(train_loader, loss_optimizer, model_optimizer)

    save_file = os.path.join(model_dir, '{:d}.pth'.format(epoch))
    # if epoch % cfg.checkpoint.save_freq == 0:
    #     torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg}, save_file)
    model.eval()
    df_all_time, report, threshold = model.test_loop(val_loader, mode = 'val', fix_shreshold = 0.5)
    f1 = report['overall_scores']['fmeasure (percentage)']
    no_imporve +=1
    if no_imporve == 15:
        break
    if f1 > best_f1:
        no_imporve = 0
        best_f1 = f1
        save_file = os.path.join(model_dir, 'best_model.pth')
        print(save_file)
        torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg, 'f1':best_f1, 'threshold' : threshold}, save_file)
        print("best model! save...")
        report_dir = normalize_path(cfg.checkpoint.report_dir)
        report_dir = os.path.join(report_dir,'val_report_best.json')
        if not os.path.exists(os.path.dirname(report_dir)):
            os.makedirs(os.path.dirname(report_dir))
        with open(report_dir, 'w') as outfile:
            json.dump(report, outfile)