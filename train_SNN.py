# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
# from src.models.ProtoMAML import *
# from src.models.ProtoMAMLcopy import *
# from src.models.ProtoMAML_loss import *



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

print('preparing training dataset')
model_name = "TNN"

if model_name == 'SNN':
    train_dataset = ClassPairDataset(cfg, mode = 'train', same_class_in_different_file=True, debug= debug)
    class_sampler = ClassSampler(cfg, train_dataset.classes, len(train_dataset))
    train_loader = DataLoader(train_dataset, sampler= class_sampler, batch_size = 128)
    cfg = compose(config_name="config_snn.yaml")
    model = SNN(cfg)
elif model_name == 'TNN':
    cfg = compose(config_name="config_tnn.yaml")
    train_dataset = ClassDataset(cfg, mode = 'TNN', same_class_in_different_file=True, debug= debug)
    class_sampler = ClassSampler(cfg, train_dataset.classes, len(train_dataset))
    train_loader = DataLoader(train_dataset, sampler= class_sampler, batch_size = 512)
    model =TriNet(cfg)

print(len(train_loader))
print(train_dataset.seg_meta.keys())
print('preparing val dataset')
val_dataset = FileDataset(cfg,val=True, debug= debug)
print(val_dataset.seg_meta.keys())
best_f1 = 0
print(train_dataset.seg_meta.keys())









val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)


# for param in model.parameters():
#     param.fast = None

model = model.cuda()
model_dir = cfg.checkpoint.model_dir
model_dir = normalize_path(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print('len',len(train_dataset))
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
for epoch in range(100):
    model.train()
    model.train_loop(train_loader, optimizer)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.eval()
    save_file = os.path.join(model_dir, '{:d}.pth'.format(epoch))
    if epoch % 10 == 0:
        torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg}, save_file)
    df_all_time, report, threshold = model.test_loop(val_loader, mode = 'val', fix_shreshold = 0.5)
    f1 = report['overall_scores']['fmeasure (percentage)']
    if f1 > best_f1:
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