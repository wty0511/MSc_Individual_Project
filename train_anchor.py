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
from src.models.ConvNet import *
from src.models.anchor_network import AnchorNet
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
cfg = compose(config_name="config_anchor.yaml")
print('preparing training dataset')

train_dataset = ClassDataset(cfg, mode = 'anchor', same_class_in_different_file=True, debug= debug)

print(train_dataset.seg_meta.keys())


class_sampler = ClassSampler(cfg, train_dataset.classes, len(train_dataset))
# print(len(train_dataset))
train_loader = DataLoader(train_dataset, sampler= class_sampler, batch_size = 64)
val_dataset = FileDataset(cfg,val=True, debug= debug)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)


model = AnchorNet(cfg).cuda()
model_dir = cfg.checkpoint.model_dir
model_dir = normalize_path(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print('len',len(train_dataset))
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
# for name in model.named_parameters():
#     print(name[0])
best_f1 = 0
no_imporve = 0

for epoch in range(10):
    model.train()
    model.train_loop(train_loader, optimizer)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    save_file = os.path.join(model_dir, '{:d}.pth'.format(epoch))
    if epoch % 10 == 0:
        torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg}, save_file)
    model.eval()
    df_all_time, report, threshold = model.test_loop(val_loader, mode = 'val', fix_shreshold = 0.5)
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