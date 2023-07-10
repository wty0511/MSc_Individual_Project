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
from src.models.backbone import PretrainClassifier
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
cfg = compose(config_name="config_pretrain.yaml")
print('preparing training dataset')

train_dataset = ClassDataset(cfg, mode = 'pretrain', same_class_in_different_file=True, debug= debug)

print(train_dataset.seg_meta.keys())


class_sampler = ClassSampler(cfg, train_dataset.classes, len(train_dataset))

train_loader = DataLoader(train_dataset, sampler= class_sampler, batch_size = 64)

model = PretrainClassifier_large()
# model = PretrainClassifier()
model = model.cuda()
model_dir = cfg.checkpoint.model_dir
model_dir = normalize_path(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print('len',len(train_dataset))
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
for epoch in range(50):
    model.train()
    acc = []
    acc_best = 0
    for batch in train_loader:
        data, label = batch
        preds = model(data)
        loss = F.cross_entropy(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc.append((preds.argmax(dim=1) == label).float().mean().item())
    acc_epoch = np.mean(acc)
    
    print('epoch',epoch,'acc',acc_epoch)
    save_file = os.path.join(model_dir, '{:d}.pth'.format(epoch))
    if epoch % 10 == 0:
        # torch.save({'epoch':epoch, 'state':model.encoder.state_dict(), 'config':cfg}, save_file)
        torch.save({'epoch':epoch, 'state':model.conv.state_dict(), 'config':cfg}, save_file)
    if acc_epoch > acc_best:
        acc_best = acc_epoch
        save_file = os.path.join(model_dir, 'best_model.pth')
        torch.save({'epoch':epoch, 'state':model.conv.state_dict(), 'config':cfg}, save_file)
        # torch.save({'epoch':epoch, 'state':model.encoder.state_dict(), 'config':cfg}, save_file)
        print("best model! save...")
