# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
from src.models.ProtoMAML import *
# from src.models.ProtoMAMLcopy import *
# from src.models.ProtoMAML_loss import *
from src.models.ProtoMAML_temperature import *
from src.models.MAML import *
# from src.models.SiameseMAML import *
from src.models.SiameseMAML_copy import *
from src.models.ProtoMAML_proxy import *
# from src.models.SiameseMAML_sigmoid import *
from src.models.ProtoMAML_query import *
from src.models.MAML_proto import *
from src.models.ProtoMAML_grad import *
import os
import sys
import omegaconf
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
from src.models.MAML2 import *
from src.models.ProtoMAML_lr import *
from src.models.MAML_lr import *
from src.models.MAML_proto_lr import *
# from src.models.TrinetMAML import *
from src.models.ProtoMAMLfw import *
from src.models.TrinetMAML_copy import *
from src.models.MAML_proxy import MAML_proxy

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
SEED = 12
set_seed(SEED)


debug = False
print_loss = False


if not GlobalHydra().is_initialized():
    initialize(config_path="./")
# Compose the configuration
# F

# cfg = compose(config_name="config.yaml")

model_name = 'ProtoMAML'  # ProtoMAML, ProtoMAMLfw, ProtoMAML_query, ProtoMAML_grad, ProtoMAML_temp, ProtoMAML_proxy, MAML, SNNMAML, TNNMAML, MAML_proxy

if model_name == 'MAML_proxy':
    cfg = compose(config_name="config_maml_proxy.yaml")
    model = MAML_proxy(cfg)
    
if model_name == 'ProtoMAML':
    cfg = compose(config_name="config_protomaml.yaml")
    model = ProtoMAML(cfg)
    
    
if model_name == 'ProtoMAML_lr':
    cfg = compose(config_name="config_protomaml_lr.yaml")
    model = ProtoMAML_lr(cfg)
elif model_name == 'ProtoMAML_temp':
    cfg = compose(config_name="config_protomaml_temp.yaml")
    model = ProtoMAML_temp(cfg)
elif model_name == 'ProtoMAML_grad':

    cfg = compose(config_name="config_protomaml_grad.yaml")
    model = ProtoMAML_grad(cfg)
    
elif model_name == 'ProtoMAML_proxy':
    cfg = compose(config_name="config_protomaml_proxy.yaml")
    model = ProtoMAML_proxy(cfg)

elif model_name == 'MAML':
    cfg = compose(config_name="config_maml.yaml")
    model = MAML(cfg)
    
elif model_name == 'MAML_lr':
    cfg = compose(config_name="config_maml.yaml")
    model = MAML_lr(cfg)

elif model_name == 'MAML2':
    cfg = compose(config_name="config_maml2.yaml")
    model = MAML2(cfg)


elif model_name == 'MAML_proto':
    cfg = compose(config_name="config_maml_proto.yaml")
    model = MAML_proto(cfg)
    
elif model_name == 'MAML_proto_lr':
    cfg = compose(config_name="config_maml_proto_lr.yaml")
    model = MAML_proto_lr(cfg)
    
elif model_name == 'TNNMAML':
    cfg = compose(config_name="config_mamltnn.yaml")
    model = TNNMAML(cfg)

elif model_name == 'SNNMAML':
    cfg = compose(config_name="config_mamlsnn.yaml")
    model = SNNMAML(cfg)
    
print('preparing training dataset')
train_dataset = ClassDataset(cfg, mode = 'train', same_class_in_different_file=True, debug= debug)
if print_loss:
    train_dataset.length = 20
print(len(train_dataset))
print(train_dataset.seg_meta.keys())
print('preparing val dataset')
val_dataset = FileDataset(cfg,val=True, debug= debug)
print(val_dataset.seg_meta.keys())
best_f1 = 0
batch_sampler = TaskBatchSampler(cfg, train_dataset.classes, len(train_dataset))

train_loader = DataLoader(train_dataset, batch_sampler= batch_sampler,collate_fn=batch_sampler.get_collate_fn())
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
# pretrain_model = torch.load('/root/task5_2023/Checkpoints/pretrain_conv/Model/best_model.pth')
# pretrain_dict = pretrain_model['state']
# pretrain_dict = {f'encoder.{k}': v for k, v in pretrain_dict.items()}


test_dataset = FileDataset(cfg,val=False, debug= debug)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)



    
# model = ProtoMAML_temp(cfg)
# model = ProtoMAML_grad(cfg)
# model = ProtoMAML_query(cfg)

# model = SNNMAML(cfg)

# model = TNNMAML(cfg)
# model =ProtoMAML_proxy(cfg)
# model = ProtoMAMLfw(cfg)
# print(pretrain_model['state'].keys())
# model.feature_extractor.load_state_dict(pretrain_dict)

print(len(train_loader))
model = model.cuda()
model_dir = cfg.checkpoint.model_dir
model_dir = normalize_path(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

# optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
# if model_name == 'MAML':
#     optimizer = optim.Adam([
#     {'params': model.feature_extractor.fc.parameters(), 'lr': 0.0001},  
#     {'params': model.feature_extractor.conv.parameters()}  
# ], lr=0.001)  # 默认学习率
model.train()

no_imporve = 0

train_loss_list = []
val_loss_list = []
test_loss_list = []

val_f1_list = []
test_f1_list = []

config_dir = normalize_path(cfg.checkpoint.exp_dir)
if not os.path.exists(os.path.dirname(config_dir)):
    os.makedirs(os.path.dirname(config_dir))
config_dir = os.path.join(config_dir,'config.json')
with open(config_dir, 'w') as outfile:
    json.dump( omegaconf.OmegaConf.to_container(cfg, resolve=True), outfile,indent=2)
    
for epoch in range(40):
    model.train()
    train_loss = model.train_loop(train_loader, optimizer)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(train_loss)
    train_loss_list.append(train_loss)
    save_file = os.path.join(model_dir, '{:d}.pth'.format(epoch))
    # if epoch % cfg.checkpoint.save_freq == 0:
    #     torch.save({'epoch':epoch, 'state':model.state_dict(), 'config':cfg}, save_file)
    model.eval()
    df_all_time, report, threshold, val_loss = model.test_loop(val_loader, mode = 'val', fix_shreshold=0.5)
    val_loss_list.append(val_loss)
    f1 = report['overall_scores']['fmeasure (percentage)']/100
    val_f1_list.append(f1)
    no_imporve +=1

    if f1 > best_f1:
        no_imporve = 0
        best_f1 = f1
        save_file = os.path.join(model_dir, 'best_model.pth')
        print(save_file)
        torch.save({'state':model.state_dict(), 'config':cfg, 'f1':best_f1, 'threshold' : threshold}, save_file)
        print("best model! save...")
        report_dir = normalize_path(cfg.checkpoint.report_dir)
        report_dir = os.path.join(report_dir,'val_report_best.json')
        if not os.path.exists(os.path.dirname(report_dir)):
            os.makedirs(os.path.dirname(report_dir))
        with open(report_dir, 'w') as outfile:
            json.dump(report, outfile)
    model.eval()
    df_all_time, report, threshold, test_loss = model.test_loop(test_loader, mode = 'test', fix_shreshold=0.5)
    print(test_loss)
    test_loss_list.append(test_loss)
    f1 = report['overall_scores']['fmeasure (percentage)']/100
    test_f1_list.append(f1)
    if no_imporve == 15:
        break
print('train_loss', train_loss_list)
print('train_loss', val_loss_list)
print('train_loss', test_loss_list)
print('train_loss', val_f1_list)
print('train_loss', test_f1_list)

loss_all = np.array([train_loss_list, val_loss_list, test_loss_list]).astype(float)
data_show = np.array([train_loss_list, val_f1_list, test_f1_list]).astype(float)
print(loss_all)
print(data_show)

loss_dir = os.path.join(normalize_path(cfg.checkpoint.report_dir),'loss.npy')
np.savetxt(loss_dir, loss_all, delimiter=',', fmt='%.3f')
data_dir = os.path.join(normalize_path(cfg.checkpoint.report_dir),'f1.npy')
np.savetxt(data_dir, data_show, delimiter=',', fmt='%.3f')

# np.savetxt(loss_dir, loss_all, delimiter=',', fmt='%.3f')
