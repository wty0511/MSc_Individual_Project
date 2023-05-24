
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import torch
from src.models.ProtoMAML import ProtoMAML
from utils.file_dataset import *
if not GlobalHydra().is_initialized():
    initialize(config_path="../")
# Compose the configuration
cfg = compose(config_name="config.yaml")
model_dir = cfg.checkpoint.model_dir
save_file = os.path.join(model_dir, 'best_model.pth')


# 加载模型
checkpoint = torch.load('/root/task5_2023/Checkpoints/protoMAMLCNN6/Model/best_model.pth')
print(checkpoint.keys())
# 从checkpoint中获取模型的状态和配置信息
model_state = checkpoint['state']
config = checkpoint['config']
config = cfg

epoch = checkpoint['epoch']
print('epoch:', epoch)
print('threshold:', checkpoint['threshold'])
# 创建一个新的模型实例
model = ProtoMAML(config).to('cuda' if torch.cuda.is_available() else 'cpu')

# 将保存的状态加载到新的模型实例中
model.load_state_dict(model_state)

val_dataset = FileDataset(cfg,val=False)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
df_all_time, report,best_threshold = model.test_loop(val_loader, fix_shreshold=checkpoint['threshold'])
acc = report['overall_scores']['fmeasure (percentage)']


# val_dataset = ClassDataset(cfg, mode = 'val',same_class_in_different_file = False)
# val_loader = DataLoader(val_dataset, batch_sampler=BatchSampler(cfg, val_dataset.classes, len(val_dataset)))

# model.test_loop_task(val_loader)

        