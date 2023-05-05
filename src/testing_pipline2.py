
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import torch
from src.models.ProtoMAML import ProtoMAML
from src.utils.dataset_val import *
if not GlobalHydra().is_initialized():
    initialize(config_path="../")
# Compose the configuration
cfg = compose(config_name="config.yaml")
best_model_dir = cfg.checkpoint.best_model_dir
save_file = os.path.join(best_model_dir, 'best_model.pth')


# 加载模型
checkpoint = torch.load('/root/task5_2023/Checkpoints/expMAML/Model/18.pth')

# 从checkpoint中获取模型的状态和配置信息
model_state = checkpoint['state']
config = checkpoint['config']
config = cfg

epoch = checkpoint['epoch']
print('epoch:', epoch)
# 创建一个新的模型实例
model = ProtoMAML(config).to('cuda' if torch.cuda.is_available() else 'cpu')

# 将保存的状态加载到新的模型实例中
model.load_state_dict(model_state)

val_dataset = FileDataset(cfg,val=True)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
df_all_time, report = model.test_loop(val_loader)
acc = report['overall_scores']['fmeasure (percentage)']
        

# val_dataset = ClassDataset(cfg, mode = 'val')
# val_loader = DataLoader(val_dataset, batch_sampler=BatchSampler(cfg, val_dataset.classes, len(val_dataset)))

# model.test_loop_task(val_loader)

        