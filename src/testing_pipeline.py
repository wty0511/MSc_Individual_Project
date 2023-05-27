
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import torch
from src.models.ProtoNet import *
from utils.file_dataset import *
if not GlobalHydra().is_initialized():
    initialize(config_path="../")
# Compose the configuration
cfg = compose(config_name="config.yaml")
model_dir = cfg.checkpoint.model_dir
# save_file = os.path.join(model_dir, 'best_model.pth')
save_file = '/root/task5_2023/Checkpoints/protoCNN5/Model/best_model.pth'
print(save_file)

# 加载模型
checkpoint = torch.load(save_file)
checkpoint['threshold'] = 0.7999999999999999

# 从checkpoint中获取模型的状态和配置信息
model_state = checkpoint['state']
config = checkpoint['config']
config = cfg

epoch = checkpoint['epoch']
print('epoch:', epoch)
# 创建一个新的模型实例
model = ProtoNet(config).to('cuda' if torch.cuda.is_available() else 'cpu')

# 将保存的状态加载到新的模型实例中
model.load_state_dict(model_state)
model.eval()

val_dataset = FileDataset(cfg,val=True, debug=False)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
df_all_time, report, threshold = model.test_loop(val_loader, fix_shreshold=checkpoint['threshold'])
acc = report['overall_scores']['fmeasure (percentage)']
