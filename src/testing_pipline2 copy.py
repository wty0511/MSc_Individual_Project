
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import torch
from src.models.ProtoMAML import ProtoMAML
# from src.models.ProtoMAMLcopy import *
from utils.file_dataset import *
if not GlobalHydra().is_initialized():
    initialize(config_path="../")
# Compose the configuration
cfg = compose(config_name="config.yaml")
model_dir = cfg.checkpoint.model_dir
save_file = os.path.join(model_dir, 'best_model.pth')


# 加载模型
checkpoint = torch.load(r"/root/task5_2023/Checkpoints/protoMAMLCNN4 (dont change)/Model/best_model.pth")
config = checkpoint['config']
model_state = checkpoint['state']
for i in model_state:
    print(i)
    print(model_state[i].shape)