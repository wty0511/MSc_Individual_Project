
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import torch
from src.models.ProtoNet import * 
from src.models.SiameseNet import SNN
from src.models.TriNet import *
from utils.file_dataset import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
SEED = 42
set_seed(SEED)


if not GlobalHydra().is_initialized():
    initialize(config_path="../")
# Compose the configuration
cfg = compose(config_name="config.yaml")
model_dir = cfg.checkpoint.model_dir
# save_file = os.path.join(model_dir, 'best_model.pth')

save_file = '/root/task5_2023/Checkpoints/Trinetconvnet__semi_2/Model/best_model.pth'

print(save_file)

# 加载模型
checkpoint = torch.load(save_file)

# 从checkpoint中获取模型的状态和配置信息
model_state = checkpoint['state']
config = checkpoint['config']
# config = cfg
print(config)
print(checkpoint['threshold'])
# 创建一个新的模型实例
model = TriNet(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = SNN(config).to('cuda' if torch.cuda.is_available() else 'cpu')

# 将保存的状态加载到新的模型实例中
model.load_state_dict(model_state)
model.eval()

val_dataset = FileDataset(cfg,val=False, debug=False)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
df_all_time, report, threshold = model.test_loop(val_loader, fix_shreshold=checkpoint['threshold'])
acc = report['overall_scores']['fmeasure (percentage)']

report_dir = normalize_path(config.checkpoint.report_dir)
report_dir = os.path.join(report_dir,'test_report_best.json')
print(report_dir)
if not os.path.exists(os.path.dirname(report_dir)):
    os.makedirs(os.path.dirname(report_dir))

with open(report_dir, 'w') as outfile:
    json.dump(report, outfile)