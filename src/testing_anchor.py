
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import torch
from src.models.ProtoNet import *
from utils.file_dataset import *
from src.models.anchor_network import AnchorNet
if not GlobalHydra().is_initialized():
    initialize(config_path="../")
# Compose the configuration

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
SEED = 42
set_seed(SEED)

save_file = r"/root/task5_2023/Checkpoints/test2/Model/best_model.pth"
checkpoint = torch.load(save_file)
cfg = checkpoint['config']
# cfg = compose(config_name="config.yaml")
model_dir = cfg.checkpoint.model_dir
# save_file = os.path.join(model_dir, 'best_model.pth')


# 加载模型

# checkpoint['config']['checkpoint']['experiment_name'] = 'ProtoNet'
print(checkpoint['config'])
# 从checkpoint中获取模型的状态和配置信息
model_state = checkpoint['state']
config = checkpoint['config']
epoch = checkpoint['epoch']
print('epoch:', epoch)
# 创建一个新的模型实例
model = AnchorNet(config).to('cuda' if torch.cuda.is_available() else 'cpu')

# 将保存的状态加载到新的模型实例中
model.load_state_dict(model_state)
model.eval()
# torch.save(checkpoint, save_file)
# checkpoint['threshold'] = None
val_dataset = FileDataset(cfg,val=False, debug=False)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
print(checkpoint['threshold'])
df_all_time, report, threshold = model.test_loop(val_loader, fix_shreshold=checkpoint['threshold'])
acc = report['overall_scores']['fmeasure (percentage)']

report_dir = normalize_path(cfg.checkpoint.report_dir)
report_dir = os.path.join(report_dir,'test_report_best.json')
print(report_dir)
if not os.path.exists(os.path.dirname(report_dir)):
    os.makedirs(os.path.dirname(report_dir))
    
with open(report_dir, 'w') as outfile:
    json.dump(report, outfile)
    