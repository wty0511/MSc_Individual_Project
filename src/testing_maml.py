
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(src_dir)

import torch
from src.models.ProtoMAML import ProtoMAML
from src.models.ProtoMAML_temperature import ProtoMAML_temp
# from src.models.ProtoMAMLcopy import *
# from src.models.ProtoMAML_loss import *
# from src.models.SiameseMAML import *
from src.models.SiameseMAML_copy import *
from src.models.ProtoMAMLfw import *
from src.models.MAML import *
from src.models.MAML2 import *
# from src.models.TrinetMAML import *
from src.models.TrinetMAML_copy import *
from src.utils.file_dataset import *
from src.models.ProtoMAML_proxy import *
from src.models.ProtoMAML_refine import *
from src.models.MAML_proto import *
from src.models.MAML_proto_lr import *
from src.models.MAML_proxy import *
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
SEED = 22
set_seed(SEED)

# cfg = compose(config_name="config.yaml")
# model_dir = cfg.checkpoint.model_dir
# save_file = os.path.join(model_dir, 'best_model.pth')
# save_file = r"/root/task5_2023/Checkpoints/proxyMAML_5way_cos/Model/best_model.pth"
# save_file = r"/root/task5_2023/Checkpoints/FOMAMLTNN_2way/Model/best_model.pth"
# save_file = r"/root/task5_2023/Checkpoints/proxyMAML_10way3/Model/best_model.pth"
# save_file = r"/root/task5_2023/Checkpoints/FOMAMLTNN_5way/Model/best_model.pth"

<<<<<<< HEAD
save_file = r"/root/task5_2023/Checkpoints/Proto_MAML_10way_5step_convnet_3/Model/best_model.pth"
=======
save_file = r"/root/task5_2023/Checkpoints/Proto_MAML_5way_5step_convnet_3/Model/best_model.pth"

save_file = r"/root/task5_2023/Checkpoints/Proxy_MAML_10way_5step_convnet_alpha32_2/Model/best_model.pth"

>>>>>>> 07b87901c58201da2e692146144eab941fdae202

# save_file = r"/root/task5_2023/Checkpoints/recent/FOMAML/Model/best_model.pth"
# 加载模型
checkpoint = torch.load(save_file)
print(checkpoint.keys())


is_val = False  
# 从checkpoint中获取模型的状态和配置信息
model_state = checkpoint['state']
config = checkpoint['config']
# config.train.inner_step = 5
cfg = config
# print(model_state.keys())
# checkpoint['config']['checkpoint']['experiment_name'] = 'FOMAMLProtoNet'
# torch.save(checkpoint, save_file)
# config = cfg
print(config)
print('threshold:', checkpoint['threshold'])
print('f1:', checkpoint['f1'])
# model = ProtoMAML_temp(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# print('f1', checkpoint['f1'])
# 创建一个新的模型实例
# model = ProtoMAML(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = ProtoMAML_refine(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = ProtoMAMLfw(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = MAML(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = MAML2(config).to('cuda' if torch.cuda.is_available() else 'cpu')

# model = MAML_proto(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = MAML_proto_lr(config).to('cuda' if torch.cuda.is_available() else 'cpu')

# model = SNNMAML(config).to('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoMAML_proxy(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = TNNMAML(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# model = MAML_proxy(config).to('cuda' if torch.cuda.is_available() else 'cpu')
# 将保存的状态加载到新的模型实例中
# for name, param in model.named_parameters():
#     print(name)
#     print(param.shape)
model.load_state_dict(model_state)
# pretrain_model = torch.load('/root/task5_2023/Checkpoints/pretrain_conv/Model/best_model.pth')
# pretrain_dict = pretrain_model['state']
# model.feature_extractor.load_state_dict(pretrain_dict)


# for name,  param in model.named_parameters():
#     print(name)
model.eval()

# checkpoint['threshold'] = None
val_dataset = FileDataset(cfg,val=is_val,debug=False)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
# checkpoint['threshold'] = None
df_all_time, report, best_threshold, loss = model.test_loop(val_loader, fix_shreshold=checkpoint['threshold'], mode = 'val' if is_val else 'test')

report_dir = normalize_path(cfg.checkpoint.report_dir)
report_dir = os.path.join(report_dir,'test_report_best.json')
# print(report_dir)
if not os.path.exists(os.path.dirname(report_dir)):
    os.makedirs(os.path.dirname(report_dir))
print(report)
# with open(report_dir, 'w') as outfile:
#     json.dump(report, outfile)

# val_dataset = ClassDataset(cfg, mode = 'val',same_class_in_different_file = False)
# val_loader = DataLoader(val_dataset, batch_sampler=BatchSampler(cfg, val_dataset.classes, len(val_dataset)))

# model.test_loop_task(val_loader)

