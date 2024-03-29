
# code modified how to train your maml
# https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch

import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(src_dir)
print(current_dir)
print(src_dir)
from src.how_to_train.experiment_builder import ExperimentBuilder
from src.how_to_train.data import MetaLearningSystemDataLoader
from src.how_to_train.few_shot_classifier import MAMLFewShotClassifier
from src.how_to_train.few_shot_classifier_with_head import MAMLFewShotClassifierWithHead
from src.how_to_train.few_shot_classifier_poxy import ProxyMAMLFewShotClassifier
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from src.how_to_train.few_shot_classifier_TNN import *

if not GlobalHydra().is_initialized():
    initialize(config_path="../../")
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
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
SEED = 42
set_seed(SEED)

cfg = compose(config_name="configmamlpp.yaml")

# Combines the arguments, model, data and experiment builders to run an experiment
# model = MAMLFewShotClassifier(cfg)
# model =TNNMAMLFewShotClassifier(cfg)
model = MAMLFewShotClassifierWithHead(cfg)
# model = ProxyMAMLFewShotClassifier(cfg)
data = MetaLearningSystemDataLoader(cfg)
maml_system = ExperimentBuilder(cfg = cfg, model=model, data=data)
# maml_system.run_experiment()

ckpt = r"/root/task5_2023/Checkpoints/MAMLPP_proto_10way_5step_convnet_6/Model/best_model.pth"

maml_system.run_experiment()

# maml_system.test(ckpt)
