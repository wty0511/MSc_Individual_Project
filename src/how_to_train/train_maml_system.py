
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
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra().is_initialized():
    initialize(config_path="../../")
# Compose the configuration
cfg = compose(config_name="config.yaml")

# Combines the arguments, model, data and experiment builders to run an experiment
model = MAMLFewShotClassifier(cfg)

data = MetaLearningSystemDataLoader(cfg)
maml_system = ExperimentBuilder(cfg = cfg, model=model, data=data)
maml_system.run_experiment()
