import librosa
import os
import numpy as np
import yaml
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from src.utils.feature_extractor import Feature_Extractor

from src.utils.feature_extractor import *
          



if __name__ == "__main__":
    if not GlobalHydra().is_initialized():
        initialize(config_path=".")
    # Compose the configuration
    cfg = compose(config_name="config.yaml")
    preprecess(cfg)