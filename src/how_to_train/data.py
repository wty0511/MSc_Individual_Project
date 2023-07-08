# code modified how to train your maml
# https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
from torchvision import transforms
from PIL import ImageFile
from src.utils.file_dataset import *
from src.utils.class_dataset import *




class MetaLearningSystemDataLoader(object):
    def __init__(self, cfg):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.cfg = cfg
        self.num_of_gpus = 1
        self.batch_size = cfg.train.task_batch_size
        self.total_train_iters_produced = 0


        self.train_dataset = ClassDataset(cfg, mode='train', same_class_in_different_file=True, debug=False)
        self.val_dataset = FileDataset(cfg, val=True, debug=False)
        self.test_dataset = FileDataset(cfg, val=False, debug=False)
        batch_sampler = TaskBatchSampler(cfg, self.train_dataset.classes, len(self.train_dataset))
        self.train_loader = DataLoader(self.train_dataset, batch_sampler= batch_sampler,collate_fn=batch_sampler.get_collate_fn())
        

        self.val_loader = DataLoader(self.val_dataset, batch_size = 1, shuffle = False)
        self.test_loader = DataLoader(self.test_dataset, batch_size = 1, shuffle = False)
        
    def get_train_batches(self):
        # print('len', len(self.train_loader))
        return self.train_loader
    
    def get_val_batches(self):
        return self.val_loader
    
    def get_test_batches(self):
        return self.test_loader
        

        
