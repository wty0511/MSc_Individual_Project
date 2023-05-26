
import librosa
import os
import numpy as np
import yaml
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm


# Feature Extraction
class Feature_Extractor():
    def __init__(self,config):
        self.config = config
        config.features.fmax = config.features.sr / 2.0
        self.sr = config.features.sr
        self.n_mels = config.features.n_mels
        self.n_fft = config.features.n_fft
        self.hop_length = config.features.hop_length
        self.win_length = config.features.win_length
        self.fmin = config.features.fmin
        self.fmax = config.features.fmax
        self.feature_list = config.features.feature_list.split("&")
    def load_audio(self,path):
        audio, sr = librosa.load(path, sr=self.sr)
        return audio, sr

    def extract_mel(self,audio):
        mel = librosa.feature.melspectrogram(audio, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, fmin=self.fmin, fmax=self.fmax).astype(np.float32)
        log_mel = None
        if 'logmel' in self.feature_list:
            # ref = ? and top_db = ?
            log_mel = librosa.power_to_db(mel)
        return mel, log_mel
    def norm(self, y):
        return y / np.max(np.abs(y))
    def extract_pcen(self,audio):
        # as sugested
        audio = self.norm(audio)
        audio = audio*(2**32)
        mel = librosa.feature.melspectrogram(audio, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, fmax=self.fmax)
        # mel or log_mel
        pcen = librosa.pcen(mel, sr=self.sr).astype(np.float32)
        return pcen
    
    def extract_mfcc(self,log_mel):
        return librosa.feature.mfcc(S=log_mel)
    
    def extract_all_features(self,path):
        res = {}
        audio, _ = self.load_audio(path)
        if "mel" in self.feature_list:
            mel, log_mel = self.extract_mel(audio)
            res["mel"] = mel
            if log_mel is not None and 'logmel' in self.feature_list:
                res["logmel"] = log_mel
                
        if "mfcc" in self.feature_list:
            if not 'logmel' in self.feature_list:
                raise Exception("Log mel should be True when extracting mfcc")
            res["mfcc"] = self.extract_mfcc(res["logmel"])
        
        if "pcen" in self.feature_list:
            res["pcen"] = self.extract_pcen(audio)
        return res
    def load_features(self, path):
        res = {}
        for feature in self.feature_list:
            res[feature] = np.load(path + feature + ".npy")
        return np.load(path)
    
    
    
def normalize_path(path=''):
    
    path = path.split("/")
    if os.name == 'posix':
        path = os.path.join(*path)
        path = '/' + path
        # print("Current operating system is Linux")
    else:
        # print("Current operating system is Windows")
        print(path)
        drive = path[1].upper() + ':' +os.sep
        path = os.path.join(drive, *path[2:])
    path = os.path.normpath(path)
    print(path)
    return path

# walk files
def walk_files(file, debug, file_extension = ('.wav', '.csv', '.npy')):
    classes = {'BV':0, 
               'HT':0, 
               'JD':0, 
               'MT':0,
               'HV':0,
               'DC':0,
               'ME':0,
               'ML':0,
               'HV':0,
               'PB':0}
    for path, subdirs, files in os.walk(file):
        for name in files:
            if name.endswith(file_extension):
                c = path.split(os.sep)[-1]
                
                if debug:
                    if classes[c] ==1:
                        continue
                    else:
                        classes[c] = 1
                yield os.path.join(path, name)



def audio2feature(path, feature_name):
    save_path = path.replace("Data_Set", "Features").replace(".wav", "_" + feature_name + ".npy")
    return save_path

def preprecess(cfg):
    feature_extractor = Feature_Extractor(cfg)
    print("Extracting features...")
    feature_list = cfg.features.feature_list.split("&")
    # print(normalize_path(cfg.path.data_dir))
    for file in tqdm(walk_files(normalize_path(cfg.path.data_dir), '.wav')):
        feature = feature_extractor.extract_all_features(file)
        for feature_name in feature_list:
            save_path = audio2feature(file, feature_name)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            np.save(save_path, feature[feature_name])

if __name__ == "__main__":
    if not GlobalHydra().is_initialized():
        initialize(config_path="../../")
    # Compose the configuration
    cfg = compose(config_name="config.yaml")
    normalize_path(cfg.path.train_dir)