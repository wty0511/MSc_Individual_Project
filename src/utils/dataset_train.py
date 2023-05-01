import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(src_dir)
from torch.utils.data import Dataset
from src.utils.feature_extractor import *
import pandas as pd
import random
from src.utils.helpers import *
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from src.utils.sampler import *
from torch.utils.data import DataLoader
import time
def meta_learning_collate_fn(batch):
    print(batch)
    support_set_indices = batch[0]['support_set']
    query_set_indices = batch[0]['query_set']

    return {
        'support_set': support_set_indices,
        'query_set': query_set_indices
    }
    
# def meta_learning_collate_fn(batch):
#     # Assuming the batch is a list of dictionaries
#     support_set_indices = batch[0]['support_set']
#     query_set_indices = batch[0]['query_set']

#     support_set_samples = [train_dataset.get_sample(idx) for idx in support_set_indices]
#     query_set_samples = [train_dataset.get_sample(idx) for idx in query_set_indices]

#     return {
#         'support_set': support_set_samples,
#         'query_set': query_set_samples
#     }

class TrainDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.feature_list = config.features.feature_list.split("&")
        self.train_dir = normalize_path(config.path.train_dir)
        self.feature_per_file = {}
        self.classes = set()
        self.seg_meta = {}
        self.sr = config.features.sr
        self.fps = self.sr / config.features.hop_length
        self.neg_prototype = self.config.train.neg_prototype
        self.collect_features()
        self.process_labels()
        self.class2index = self._class2index()
        self.index2class = self._index2class()
        self.length = int(3600 * 8 / self.config.features.segment_length)
    

    def __getitem__(self, class_name):
        #start_time = time.time()


        # print('getting item', class_name)
        selected_class_neg = class_name + '_neg'
        pos = self.get_pos_sample(class_name, self.config.features.segment_len_frame)
        neg = self.get_neg_sample(selected_class_neg, self.config.features.segment_len_frame)
        pos_index = self.class2index[class_name]
        
        neg_index = self.class2index[selected_class_neg]
        #end_time = time.time()
        #elapsed_time = end_time - start_time
        # print(f"代码执行时间为 {elapsed_time:.2f} 秒")
        if self.neg_prototype:
            return (class_name, pos, pos_index), (selected_class_neg, neg, neg_index)
        
        else:
            return(class_name, pos, pos_index), None
        
    def __len__(self):
        return(self.length )
    def collect_features(self):
        print("Collecting training set features...")
        for file in tqdm(walk_files(self.train_dir, file_extension = ('.wav'))):
            for feature in self.feature_list:
                save_path = audio2feature(file, feature)
                self.feature_per_file[file] = self.feature_per_file.get(file, {})
                self.feature_per_file[file][feature] = np.load(save_path)
    
    def process_labels(self):
        print("Processing labels...")
        for file in tqdm(walk_files(self.train_dir, file_extension = ('.csv'))):
            df = pd.read_csv(file)
            file = file.replace('.csv','.wav')
            # class name starts from the 4th column
            for column in df.columns[3:]:
                self.classes.add(column)
                pos_idx = df[df[column] == 'POS'].index.tolist()
                start = df['Starttime'][pos_idx].tolist()
                end = df['Endtime'][pos_idx].tolist()
                # print('start', start)
                # print('end', end)
                self.seg_meta[column] = self.seg_meta.get(column, {})
                self.seg_meta[column]['time_spane'] = self.seg_meta[column].get('time_spane', [])
                self.seg_meta[column]['duration'] = self.seg_meta[column].get('duration', [])

                for s, e in zip(start, end):
                    self.seg_meta[column]['time_spane'].append({'start': s, 'end': e, 'file': file})
                    self.seg_meta[column]['duration'].append(e - s)
                
                if self.seg_meta[column]['time_spane'] == []:
                    self.classes.remove(column)
                    self.seg_meta.pop(column)
                    continue
                
                if self.neg_prototype:
                    column_neg = column + '_neg'
                    self.seg_meta[column_neg] = self.seg_meta.get(column_neg, {})
                    self.seg_meta[column_neg]['time_spane'] = self.seg_meta[column_neg].get('time_spane', [])
                    self.seg_meta[column_neg]['duration'] = self.seg_meta[column_neg].get('duration', [])
                    end.insert(0, 0.0)
                    
                                 
                    for i in range(len(start)):
                        if start[i] > end[i]:
                            self.seg_meta[column_neg]['time_spane'].append({'start': end[i], 'end': start[i], 'file': file})
                            self.seg_meta[column_neg]['duration'].append(start[i] - end[i])

        
        
    def get_pos_sample(self, selected_class, seg_length = 10):
        # print('getting pos sample', self.seg_meta[selected_class])
        # print('getting pos sample')
        class_meta = self.seg_meta[selected_class]
        if not class_meta['time_spane'] or not class_meta['duration']:
            # Handle the case when either list is empty
            # You may want to return a default value or raise an exception
            print('empty', selected_class)
        else:
            sample = random.choices(class_meta['time_spane'], weights=class_meta['duration'], k=1)

        
        file = sample[0]['file']
        start = sample[0]['start']
        end = sample[0]['end']
        start = time2frame(start, self.fps)
        end = time2frame(end, self.fps)
        
        return self.select_sample(file, start, end, seg_length)

    def get_neg_sample(self, selected_class, seg_length = 10):
        # 要不要保证最短negative sample的长度
        # 要不要把negative sample的边界放宽一点 +0.025s
        # 要不要归一化 计算均值和方差
        # print('getting neg sample')
        class_meta = self.seg_meta[selected_class]
        sample = random.choices(class_meta['time_spane'], weights=class_meta['duration'], k=1)
        file = sample[0]['file']
        start = sample[0]['start']
        end = sample[0]['end']
        start = time2frame(start, self.fps)
        end = time2frame(end, self.fps)
        return self.select_sample(file, start, end, seg_length)
    
    
    def select_sample(self, file, start, end, seg_length = 10):
        feature = self.feature_per_file[file][self.feature_list[0]]
        duration = end - start
        if  duration < seg_length:
            start = end - seg_length
            feature = np.tile(feature, (1,np.ceil(seg_length/duration).astype(int)))
            res = feature[:, start:end]
        else:
            start = random.randint(start, end - seg_length)
            res = feature[:, start:start+seg_length]
        return res
    
    def _class2index(self):
        
        label_to_index = {label: index for index, label in enumerate(self.classes)}
        if self.neg_prototype:
            temp = {}
            for label in label_to_index.keys():
                label_to_index[label] = label_to_index[label] * 2
                temp[label+'_neg'] = label_to_index[label] + 1
            label_to_index.update(temp)
        return label_to_index
            
    
    def _index2class(self):
        index_to_label = {self.class2index[label]:label  for label in (self.class2index).keys()}
        return index_to_label
        
        
        
        
        
        

if __name__ == "__main__":
    if not GlobalHydra().is_initialized():
        initialize(config_path="../../")
    # Compose the configuration
    cfg = compose(config_name="config.yaml")
    train_dataset =  TrainDataset(cfg)

    dataloader = DataLoader(train_dataset, batch_sampler = BatchSampler(cfg, train_dataset.classes, len(train_dataset)))
    for batch in dataloader:
        print(batch[0])

