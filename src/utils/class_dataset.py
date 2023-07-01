# This code is modified from https://github.com/haoheliu/DCASE_2022_Task_5
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
from src.utils.helpers import *
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

class ClassDataset(Dataset):
    def __init__(self, config, mode,same_class_in_different_file, debug):
        self.config = config
        self.feature_list = config.features.feature_list.split("&")
        if mode == 'train' or mode == 'anchor':
            self.data_dir = normalize_path(config.path.train_dir)
        elif mode == 'val':
            self.data_dir = normalize_path(config.path.val_dir)
        elif mode == 'test':
            self.data_dir = normalize_path(config.path.test_dir)
        elif mode == 'pretrain':
            self.data_dir = normalize_path(config.path.pretrain_dir)
        else:
            raise ValueError('Unknown mode')
        self.mode = mode
        self.debug = debug
        self.feature_per_file = {}
        self.classes = set()
        self.seg_meta = {}
        self.sr = config.features.sr
        self.fps = self.sr / config.features.hop_length
        
        self.neg_prototype = self.config.train.neg_prototype
        self.mean = 0.50484073
        self.std =  0.36914015
        
        

        self.collect_features()
        self.process_labels(same_class_in_different_file)
        # sum = 0
        # for key in self.seg_meta.keys():
        #     print('key:', key, 'mean:', np.sum(self.seg_meta[key]['duration']))
        #     sum += np.sum(self.seg_meta[key]['duration'])
        # print('sum', sum/60)
        
        self.class2index = self._class2index()
        self.index2class = self._index2class()
        if self.debug:
            self.length = int(0.03 * 3600 / (self.config.features.segment_len_frame * (1/self.fps)))
        else:
            # self.length = int(3* 3600 / (self.config.features.segment_len_frame * (1/self.fps)))
            self.length = 500
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.recent_return_sample = {}
        for key in self.seg_meta.keys():
            self.recent_return_sample[key] = set()

    
    
    def __getitem__(self, class_name):
        if isinstance(class_name, np.ndarray):
            task_batch = []
            for name in class_name:
                if len(self.recent_return_sample[name]) == self.config.train.n_support + self.config.train.n_query:
                    self.recent_return_sample[name] = set()
                while True:
                    sample = self.get_task(name)
                    # print('sample', sample)
                    if sample not in self.recent_return_sample[name]:
                        self.recent_return_sample[name].add(sample)
                        break
                sample = self.get_task(name)
                task_batch.append(sample)
            return task_batch
        elif isinstance(class_name, str) and (self.mode != 'pretrain' and self.mode != 'anchor'):
            if len(self.recent_return_sample[class_name]) == self.config.train.n_support + self.config.train.n_query:
                    self.recent_return_sample[class_name] = set()
            while True:
                    sample = self.get_task(class_name)
                    if sample not in self.recent_return_sample[class_name]:
                        self.recent_return_sample[class_name].add(sample)
                        break
            sample = self.get_task(class_name)
            return [sample]
        elif self.mode == 'pretrain' or self.mode == 'anchor':
            task =self.get_task(class_name)
            
            return task[1], torch.tensor(task[2]).to(self.device)
        else:
            raise ValueError('Unknown type')
    
    
    def get_task(self, class_name):
        start_time = time.time()
        selected_class_neg = class_name + '_neg'
        pos = self.get_pos_sample(class_name, self.config.features.segment_len_frame)
        # pos = torch.from_numpy(pos).to(self.device)
        if self.neg_prototype:
            neg = self.get_neg_sample(selected_class_neg, self.config.features.segment_len_frame)
        # neg = torch.from_numpy(neg).to(self.device)

        pos_index = self.class2index[class_name] # int
        if self.neg_prototype:
            neg_index = self.class2index[selected_class_neg] # int
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"代码执行时间为 {elapsed_time:.2f} 秒")
        if self.neg_prototype:
            return (class_name, pos, pos_index), (selected_class_neg, neg, neg_index)
        
        else:
            return(class_name, pos, pos_index)
    
    def __len__(self):
        return(self.length )
    def collect_features(self):
        print("Collecting training set features...")
        array_list = []
        for file in tqdm(walk_files(self.data_dir,debug= self.debug, file_extension = ('.wav'))):
            for feature in self.feature_list:
                self.feature_per_file[file] = self.feature_per_file.get(file, {})
                self.feature_per_file[file]['duration'] = librosa.get_duration(filename = file)
                save_path = audio2feature(file, feature)
                self.feature_per_file[file][feature] = np.load(save_path)
                print('file', file, 'feature', feature, 'shape', self.feature_per_file[file][feature].shape)
                array_list.append(self.feature_per_file[file][feature])
        array_list = np.concatenate(array_list, axis = 1)
        self.mean = np.mean(array_list)
        self.std = np.std(array_list)
        print('mean', self.mean)
        print('std', self.std)
        
    def process_labels(self, same_label):
        print("Processing labels...")
        for file in tqdm(walk_files(self.data_dir,debug= self.debug, file_extension = ('.csv'))):

            df = pd.read_csv(file)
            df = df.sort_values(by='Starttime', ascending=True)
            file = file.replace('.csv','.wav')

            # class name starts from the 4th column
            for column in df.columns[3:]:
                pos_idx = df[df[column] == 'POS'].index.tolist()
                if not same_label:
                    column = column + '&'+ file
                self.classes.add(column)
                start = df['Starttime'][pos_idx].tolist()
                end = df['Endtime'][pos_idx].tolist()
                # print('start', start)
                # print('end', end)
                self.seg_meta[column] = self.seg_meta.get(column, {})
                self.seg_meta[column]['time_spane'] = self.seg_meta[column].get('time_spane', [])
                self.seg_meta[column]['duration'] = self.seg_meta[column].get('duration', [])
                meta_temp = {'time_spane': [], 'duration': []}
                for s, e in zip(start, end):
                    meta_temp['time_spane'].append({'start': s, 'end': e, 'file': file})
                    meta_temp['duration'].append(e - s)
                # print('file', file, 'column', column)
                # print('before merge', len(meta_temp['time_spane']))
                meta_temp = merge_intervals(meta_temp)
                # print('after merge', len(meta_temp['time_spane']))

                self.seg_meta[column]['time_spane'] += meta_temp['time_spane']
                self.seg_meta[column]['duration'] += meta_temp['duration']
                
                if self.seg_meta[column]['time_spane'] == []:
                    self.classes.remove(column)
                    self.seg_meta.pop(column)
                    continue

                
                if self.neg_prototype:
                    start = [meta['start'] for meta in meta_temp['time_spane']]
                    start = start + [self.feature_per_file[file]['duration']]
                    end = [0.0] + [meta['end'] for meta in meta_temp['time_spane']]
                    column_neg = column + '_neg'
                    self.seg_meta[column_neg] = self.seg_meta.get(column_neg, {})
                    self.seg_meta[column_neg]['time_spane'] = self.seg_meta[column_neg].get('time_spane', [])
                    self.seg_meta[column_neg]['duration'] = self.seg_meta[column_neg].get('duration', [])                    
                    
                    for i in range(len(start)):
                        if start[i] > end[i]:
                            self.seg_meta[column_neg]['time_spane'].append({'start': end[i], 'end': start[i], 'file': file})
                            self.seg_meta[column_neg]['duration'].append(start[i] - end[i])
                        else:
                            raise Exception('start time is smaller than end time')

                    
        
    def get_pos_sample(self, selected_class, seg_length = 10):
        # print('getting pos sample', self.seg_meta[selected_class])
        # print('getting pos sample')

        class_meta = self.seg_meta[selected_class]
        if not class_meta['time_spane'] or not class_meta['duration']:
            # Handle the case when either list is empty
            # You may want to return a default value or raise an exception
            raise Exception('empty', selected_class)
        else:
            sample = random.choices(class_meta['time_spane'], weights=class_meta['duration'], k=1)

        
        file = sample[0]['file']
        start = sample[0]['start']
        end = sample[0]['end']
        start = time2frame(start, self.fps)
        end = time2frame(end, self.fps)

        res = self.select_sample(file, start, end, seg_length)
        
        
        return res

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
        length = self.feature_per_file[file]['duration']
        # print('my fps', self.fps)
        # print('real fps', feature.shape[1]/length)
        if end > feature.shape[1]:
            raise ValueError('end is larger than feature length')
        
        duration = end - start
        # print('file', file, 'start', start, 'end', end)
        if duration == 0:
            feature = np.tile(feature[:,start:end+1], (1,np.ceil(seg_length).astype(int)))
            print('file', file, 'start', start, 'end', end)
            raise ValueError('duration is 0')
            res.append(feature[:, : seg_length])
            
        
        elif  0 < duration and duration < seg_length:
            start_time = time.time()
            feature = np.tile(feature[:, start:end], (1,np.ceil(seg_length/duration).astype(int)))
            res = feature[:, :seg_length]
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            # print(f"代码执行时间为 {elapsed_time:.2f} 秒")
        else:

            # print('normal')
            start = random.randint(start, end - seg_length)
            # print('start', start)
            # print('end', end)
            # print('shape', feature.shape)
            res = feature[:, start:start+seg_length]
        # print('mean',self.mean)
        # print('std', self.std)  
        res = (res - self.mean) / self.std
        res = torch.from_numpy(res).to(self.device)
        
        
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
    train_dataset =  ClassDataset(cfg)
    meta = train_dataset.seg_meta
    
    for key in meta.keys():
        if 'neg' not in key:
            print('key:', key, 'mean:', np.mean(meta[key]['duration']), 'std:', np.std(meta[key]['duration']))
    
    # dataloader = DataLoader(train_dataset, batch_sampler = BatchSampler(cfg, train_dataset.classes, len(train_dataset)))
    # for batch in dataloader:
    #     print(batch[0])

