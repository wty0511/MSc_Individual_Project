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
from src.utils.dataset_train import *



class ValDataset(Dataset):
    def __init__(self, config, val=True):
        self.is_val = val
        self.config = config
        self.feature_list = config.features.feature_list.split("&")
        self.val_dir = normalize_path(config.path.val_dir) if val else normalize_path(config.path.test_dir)
        # for each csv file, we have a list of features
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
        self.classes = list(self.classes)
        self.length = len(self.classes)
        self.seg_len_base = config.val.seg_len_base
        self.hop_len_frac = config.val.hop_len_frac
        self.seg_len = 0
        self.seg_hop = 0
    

    def __getitem__(self, idx):
        class_name = self.classes[idx]
        selected_class_neg = class_name + '_neg'
        pos = self.get_pos_sample(class_name)
        neg = self.get_neg_sample(selected_class_neg)
        pos_index = self.class2index[class_name]
        neg_index = self.class2index[selected_class_neg]
        query_start = self.seg_meta[class_name]['time_spane'][-1]['end']
        query = self.get_query_sample(class_name,query_start)
        if self.neg_prototype:
            return (class_name, pos, pos_index), (selected_class_neg, neg, neg_index), query, self.seg_len, self.seg_hop, query_start
        
        else:
            return(class_name, pos, pos_index), None, query, self.seg_len, self.seg_hop, query_start
        
    def __len__(self):
        return self.length
    
    def collect_features(self):
        if self.is_val:
            print("Collecting val set features...")
        else:
            print("Collecting test set features...")
        for file in tqdm(walk_files(self.val_dir, file_extension = ('.wav'))):
            for feature in self.feature_list:
                save_path = audio2feature(file, feature)
                self.feature_per_file[file] = self.feature_per_file.get(file, {})
                self.feature_per_file[file][feature] = np.load(save_path)
                
    
    def process_labels(self):
        if self.is_val:
            print("Processing val labels...")
        else:
            print("Processing test labels...")
        for file in tqdm(walk_files(self.val_dir, file_extension = ('.csv'))):
            df = pd.read_csv(file)
            file = file.replace('.csv','.wav')
            # class name starts from the 4th column(only one class in this case)
            for column in df.columns[3:]:
                # class in each file is not clear, so we need to add the file name to the class name
                pos_idx = df[df[column] == 'POS'].index.tolist()
                column = column+'&'+file
                self.classes.add(column)
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
                                 
                    for i in range(self.config.train.n_support):
                        if start[i] > end[i]:
                            self.seg_meta[column_neg]['time_spane'].append({'start': end[i], 'end': start[i], 'file': file})
                            self.seg_meta[column_neg]['duration'].append(start[i] - end[i])

        
    
    def get_pos_sample(self, selected_class):
        # print('getting pos sample', self.seg_meta[selected_class])
        # print('getting pos sample')
        class_meta = self.seg_meta[selected_class]
        print(class_meta['time_spane'])
        print(class_meta['duration'])
        
        if not class_meta['time_spane'] or not class_meta['duration']:
            # Handle the case when either list is empty
            # You may want to return a default value or raise an exception
            raise Exception('empty', selected_class)
        else:
            duration_frame = time2frame(class_meta['duration'], self.fps)
            self.seg_len, self.seg_hop = self.adaptive_length_hop(duration_frame)
            all_pos_seg = []
            for i in class_meta['time_spane'][:self.config.train.n_support]:
                f = i['file']
                s = i['start']
                e = i['end']
                s = time2frame(s, self.fps)
                e = time2frame(e, self.fps)
                # print('s', s)
                # print('e', e)
                pos_seg = self.select_sample(f, s, e, self.seg_len, self.seg_hop)
                # print('pos_seg', pos_seg)
                all_pos_seg.extend(pos_seg)

            return np.stack(all_pos_seg)

    def get_neg_sample(self, selected_class):
        # print('getting pos sample', self.seg_meta[selected_class])
        # print('getting pos sample')
        class_meta = self.seg_meta[selected_class]
        if not class_meta['time_spane'] or not class_meta['duration']:
            # Handle the case when either list is empty
            # You may want to return a default value or raise an exception
            raise Exception('empty', selected_class)
        else:
            duration_frame = time2frame(class_meta['duration'], self.fps)
            all_neg_seg = []
            for i in class_meta['time_spane']:
                f = i['file']
                s = i['start']
                e = i['end']
                s = time2frame(s, self.fps)
                e = time2frame(e, self.fps)
                pos_seg = self.select_sample(f, s, e, self.seg_len, self.seg_hop)
                all_neg_seg.extend(pos_seg)
            all_neg_seg =  np.stack(all_neg_seg)
            if all_neg_seg.shape[0] > self.config.val.test_loop_neg_sample:
                neg_indices = torch.randperm(all_neg_seg.shape[0])[: self.config.val.test_loop_neg_sample]
                all_neg_seg = all_neg_seg[neg_indices]
            
            return all_neg_seg
    
    
    def select_sample(self, file, start, end, seg_length, seg_hop):
        feature = self.feature_per_file[file][self.feature_list[0]]
        
        res = []
        duration = end - start
        # print('duration', duration)
        # print('seg_length', seg_length)
        
        if duration == 0:
            feature = np.tile(feature[:,start:end+1], (1,np.ceil(seg_length).astype(int)))
            res.append(feature[:, : seg_length])
            print('file', file, 'start', start, 'end', end)
            raise ValueError('duration is 0')
        elif  0 < duration and duration< seg_length:
            # print('feature', feature.shape)
            feature = np.tile(feature[:,start:end], (1,np.ceil(seg_length/duration).astype(int)))
            res.append(feature[:, : seg_length])

        else:
            for i in range(start, end - seg_length + 1, seg_hop):
                segment = feature[:, i: i + seg_length]
                res.append(segment)
            
            if (end -start - seg_length) % seg_hop != 0:
                last_segment = feature[:, -seg_length:]
                res.append(last_segment)

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
        
    def adaptive_length_hop(self, duration_list):
        # from task5 2022
        #################################Adaptive hop_seg#########################################

        # Adaptive segment length based on the audio file.
        max_len = max(duration_list)
        # print('max_len', max_len)
        # Choosing the segment length based on the maximum size in the 5-shot.
        # Logic was based on fitment on 12GB GPU since some segments are quite long.
        if max_len < 8:
            seg_len = 8
        elif max_len < self.seg_len_base:
            seg_len = max_len
        elif (
            max_len >= self.seg_len_base
            and max_len <= self.seg_len_base * 2
        ):
            seg_len = max_len // 2
        elif max_len > self.seg_len_base * 2 and max_len < 500:
            seg_len = max_len // 4
        else:
            seg_len = max_len // 8
        # print(f"Adaptive segment length for %s is {seg_len}" % (file))
        hop_seg = seg_len // self.hop_len_frac
        return seg_len, hop_seg
        #################################################################################
        
    def get_query_sample(self,class_name, query_start):
        class_meta = self.seg_meta[class_name]
        file =class_meta['time_spane'][0]['file']
        # self.seg_len, self.seg_hop = self.adaptive_length_hop(class_meta['duration'])
        feature = self.feature_per_file[file][self.feature_list[0]]
        res = []
        end = feature.shape[1]
        query_start = time2frame(query_start, self.fps)
        # print('query_start', query_start)
        # print('end', end)
        # print('duration', end - query_start)

        duration = end - query_start
        if  duration< self.seg_len:
            feature = np.tile(feature[:, query_start:end], (1,np.ceil(self.seg_len/duration).astype(int)))
            res.append(feature[:, : self.seg_len])
        else:
            for i in range(query_start, end - self.seg_len + 1, self.seg_hop):
                segment = feature[:, i: i + self.seg_len]
                res.append(segment)
            # 最后一个怎么处理？
            if (end -query_start - self.seg_len) % self.seg_hop != 0:
                last_segment = feature[:, -self.seg_len:]
                res.append(last_segment)
        
        res = np.stack(res)
        return res
        
        
        

if __name__ == "__main__":
    if not GlobalHydra().is_initialized():
        initialize(config_path="../../")
    # Compose the configuration
    cfg = compose(config_name="config.yaml")
    train_dataset =  TrainDataset(cfg)

    dataloader = DataLoader(train_dataset, batch_sampler = BatchSampler(cfg, train_dataset.classes, len(train_dataset)))
    for batch in dataloader:
        print(batch[0])
        
