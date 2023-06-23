import torch
import numpy as np
from torch.utils.data.sampler import Sampler
# modified from  https://github.com/haoheliu/DCASE_2022_Task_5
class BatchSampler(Sampler):
    def __init__(self, config, classes, data_set_len):
        self.config = config
        self.n_way = config.train.n_way
        self.n_support = config.train.n_support
        self.n_query = config.train.n_query
        self.class_list = list(classes)
        self.n_episode = data_set_len//(self.n_query)
    def __iter__(self):
        for _ in range(self.n_episode):
            # randomly select n_way classes
            selected_classes = np.random.choice(self.class_list, self.n_way, replace=False)
            yield np.tile(selected_classes, self.n_support +self.n_query)
            
    def __len__(self):
        return self.n_episode


class ClassSampler(Sampler):
    def __init__(self, config, classes, data_set_len):
        self.config = config
        self.class_list = list(classes)
        self.len = data_set_len
    def __iter__(self):
        for _ in range(self.len):
            # randomly select n_way classes
            selected_classes = np.random.choice(self.class_list, 1, replace=False)
            yield selected_classes[0]
            
    def __len__(self):
        return self.len

class IntClassSampler(Sampler):
    def __init__(self, config, classes, data_set_len, mode='train'):
        self.config = config
        self.class_list = list(classes)
        # print(self.class_list)
        self.mode = mode
        self.len = data_set_len
    def __iter__(self):
        for _ in range(self.len):
            # randomly select n_way classes
            selected_classes = np.random.choice(self.class_list, 1, replace=False)
            if self.mode == 'train':
                yield int(selected_classes[0])
            else:
                yield 0

            
    def __len__(self):
        return self.len

# modified from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial16/Meta_Learning.html
class TaskBatchSampler(Sampler):
    
    def __init__(self, config, classes, data_set_len):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            batch_size - Number of tasks to aggregate in a batch
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which 
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but 
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
        """
        self.batch_sampler = BatchSampler(config, classes, data_set_len)
        self.task_batch_size = config.train.task_batch_size
        
    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        for batch_idx, batch in enumerate(self.batch_sampler):
            # print('len:', len(batch))
            batch_list.append(batch)
            if (batch_idx+1) % self.task_batch_size == 0:
                # task_batch = tuple(batch_list)
                # task_batch = np.stack(batch_list)
                # print(task_batch.shape)
                yield batch_list
                batch_list = []

    def __len__(self):
        return len(self.batch_sampler)//self.task_batch_size
    
    def get_collate_fn(self):
        # Returns a collate function that converts one big tensor into a list of task-specific tensors
        def collate_fn(item_list):
            task_list = []
            if len(item_list[0][0]) == 2:
                for task in item_list:
                    class_temp_pos = []
                    data_temp_pos = []
                    index_temp_pos = []
                    class_temp_neg = []
                    data_temp_neg = []
                    index_temp_neg = []
                    for i in range(len(task)):
                        pos, neg = task[i]
                        (class_name, data_pos, pos_index) = pos
                        (selected_class_neg, data_neg, neg_index) = neg
                        class_temp_pos.append(class_name)
                        data_temp_pos.append(data_pos)
                        index_temp_pos.append(pos_index)
                        class_temp_neg.append(selected_class_neg)
                        data_temp_neg.append(data_neg)
                        index_temp_neg.append(neg_index)
                    class_temp_pos = np.array(class_temp_pos)
                    data_temp_pos = torch.stack(data_temp_pos)
                    index_temp_pos = np.array(index_temp_pos)
                    class_temp_neg = np.array(class_temp_neg)
                    data_temp_neg = torch.stack(data_temp_neg)
                    index_temp_neg = np.array(index_temp_neg)
                    pos_task = (class_temp_pos, data_temp_pos, index_temp_pos)
                    neg_task = (class_temp_neg, data_temp_neg, index_temp_neg)
                    task_list.append((pos_task, neg_task))
                    
                return task_list
            elif len(item_list[0][0]) == 3:
                for task in item_list:
                    class_temp_pos = []
                    data_temp_pos = []
                    index_temp_pos = []
      
                    for i in range(len(task)):

                        (class_name, data_pos, pos_index) = task[i]

                        class_temp_pos.append(class_name)
                        data_temp_pos.append(data_pos)
                        index_temp_pos.append(pos_index)

                    class_temp_pos = np.array(class_temp_pos)
                    data_temp_pos = torch.stack(data_temp_pos)
                    index_temp_pos = np.array(index_temp_pos)
                    pos_task = (class_temp_pos, data_temp_pos, index_temp_pos)
                    task_list.append(pos_task)
                return task_list
            else:
                raise ValueError('Unsupported number of elements in the item list')

        return collate_fn