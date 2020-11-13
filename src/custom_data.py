from kobert_transformers import get_distilkobert_model
from kobert_transformers import get_tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, config):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.x = []
        self.y = []
        self.class_dict = {}
        
        print(f"Processing {file_path}...")
        for i, line in enumerate(tqdm(lines)):
            comps = line.strip().split('\t')
            y = int(comps[0])
            class_name = comps[1]
            x = comps[2]
            
            if class_name not in self.class_dict:
                self.class_dict[class_name] = y
                
            tokens = tokenizer.tokenize(x)
            tokens = tokenizer.convert_tokens_to_ids(tokens)
            tokens  = [config['cls_id']] + tokens + [config['sep_id']]
            
            if len(tokens) <= config['max_len']:
                tokens += [config['pad_id']] * (config['max_len'] - len(tokens))
            else:
                tokens = tokens[:config['max_len']]
                tokens[-1] = config['sep_id']
                
            self.x.append(tokens)
            self.y.append(y)
        
        self.x = torch.LongTensor(self.x)
        self.y = torch.LongTensor(self.y)
        self.one_hot_label = self.encode_label()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.one_hot_label[idx]
    
    def encode_label(self):
        # Make one-hot encoded label vectors for calculating margin losses.
        sample_num = self.y.shape[0]
        labels = np.unique(self.y)
        class_num = labels.shape[0]
        labels = range(class_num)

        # Get one-hot-encoded label tensors
        vecs = np.zeros((sample_num, class_num), dtype=np.float32)
        for i in range(class_num):
            vecs[self.y == labels[i], i] = 1

        return torch.LongTensor(vecs)
