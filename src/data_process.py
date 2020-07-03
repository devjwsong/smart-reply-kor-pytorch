from kobert_transformers import get_distilkobert_model
from kobert_transformers import get_tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import torch


def processing(lines, tokenizer, max_len, pad_id):
    # Read dataset, tokenize each sentences, and extract classes in it.
    not_padded_x = []
    x = []
    y = []
    lens = []
    class_dict = {}

    # Process raw data
    for i, line in enumerate(tqdm(lines)):
        arr = line.strip().split('\t')

        # Since each word in labels is separated in _, we should split them first.
        class_words = [w for w in arr[0].split('_')]
        class_name = ' '.join(class_words)

        # DistilKoBERT uses KoBert Tokenizer.
        x_arr = tokenizer.tokenize('[CLS] ' + arr[1] + ' [SEP]')
        x_arr = tokenizer.convert_tokens_to_ids(x_arr)

        x_len = len(x_arr)

        if x_len <= 1:
            continue

        not_padded_x.append(x_arr)

        # Note class dictionary.
        if class_name not in class_dict:
            class_dict[class_name] = len(class_dict)
        y.append(class_dict[class_name])
        lens.append(x_len)

    # Add paddings
    for i, text in enumerate(not_padded_x):
        if max_len < lens[i]:
            x.append(not_padded_x[i][0:max_len])
            lens[i] = max_len
        else:
            temp = not_padded_x[i] + [pad_id] * (max_len - lens[i])
            x.append(temp)

    return torch.LongTensor(x), torch.LongTensor(y), class_dict


def encode_label(data_y):
    # Make one-hot encoded label vectors for calculating margin losses.

    sample_num = data_y.shape[0]
    labels = np.unique(data_y)
    class_num = labels.shape[0]
    labels = range(class_num)

    # Get one-hot-encoded label tensors
    vecs = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        vecs[data_y == labels[i], i] = 1

    return torch.LongTensor(vecs)


class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, pad_id):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Processing {file_path}...")
        self.x, self.y, self.class_dict = processing(lines, tokenizer, max_len, pad_id)
        self.one_hot_label = encode_label(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.one_hot_label[idx]


def read_datasets(data_path):
    # Read datasets and give the data dictionary containing essential data objects for training.

    data = {}
    
    train_data_path = f"{data_path}/train.txt"
    test_data_path = f"{data_path}/test.txt"

    # Load tokenizer and set max length of sentences.
    print("Loading KoBertTokenizer...")
    tokenizer = get_tokenizer()
    bert_config = get_distilkobert_model().config
    max_len = bert_config.max_position_embeddings
    pad_id = tokenizer.token2idx['[PAD]']

    # Preprocess train/test data
    print("Preprocessing train/test data...")
    train_set = CustomDataset(train_data_path, tokenizer, max_len, pad_id)
    test_set = CustomDataset(test_data_path, tokenizer, max_len, pad_id)
    data['train_set'] = train_set
    data['test_set'] = test_set

    # These are train/test custom dataset object.
    data['train_class_dict'] = train_set.class_dict
    data['test_class_dict'] = test_set.class_dict

    data['max_len'] = max_len
    data['pad_id'] = pad_id

    data['vocab_size'] = len(tokenizer.token2idx)
    data['word_emb_size'] = bert_config.dim
    data['embedding'] = None

    return data
