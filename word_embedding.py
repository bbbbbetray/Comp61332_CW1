import load_dataset
import setting
from collections import Counter
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

torch.manual_seed(1)
random.seed(1)


# Build up vocabulary
def build_vocab(train_X, min_freq):
    # tokenize
    tokens = [token for sentence in train_X for token in sentence.split()]

    # Filter out the words appear at least min_freq times, including an "unknown" token
    vocab = {'<unk>': 0}
    vocab.update(dict(filter(lambda x: x[1] >= min_freq, Counter(tokens).items())))

    # Reset the index of the vocab
    idx = 0
    for key in vocab.keys():
        vocab[key] = idx
        idx += 1
    return vocab


class TextDataset(Dataset):
    def __init__(self, text, labels):
        self.texts = text
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


# build up my vocabulary
vocab = build_vocab(train_X=load_dataset.train_X, min_freq=3)
# print('--------------------------------word embedding------------------------------------')
# print(vocab)

# get vocab size
vocab_size = len(vocab)

# print(vocab_size)
# convert each sentence into every token index in the vocabulary, those not appear using "#unk"(index 0) instead
idx_train_X = [[vocab.get(token) if token in vocab else 0 for token in sent.split()] for sent in load_dataset.train_X]
# print(idx_train_X[:2])

# the sentence corresponding labels
labels_train_X = [load_dataset.lab_dict1.get(load_dataset.train_coarse[i]) for i in
                  range(len(load_dataset.train_coarse))]
# print(labels_train_X[:2])


# pad or truncate?
def pad_or_truncate(sequences, max_len, pad_value):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return torch.LongTensor(padded_sequences)


padded_idx_train_X = pad_or_truncate(idx_train_X, 30, vocab_size)

dataset = TextDataset(padded_idx_train_X, labels_train_X)

train_loader = DataLoader(dataset, batch_size=setting.batch_size, shuffle=True)

####### test loader
idx_test_X = [[vocab.get(token) if token in vocab else 0 for token in sent.split()] for sent in load_dataset.test_X]

padded_idx_test_X = pad_or_truncate(idx_test_X, 30, vocab_size)

labels_test_X = [load_dataset.lab_dict1.get(load_dataset.test_coarse[i]) for i in
                 range(len(load_dataset.test_coarse))]

# print(labels_test_X)

dataset_test = TextDataset(padded_idx_test_X, labels_test_X)
#
test_loader = DataLoader(dataset_test, batch_size=setting.batch_size, shuffle=True)

### eva loader

idx_eval_X = [[vocab.get(token) if token in vocab else 0 for token in sent.split()] for sent in load_dataset.dev_X]

# print(idx_eval_X[:2])

padded_idx_dev_X = pad_or_truncate(idx_eval_X, 30, vocab_size)

labels_dev_X = [load_dataset.lab_dict1.get(load_dataset.dev_coarse[i]) for i in
                range(len(load_dataset.dev_coarse))]

# print("----------------------here---------------------")
# print(padded_idx_dev_X[:20])
# print(labels_dev_X[:20])
dataset_dev = TextDataset(padded_idx_dev_X, labels_dev_X)
#
dev_loader = DataLoader(dataset_dev, batch_size=setting.batch_size, shuffle=True)
