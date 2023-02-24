import load_dataset
import setting
from collections import Counter
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(1)

# Build up vocabulary
def build_vocab(train_X, min_freq):
    # Split the sentences into tokens
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


# get vocab size
vocab_size = len(vocab)

# print(vocab_size)
# convert each sentence into every token index in the vocabulary, those not appear using "#unk"(index 0) instead
idx_train_X = [[vocab[token] if token in vocab else 0 for token in sent.split()] for sent in load_dataset.train_X]


# the sentence corresponding labels
labels_train_X = [load_dataset.lab_dict1.get(load_dataset.train_coarse[i]) for i in
                  range(len(load_dataset.train_coarse))]

# Pad index and convert word embedding into tensor
padded_idx_train_X = nn.utils.rnn.pad_sequence([torch.tensor(indexed_sentence) for indexed_sentence in idx_train_X],
                                               batch_first=True, padding_value=vocab_size)

dataset = TextDataset(padded_idx_train_X, labels_train_X)

train_loader = DataLoader(dataset, batch_size=setting.batch_size, shuffle=True)

sent = [sent for sent,_ in train_loader]
lab = [lab for _,lab in train_loader]
print(sent[1])