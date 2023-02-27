import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter

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

# pad or truncate?
# truncate sequence if it exceeds max_len
# else pad it
def pad_or_truncate(sequences, max_len, pad_value):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return torch.LongTensor(padded_sequences)


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
