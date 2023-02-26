import torch
import word_embedding
import load_dataset
import setting
from torch import nn
from torch.utils.data import DataLoader, Dataset


### unknown word may mislead dictionary
def load_glove(path, my_vocab):
    # load GloVe, path need to be altered when have configuration file
    # only loads those words in the vocabulary pruning
    glove_embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in my_vocab:
                vectors = torch.tensor([float(x) for x in values[1:]], dtype=torch.float32)
                glove_embeddings[word] = vectors
        f.close()
    return glove_embeddings


glove_vocab = word_embedding.build_vocab(load_dataset.train_X, 1)

glove_embedding = load_glove(setting.path_glove, glove_vocab)

# Define the "unknown" token
unk_token = "<unk>"

# Generate a random tensor of shape (300,)
unk_vec = torch.rand(300)

# Scale and shift the values to the range of -1 to 1
unk_vec = unk_vec * 2 - 1

# Generate a random tensor of shape (300,)
padded_vec = torch.rand(300)

# Scale and shift the values to the range of -1 to 1
padded_vec = padded_vec * 2 - 1

# Add unk tensor inside the dictionary
glove_embeddings = {unk_token: unk_vec, **glove_embedding}

# unk vector is at the begining of the embedding matrix
embedding_matrix1 = torch.stack(list(glove_embeddings.values()))

# add padding vectors to the end of embedding matrix
embedding_matrix2 = torch.cat((embedding_matrix1, padded_vec.unsqueeze(0)), dim=0)

# --------------------training set

# convert each sentence into every token index in the vocabulary, those not appear using "#unk"(index 0) instead
train_idx_X = [[list(glove_embeddings).index(token) if token in glove_embeddings else 0 for token in sent.split()] for
               sent in
               load_dataset.train_X]

# Pad or truncate index and convert word embedding into tensor
train_padded_idx_X = word_embedding.pad_or_truncate(train_idx_X, 30, len(glove_embeddings))

labels_train_X = [load_dataset.lab_dict1.get(load_dataset.train_coarse[i]) for i in
                  range(len(load_dataset.train_coarse))]

dataset_train = word_embedding.TextDataset(train_padded_idx_X, labels_train_X)

train_loader = DataLoader(dataset_train, batch_size=setting.batch_size, shuffle=True)

# ------------------------dev set

# convert each sentence into every token index in the vocabulary, those not appear using "#unk"(index 0) instead
dev_idx_X = [[list(glove_embeddings).index(token) if token in glove_embeddings else 0 for token in sent.split()] for
             sent in
             load_dataset.dev_X]

# Pad or truncate index and convert word embedding into tensor
dev_padded_idx_X = word_embedding.pad_or_truncate(dev_idx_X, 30, len(glove_embeddings))

labels_dev_X = [load_dataset.lab_dict1.get(load_dataset.dev_coarse[i]) for i in
                range(len(load_dataset.dev_coarse))]

# print(dev_padded_idx_X[:5])
# print(labels_dev_X[:5])


dataset_dev = word_embedding.TextDataset(dev_padded_idx_X, labels_dev_X)

dev_loader = DataLoader(dataset_dev, batch_size=setting.batch_size, shuffle=True)


