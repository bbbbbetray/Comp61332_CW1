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

embedding_matrix = torch.stack(list(glove_embeddings.values()))

embedding_matrix = torch.cat((embedding_matrix, padded_vec.unsqueeze(0)), dim=0)



# convert each sentence into every token index in the vocabulary, those not appear using "#unk"(index 0) instead
idx_X = [[list(glove_embeddings).index(token) if token in glove_embeddings else 0 for token in sent.split()] for sent in
         load_dataset.train_X]

# Pad index and convert word embedding into tensor
padded_sentences_tensor = nn.utils.rnn.pad_sequence([torch.tensor(indexed_sentence) for indexed_sentence in idx_X],
                                                    batch_first=True, padding_value=len(glove_embeddings))

labels_train_X = [load_dataset.lab_dict1.get(load_dataset.train_coarse[i]) for i in
                  range(len(load_dataset.train_coarse))]

dataset = word_embedding.TextDataset(padded_sentences_tensor, labels_train_X)

train_loader = DataLoader(dataset, batch_size=setting.batch_size, shuffle=True)

# sent = [sent for sent, _ in train_loader]
# lab = [lab for _, lab in train_loader]
# print(sent[0])
# print(lab[0])
