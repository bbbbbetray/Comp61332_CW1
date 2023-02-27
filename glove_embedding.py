import torch
import load_dataset
import setting
from torch.utils.data import DataLoader, Dataset
import utils

# GloVe does need preprocessed data
train_X = load_dataset.preprocessing(processing_set=load_dataset.train_X,
                                     stopword_path=setting.path_stop, lc=True, is_processed=False)
test_X = load_dataset.preprocessing(processing_set=load_dataset.test_X,
                                    stopword_path=setting.path_stop, lc=True, is_processed=False)
dev_X = load_dataset.preprocessing(processing_set=load_dataset.dev_X,
                                   stopword_path=setting.path_stop, lc=True, is_processed=False)

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


glove_vocab = utils.build_vocab(train_X, 1)

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

# unk vector is at the beginning of the embedding matrix
embedding_matrix = torch.stack(list(glove_embeddings.values()))

# add padding vector to the end of embedding matrix
embedding_matrix = torch.cat((embedding_matrix, padded_vec.unsqueeze(0)), dim=0)


# convert data and tag set into dataloader
def to_dataloader(dataset, padding_criteria, TagSet):
    idx = [[list(glove_embeddings).index(token) if token in glove_embeddings else 0 for token in sent.split()]
           for sent in dataset]
    # Pad or truncate index and convert word embedding into tensor
    padded_idx = utils.pad_or_truncate(idx, padding_criteria, len(glove_embeddings))
    dataset = utils.TextDataset(padded_idx, TagSet)
    data_loader = DataLoader(dataset, batch_size=setting.batch_size, shuffle=True)
    return data_loader


train_loader = to_dataloader(train_X, 25, load_dataset.coarse_train)
dev_loader = to_dataloader(dev_X, 25, load_dataset.coarse_dev)
test_loader = to_dataloader(test_X, 25, load_dataset.coarse_test)

