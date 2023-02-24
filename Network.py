# import dependencies and libries
import setting
import load_dataset
import numpy as np
import torch, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import word_embedding

torch.manual_seed(1)
random.seed(1)


# define a feed-forward network
class FFNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes):
        super(FFNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size,padding_idx=1933)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        # text is the shape of [batch_size,sequence length]
        embedded = self.embedding(text)
        # mask out padding index
        mask = text != self.embedding.padding_idx
        # calculate mean average
        sum_embedded = torch.sum(embedded * mask.unsqueeze(-1).float(), dim=1)
        avg_embedded = sum_embedded / mask.sum(dim=1, keepdim=True).float()

        # proceed sentence
        x = self.fc1(avg_embedded)
        x = nn.functional.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # output using softmax (multiclass classification)
        x = nn.functional.softmax(x, dim=0)
        return x

