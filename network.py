# import dependencies and libries
import setting
import load_dataset
import numpy as np
import torch, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glove_embedding

import word_embedding

torch.manual_seed(1)
random.seed(1)


# define a feed-forward network
class FFNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes, IsGloVe=False, embedding_matrix=None):
        super(FFNetwork, self).__init__()
        if IsGloVe:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False,
                                                          padding_idx=len(glove_embedding.glove_embedding)+1)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=word_embedding.vocab_size)
        self.fc1 = nn.Linear(embedding_size, num_classes)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)


    def forward(self, text):
        # text is the shape of [batch_size,sequence length]
        embedded = self.embedding(text)

        # mask out padding index
        mask = (text != self.embedding.padding_idx) & (text != 0) # especially 0 is unknown word
        # calculate mean average
        sum_embedded = torch.sum(embedded * mask.unsqueeze(-1).float(), dim=1)
        avg_embedded = sum_embedded / mask.sum(dim=1, keepdim=True).float()

        # proceed sentence
        x = self.fc1(avg_embedded)
        # x = nn.functional.relu(x)
        # x = self.fc2(x)
        # output using softmax (multiclass classification)
        x = nn.functional.softmax(x, dim=1)
        return x


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, _ = self.lstm(embedded)
        predictions = self.fc(self.dropout(output[:, -1, :]))
        final = self.softmax(predictions)
        return final
