
import network
import word_embedding
import load_dataset
import network
import setting
from torch import nn
import torch
import glove_embedding

# sent = [sent for sent,_ in word_embedding.train_loader]
# lab = [lab for _,lab in word_embedding.train_loader]

# print(sent[0])
# print(lab[0])
#
# vocab_size = word_embedding.vocab_size
# embedding_size = setting.embedding_dim
# hidden_size = setting.hidden_size
# # dropout = 0.5
# num_classes = 6
# num_layers= 2
#
# embedding = nn.Embedding(vocab_size+1, embedding_size)
# lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=num_layers, dropout=dropout)
# fc = nn.Linear(hidden_size * 2, num_classes)
# dropout = nn.Dropout(dropout)
# softmax = nn.Softmax(dim=1)
#
# embedded = dropout(embedding(sent[0]))
# output, _ = lstm(embedded)
# predictions = fc(dropout(output[:,-1, :]))
# final = softmax(predictions)
#
# print(final)


# sent = [sent for sent,_ in glove_embedding.train_loader]
# lab = [lab for _,lab in glove_embedding.train_loader]
# embedding_matrix = glove_embedding.embedding_matrix
#
# embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False,
#                                                           padding_idx=len(glove_embedding.glove_embeddings))
# fc1 = nn.Linear(embedding_size, hidden_size)
# fc2 = nn.Linear(hidden_size, num_classes)
#
# embedded = embedding(sent[0])
#
# # mask out padding index
# mask = sent[0] != embedding.padding_idx
# # calculate mean average
# sum_embedded = torch.sum(embedded * mask.unsqueeze(-1).float(), dim=1)
# avg_embedded = sum_embedded / mask.sum(dim=1, keepdim=True).float()
#
# # proceed sentence
# x = fc1(avg_embedded)
# x = nn.functional.relu(x)
# # x = self.dropout(x)
# x = fc2(x)
# # output using softmax (multiclass classification)
# x = nn.functional.softmax(x, dim=1)



y= {'hello':0,'world':1}
print(list(y))


