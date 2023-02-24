
import Network
import word_embedding
import load_dataset
import Network
import setting
from torch import nn
import torch

sent = [sent for sent,_ in word_embedding.train_loader]
print(sent[0])

vocab_size = word_embedding.vocab_size
print(vocab_size)
embedding = nn.Embedding(num_embeddings=vocab_size+1,embedding_dim=100,padding_idx=1933)
embedded = embedding(sent[0])

mask = sent[0] != embedding.padding_idx
# calculate mean average
sum_embedded = torch.sum(embedded * mask.unsqueeze(-1).float(), dim=1)
avg_embedded = sum_embedded / mask.sum(dim=1, keepdim=True).float()

print(avg_embedded[0])
# embedding = nn.Embedding(word_embedding.vocab_size +1 , setting.embedding_dim)
#
#
#
# mask = (text != setting.pad_idx).float()
# BoWLayer = nn.Linear(,1)
# num_nonzero = mask.sum(dim=1).unsqueeze(1)
# bow_output = BoWLayer((embedding * mask.unsqueeze(2)).sum(dim=1) / num_nonzero)

