import torch.optim as optim
import Network
import word_embedding
import load_dataset
import setting
from torch import nn, optim

model = Network.FFNetwork(vocab_size=word_embedding.vocab_size+1, embedding_size=setting.embedding_dim,
                          hidden_size=setting.hidden_size, num_classes=load_dataset.num_coarse)

# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=setting.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(setting.num_epochs):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(word_embedding.train_loader):

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        print(labels)

        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
