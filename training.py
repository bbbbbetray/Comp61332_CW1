import torch.optim as optim

import glove_embedding
import network
import word_embedding
import load_dataset
import setting
from torch import nn, optim
import torch
import random

model = network.FFNetwork(vocab_size=word_embedding.vocab_size + 1, embedding_size=setting.embedding_dim,
                          hidden_size=setting.hidden_size, num_classes=load_dataset.num_fine, IsGloVe=False,
                          embedding_matrix=glove_embedding.embedding_matrix2)

torch.manual_seed(1)
random.seed(1)
# print("---------------------------------------training-----------------------------------------")
# print(word_embedding.vocab_size+1)
# print(setting.embedding_dim)
# print(setting.hidden_size)
# print(load_dataset.num_coarse)

# model = network.BiLSTM(vocab_size=word_embedding.vocab_size + 1, embedding_size=setting.embedding_dim,
#                        hidden_size=setting.hidden_size, num_classes=load_dataset.num_coarse, num_layers=2, dropout=0.5)
# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=setting.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(setting.num_epochs):
    running_loss = 0.0
    correct = 0
    for i, (inputs, labels) in enumerate(word_embedding.train_loader):

        # zero the gradients
        model.zero_grad()

        # forward pass
        outputs = model(inputs)

        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # compute the number of correct predictions
        predicted_labels = torch.argmax(outputs, dim=1)
        correct += (predicted_labels == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    # compute the accuracy on the training data
    accuracy = correct / len(word_embedding.train_loader.dataset)
    # print epoch statistics
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch + 1, setting.num_epochs, running_loss / len(word_embedding.train_loader), accuracy * 100))

# Set the model to evaluation mode
model.eval()

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Create empty lists to store the true labels and predicted labels
true_labels = []
pred_labels = []

# Iterate over the test data and make predictions
with torch.no_grad():
    for inputs, labels in word_embedding.dev_loader:
        # Forward pass
        outputs = model(inputs)
        # Compute predicted labels
        predicted_labels = torch.argmax(outputs, dim=1)
        # Append true and predicted labels to the lists
        true_labels.extend(labels.numpy())
        pred_labels.extend(predicted_labels.numpy())

# Calculate accuracy and other metrics using sklearn
accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted',
                                                                 zero_division=True)

# Print the evaluation metrics
print('Dev Accuracy: {:.2f}%'.format(accuracy * 100))
print('Dev Precision: {:.2f}%'.format(precision * 100))
print('Dev Recall: {:.2f}%'.format(recall * 100))
print('Dev F1 Score: {:.2f}%'.format(f1_score * 100))
#

# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#
# # Create empty lists to store the true labels and predicted labels
# true_labels = []
# pred_labels = []
#
# # Iterate over the test data and make predictions
# with torch.no_grad():
#     for inputs, labels in word_embedding.test_loader:
#         # Forward pass
#         outputs = model(inputs)
#         # Compute predicted labels
#         predicted_labels = torch.argmax(outputs, dim=1)
#         # Append true and predicted labels to the lists
#         true_labels.extend(labels.numpy())
#         pred_labels.extend(predicted_labels.numpy())
#
# # Calculate accuracy and other metrics using sklearn
# accuracy = accuracy_score(true_labels, pred_labels)
# precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted',
#                                                                  zero_division=True)
#
# # Print the evaluation metrics
# print('Test Accuracy: {:.2f}%'.format(accuracy * 100))
# print('Test Precision: {:.2f}%'.format(precision * 100))
# print('Test Recall: {:.2f}%'.format(recall * 100))
# print('Test F1 Score: {:.2f}%'.format(f1_score * 100))
