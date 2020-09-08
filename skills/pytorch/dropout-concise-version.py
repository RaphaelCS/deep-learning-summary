import torch
from torch import nn

from raphtools.data import FashionMNIST
from raphtools.dl import train


# Defining the Model
dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    # Add a dropout layer after the first fully connected layer
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    # Add a dropout layer after the second fully connected layer
    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        torch.nn.init.zeros_(m.bias)


net.apply(init_weights)
# Test accuray 到底用的是啥?


# Training and Testing
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = FashionMNIST.load(
    batch_size, "/mnt/e/DATASETS/FashionMNIST")
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, loss, num_epochs, optimizer)
