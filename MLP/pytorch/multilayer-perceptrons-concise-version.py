import torch
from torch import nn

from raphtools.data import FashionMNIST
from raphtools.dl import train


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 784)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net = nn.Sequential(
    Reshape(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
net.apply(init_weights)


num_epochs, lr, batch_size = 10, 0.5, 256
train_iter, test_iter = FashionMNIST.load(
    batch_size, "/mnt/e/DATASETS/FashionMNIST")
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, loss, num_epochs, optimizer)
