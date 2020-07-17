import torch
from torch import nn

from raphtools.dl import train
from raphtools.data import FashionMNIST


batch_size = 256
train_iter, test_iter = FashionMNIST.load(
    batch_size, "/mnt/e/DATASETS/FashionMNIST")


# Initializing Model Parameters
# PyTorch does not implicitly reshape the inputs. Thus we define a layer to
# reshape the inputs in our network
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 784)


@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net = nn.Sequential(Reshape(), nn.Linear(784, 10))
net.apply(init_weights)


# Softmax + Cross-Entropy
loss = nn.CrossEntropyLoss()


# Optimization Algorithm
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
train(net, train_iter, test_iter, loss, num_epochs, optimizer)
