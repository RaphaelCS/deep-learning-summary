import torch
from torch import nn

from raphtools.dl import train, predi
from raphtools.data import FashionMNIST


batch_size = 256
train_iter, test_iter = FashionMNIST.load(batch_size, "/mnt/e/DATASETS/FashionMNIST")


# Initializing Model Parameters
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


# Activation Function
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# Model
def net(X):
    X = torch.reshape(X, (-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)


# Loss Function
loss = nn.CrossEntropyLoss()


# Training
num_epochs, lr = 10, 0.5
updater = torch.optim.SGD(params, lr=lr)
train(net, train_iter, test_iter, loss, num_epochs, updater)


# Evaluate
def predict(net, test_iter, n=6):
    """Predict Labels."""
    for X, y in test_iter:
        break
    trues = FashionMNIST.get_labels(y)
    preds = FashionMNIST.get_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    FashionMNIST.show_images(
        X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])


predict(net, test_iter)
