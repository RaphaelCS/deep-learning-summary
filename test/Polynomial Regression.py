import math

import numpy as np
import torch
from torch import nn

from raphtools.utils import Accumulator
from raphtools.data import load_array
from raphtools import plotter
from raphtools.plotter import Animator
from raphtools.dl import train_epoch


maxdegree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(maxdegree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(maxdegree).reshape(1, -1))
for i in range(maxdegree):
    poly_features[:, i] /= math.gamma(i+1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.01, size=labels.shape)

# Convert from NumPy to PyTorch tensors
true_w, features, poly_features, labels = [torch.from_numpy(x).type(
    torch.float32) for x in [true_w, features, poly_features, labels]]


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # sum_loss, num_examples
    for X, y in data_iter:
        loss_ = loss(net(X), y)
        metric.add(loss_.sum(), loss_.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=1000):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                            batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                           batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], ylim=[0, 1e2],
                        legend=['train', 'test'])
    for epoch in range(1, num_epochs+1):
        train_epoch(net, train_iter, loss, optimizer)
        if epoch % 50 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            animator.add(epoch, (train_loss, test_loss))
    print('weight:', net[0].weight)
    return train_loss, test_loss


# Pick the first k dimensions, i.e., 1, x, x^2,..., x^k from the polynomial
# features
train_losses, test_losses = [], []
for k in range(1, 21):
    train_loss, test_loss = train(
        poly_features[:n_train, 0:k], poly_features[n_train:, 0:k],
        labels[:n_train], labels[n_train:])
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# plotter.plot([train_losses, test_losses], xlabel='Complexity (degree)',
#              ylabel='Error', legend=['train_error', 'test_error'])
