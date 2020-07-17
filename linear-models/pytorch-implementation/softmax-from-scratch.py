import torch

from raphtools.data import FashionMNIST
from raphtools import plotter
from raphtools.plotter import Animator
from raphtools.utils import Accumulator
plotter.use_svg_display()


# Setting up a data iterator with batchsize 256
batch_size = 256
train_iter, test_iter = FashionMNIST.load(
    batch_size, "/mnt/e/DATASETS/FashionMNIST")


# Initializing Model Parameters
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


# Defining the Softmax Operation
def softmax(X):
    X_exp = torch.exp(X)
    partition = torch.sum(X_exp, dim=1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here

# X = torch.normal(0, 1, size=(2, 5))
# X_prob = softmax(X)
# print(X_prob, torch.sum(X_prob, dim=1))


# Defining the Model
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

# X = torch.normal(0, 1, size=(1000, 784))
# print(W.shape[0])
# print(X.shape, X.reshape(-1, W.shape[0]).shape)


# Cross-Entropy Loss Function
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


# Classification Accuracy
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.type(y.dtype) == y).sum())


def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    # Sum of training Loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        loss_ = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            loss_.backward()
            updater.step()
            """WHY"""
            metric.add(
                (float(loss_) * len(y), accuracy(y_hat, y), y.size().numel())
            )
        else:
            loss_.sum().backward()
            updater(X.shape[0])
            metric.add(
                float(loss_.sum()), accuracy(y_hat, y), y.size().numel()
            )
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()


lr = 0.1


def updater(batch_size):
    return sgd([W, b], lr, batch_size)


num_epochs = 10
train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


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
