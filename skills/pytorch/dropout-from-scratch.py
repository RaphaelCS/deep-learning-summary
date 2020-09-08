import torch
from torch import nn

from raphtools.data import FashionMNIST
from raphtools.dl import train


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)


# X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
# print(X)
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))


# Defining Model Parameters
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256


# Defining the Model
dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()

        self.num_inpputs = num_inputs
        self.training = is_training

        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)

        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inpputs))))
        # Use dropout only when training the model
        if self.training is True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, dropout1)
            self.H1 = H1
        H2 = self.relu(self.lin2(H1))
        if self.training is True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)


# Training and Testing
num_epochs, lr, batch_size = 10, 0.5, 1
loss = nn.CrossEntropyLoss()
train_iter, test_iter = FashionMNIST.load(
    batch_size, "/mnt/e/DATASETS/FashionMNIST")
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, loss, num_epochs, optimizer)


def test_gradients_wrt_zeros_are_zero(net, train_iter, loss, updater):
    X, y = next(iter(train_iter))
    # Compute gradients and update parameters
    y_hat = net(X)
    loss_ = loss(y_hat, y)
    if isinstance(updater, torch.optim.Optimizer):
        updater.zero_grad()
        loss_.backward()
    else:
        loss_.sum().backward()
    # print("H1: ", net.H1.shape, net.H1)
    # print("weight.grad: ", net.lin2.weight.shape, net.lin2.weight.grad[:, 1])
    for j in range(net.H1.shape[1]):
        if net.H1[0, j] == 0:
            print(all(net.lin2.weight[:, j]))


test_gradients_wrt_zeros_are_zero(net, train_iter, loss, optimizer)
