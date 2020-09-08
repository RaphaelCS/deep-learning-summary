import torch

from raphtools.data import load_array
from raphtools.dl import linreg, squared_loss, sgd, evaluate_loss
from raphtools.plotter import Animator


def synthetic_linear_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.zeros(size=(num_examples, len(w))).normal_()
    y = X @ w + b
    y += torch.zeros(size=y.shape).normal_(std=0.01)
    return X, y


n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_linear_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_linear_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)


# Initializing Model Parameters
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# Defining l2 Norm Penalty
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], legend=['train', 'test'])
    for epoch in range(1, num_epochs + 1):
        for X, y in train_iter:
            with torch.enable_grad():
                # The L2 norm penalty term has been added, and broadcasting
                # makes 'L2_penalty(w)' a vector whose Length is 'batch_size'
                loss_ = loss(net(X), y) + lambd * l2_penalty(w)
            loss_.sum().backward()
            sgd([w, b], lr, batch_size)
        if epoch % 5 == 0:
            animator.add(epoch, (evaluate_loss(net, train_iter, loss),
                                 evaluate_loss(net, test_iter, loss)))
    print('l2 norm of w:', torch.norm(w).item())


train(lambd=0)

train(lambd=3)
