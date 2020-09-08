import torch
from torch import nn

from raphtools.data import load_array
from raphtools.plotter import Animator
from raphtools.dl import evaluate_loss


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


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed. Bias names generally end with "bias"
    optimizer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}],
        lr=lr
    )

    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], legend=['train', 'test'])
    for epoch in range(1, num_epochs+1):
        for X, y in train_iter:
            with torch.enable_grad():
                optimizer.zero_grad()
                loss_ = loss(net(X), y)
            loss_.backward()
            optimizer.step()
        if epoch % 5 == 0:
            animator.add(epoch, (evaluate_loss(net, train_iter, loss),
                                 evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())


train_concise(0)

train_concise(3)
