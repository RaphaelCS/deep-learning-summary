import torch

from raphtools import plotter


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)


# ReLU
y = torch.relu(x)
plotter.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

y.backward(torch.ones_like(y))
plotter.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))


# Sigmoid
y = torch.sigmoid(x)
plotter.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

x.grad.zero_()
y.backward(torch.ones_like(y))
plotter.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))


# Tanh
y = torch.tanh(x)
plotter.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

x.grad.zero_()
y.backward(torch.ones_like(x))
plotter.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
