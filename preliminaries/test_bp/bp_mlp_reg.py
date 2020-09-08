# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Test my backpropagation calculus

# %%
import numpy as np
import torch
from torch import nn

# IN_SIZE, OUT_SIZE = 784, 100
# HIDDEN_SIZE = [512, 1024, 1024]

IN_SIZE, OUT_SIZE = 2, 1
HIDDEN_SIZE = []

# %%
x = torch.empty((1, IN_SIZE)).uniform_(-10, 10)
y = torch.empty((1, OUT_SIZE)).uniform_(-10, 10)


# %%
class TestNet(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output):
        super(TestNet, self).__init__()

        n_layers = [n_input] + n_hiddens + [n_output]
        self.layers = nn.ModuleList()
        for i in range(len(n_layers) - 1):
            self.layers.append(nn.Linear(n_layers[i], n_layers[i+1]))
    
    def forward(self, X):
        self.z, self.a = [], [X]
        for layer in self.layers:
            X = layer(X)
            self.z.append(X)
            X = torch.sigmoid(X)
            self.a.append(X)
        return X

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -10, 10)
        nn.init.uniform_(m.bias, -10, 10)


# %%
net = TestNet(IN_SIZE, HIDDEN_SIZE, OUT_SIZE)
net.apply(init_weights)


# %%
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
optimizer.zero_grad()

loss = nn.MSELoss()
loss(net(x), y).sum().backward()


# %%
params = [param for param in net.parameters()]
# for param in params:
#     print(param.shape)


# %%
def D_sigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def calculate_gradients(num_layers=4):
    delta = [torch.empty(net.z[i].shape[1]) for i in range(num_layers - 1)]
    delta.append((net.a[-1] - y) * D_sigmoid(net.z[-1]))
    delta[-1] = delta[-1].reshape((-1, 1))

    for i in range(num_layers - 1, 0, -1):
        delta[i - 1] = (params[i * 2].T @ delta[i]) * D_sigmoid(net.z[i - 1]).T
    
    gradients = []
    for i in range(len(delta)):
        gradients.append(delta[i] @ net.a[i])
        gradients.append(delta[i])
    
    return gradients


# %%
gradients = calculate_gradients(len(HIDDEN_SIZE) + 1)


# %%
for i, param in enumerate(params):
    torch_grad = param.grad if len(param.shape) > 1 else param.grad.view(-1, 1)
    my_grad = gradients[i]
    diff = abs(torch_grad - my_grad)
    print(torch_grad, my_grad)
    # print(diff[diff > 1])
    # print(torch.max(diff))


# %%
x = D_sigmoid(x)
from raphtools import plotter
plotter.plot(x)




# %%
