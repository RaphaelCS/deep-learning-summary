import torch
from torch.utils import data
from torch import nn


# Generating the Dataset
def synthetic_data(w, b, num_example):
    """Generate y = Xw + b + noise."""
    X = torch.zeros(size=(num_example, len(w))).normal_()
    y = torch.matmul(X, w) + b
    y += torch.zeros(size=y.shape).normal_(std=0.01)
    return X, y


true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
labels = labels.reshape(-1, 1)


# Reading the Dataset
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)


# Defining the Model
net = nn.Sequential(nn.Linear(2, 1))


# Initializing Model Parameters
net[0].weight.data.uniform_(0.0, 0.01)
net[0].bias.data.fill_(0)


# Defining the Loss Function
loss = nn.MSELoss()


# Defining the Optimization Algorithm
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# Training
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        loss_ = loss(net(X), y)
        trainer.zero_grad()
        loss_.backward()
        trainer.step()
    loss_ = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {loss_:f}')

w = net[0].weight.data
print('error in estimating w:', true_w - torch.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
