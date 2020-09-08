# Pytorch



## Initializing Parameters

1. ```python
   net = nn.Sequential(nn.Linear(num_inputs, 1))
   for param in net.parameters():
       param.data.normal_()
   ```

2. ```python
   def init_weights(m):
       if type(m) == nn.Linear:
           torch.nn.init.normal_(m.weight, std=0.01)
           torch.nn.init.zeros_(m.bias)
   
   net.apply(init_weights)
   ```

3. 





## Optimizer Settings

```python
# Only decay weight, bias will not decay
trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
```





## Create Tensor

According to the shape of another tensor:

```python
torch.Tensor(X.shape).uniform_(0, 1)
```





## Build Model

- ```Sequential()``` has to forward in order
- ```nn.Module``` can forward in a specific order