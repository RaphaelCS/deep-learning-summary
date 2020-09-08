# Optimization



## Minibatch Stochastic Gradient Descent

$$
(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \frac{\eta}{\vert \mathcal{B} \vert} \sum_{i \in \mathcal{B}} \nabla_{(\mathbf{w}, b)} l^{(i)}(\mathbf{w}, b)
$$

