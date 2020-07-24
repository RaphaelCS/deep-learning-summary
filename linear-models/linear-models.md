# Linear Regression

<img src=".\linear-models.assets\singleneuron.svg" alt="Not Found" style="zoom:150%;" />



## Assumptions

1. The relationship between the independent variables $\boldsymbol{x}$ and the dependent variable $\boldsymbol{y}$ is **linear**, given some noise on the observations.
2. Any noise is well-behaved (following a **Gaussian distribution**)



## Model

- Single data instance:

  $$
  \begin{split}
  \hat{y} &= w_1x_1 + \cdots + w_dx_d + b
  \\&= \mathbf{w}^\top \mathbf{x} + b
  \end{split}
  $$

- $n$ examples:
  $$
  \mathbf{\hat{y}} = \mathbf{Xw} + b
  $$

### Notation

- $\mathbf{x} \in \R^d$  :  Independent variables 
- $y$  :  Dependent variable
- $\hat{y}$  :  Prediction 
- $\mathbf{w} \in \R^d$  :  Weights 
- $b$  :  Bias 



- $\mathbf{X} \in \R^{n \times d}$  :  Design matrix, containing one row for each example
- $\mathbf{\hat{y}} \in \R^n$  :  Predictions



## Loss Function

Squared error:
$$
\begin{split}
l^{(i)}(\mathbf{w}, b) &= \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
\\
L(\mathbf{w}, b) &= \frac{1}{n} \sum_{i=1}^n l^{(i)}(\mathbf{w}, b)
\\&= \frac{1}{n} \sum_{i=1}^n \frac{1}{2} \left( \mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)} \right)^2
\end{split}
$$

### Motivation of Squared Error Loss



#### Assumptions

- $$
  \epsilon \sim \mathcal{N}(0, \sigma^2).
  $$

  , where $\epsilon = y - (\mathbf{w}^\top \mathbf{x} + b)$.

- $\sigma$ is a constant 



#### Proof

The *likelihood* of seeing a particular $y$ for a given $\mathbf{x}$ is
$$
P(y\ \vert\ \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} 
\exp \left( -\frac{1}{2\sigma^2} (y - \mathbf{w^\top x} - b)^2 \right)
$$
Thus, the *likelihood* of the entire dataset is
$$
P(\mathbf{y}\ \vert\ \mathbf{X}) = 
\prod_{i=1}^n p(y^{(i)} | \mathbf{x}^{(i)})
$$
According to *maximum likelihood estimation*, which is equivalent to minimize 
$$
-\log P(\mathbf{y} \mid \mathbf{X}) = 
\sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + 
\frac{1}{2\sigma^2} \sum_{i=1}^n \left( y^{(i)} - \mathbf{w^\top} \mathbf{x}^{(i)} - b \right)^2
$$

Because $\sigma$ is supposed to be constant, *maximum likelihood estimation* is equivalent to the *mean squared error* loss.



## Optimization

$$
\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\ L(\mathbf{w}, b)
$$



### Analytic Solution

Considering an easier form 
$$
L(\mathbf{w}) = \Vert \mathbf{y} - \mathbf{Xw} \Vert^2
$$
, we take derivative of the loss with respect to $\mathbf{w}$
$$
\begin{split}
\nabla_\mathbf{w} L &= \nabla_\mathbf{w}u \cdot \nabla_\mathbf{u}L
\\&= -\mathbf{X}^\top \cdot 2\mathbf{u}
\end{split}
$$
where $\mathbf{u} = \mathbf{y} - \mathbf{Xw}$. Set it equal to zero, then we have the closed-form solution
$$
\mathbf{w}^* = (\mathbf{X^\top X})^{-1} \mathbf{X^\top y}
$$
if $\mathbf{X^\top X}$ is invertible. If it is not, we can solve this problem by re-design $\mathbf{X}$ to satisfy the full-rank restriction.

> Note: Whether $\mathbf{X^\top X}$ is invertible or not, the hessian is always semi-definite positive
> $$
> \begin{split}
> L(\mathbf{w+h}) 
> &= L(\mathbf{w}) + \mathbf{h}^\top \nabla_\mathbf{w}L + \mathbf{h^\top X^\top X h}
> \\&= L(\mathbf{w}) + \mathbf{h}^\top \nabla_\mathbf{w}L + \Vert \mathbf{Xh} \Vert^2_2
> \end{split}
> $$



### Minibatch SGD

$$
\begin{split}
\mathbf{w} &\ \leftarrow\ 
\mathbf{w} &\ -\ \frac{\eta}{\vert \mathcal{B} \vert} \sum_{i \in \mathcal{B}} \nabla_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) 
&\ =\ \mathbf{w} &\ -\ \frac{\eta}{\vert \mathcal{B} \vert} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left( \mathbf{w^\top x}^{(i)} + b - y^{(i)} \right) 
\\ 
b &\ \leftarrow\ 
b &\ -\ \frac{\eta}{\vert \mathcal{B} \vert} \sum_{i \in \mathcal{B}} \nabla_{b} l^{(i)}(\mathbf{w}, b) 
&\ =\ b &\ -\ \frac{\eta}{\vert \mathcal{B} \vert} \sum_{i \in \mathcal{B}} \left( \mathbf{w^\top x}^{(i)} + b - y^{(i)} \right) 
\end{split}
$$

, where $\mathcal{B}$ is a sampled minibatch consisting of a fixed number of training examples, $\eta > 0$.





# Softmax Regression

<img src=".\linear-models.assets\softmaxreg.svg" alt="Not Found" style="zoom:150%;" />



## Model

- Single data instance:

  $$
  \mathbf{o} = \mathbf{Wx} + \mathbf{b}
  $$

  , here $\mathbf{W} \in \R^{q \times d}$, $\mathbf{b} \in \R^q$.
  $$
  \hat{\mathbf{y}} = \mathrm{softmax} (\mathbf{o}) 
  \quad \text{where} \quad 
  \hat{y}_j = \frac{\exp (o_j)}{\sum_k \exp (o_k)}
  $$

  > Note : $\operatorname*{argmax}\limits_j \hat{y}_j = \operatorname*{argmax}\limits_j o_j$ 

- $n$ examples:
  $$
  \mathbf{O} = \mathbf{XW} + \mathbf{b}
  $$
  
  , here $\mathbf{W} \in \R^{d \times q}$, $\mathbf{b} \in \R^{1 \times q}$.
  $$
  \hat{\mathbf{Y}} = \operatorname*{softmax} (\mathbf{O})
  $$

### Notation

- $\mathbf{x} \in \R^d$  :  Inputs
- $\mathbf{y} \in \R^q$  :  A one-hot label vector 
- $\mathbf{o} \in \R^q$  :  Outputs
- $\hat{\mathbf{y}} \in \R^{q}$  :  Output probabilities
- $\mathbf{W}$  :  Weights
- $\mathbf{b}$  :  Biases



- $\mathbf{X} \in \R^{n \times d}$  :  Design matrix
- $\mathbf{O} \in \R^{n \times q}$  :  Outputs, containing one row for each example's outputs
- $\hat{\mathbf{Y}} \in \R^{n \times q}$  :  Output probabilities, containing one row for each example's output probabilities



## Loss Function

Cross-Entropy: 
$$
\begin{split}
l(\mathbf{y}, \hat{\mathbf{y}}) 
&= -\sum_{j=1}^q y_j \log \hat{y_j} 
\\&= - \sum_{j=1}^q y_j \log \frac{\exp (o_j)}{\sum_{k=1}^q \exp (o_k)} 
\\&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp (o_k) - \sum_{j=1}^q y_j o_j
\\&= \log \sum_{k=1}^q \exp (o_k) - \sum_{j=1}^q y_j o_j
\\
L(\mathbf{Y}, \hat{\mathbf{Y}}) &= \sum_{i=1}^n 
l^{(i)}(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}) 
\end{split}
$$

> Note 1: This simplification is very useful because of avoiding overflow and underflow (cause by large exponential).
>
> Note 2: In PyTorch (same in other DL lib), we use ```nn.CrossEntropyLoss()``` which has more optimization in coding, such as [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp), instead of ```softmax``` + ```cross-entropy```. [More information](https://d2l.ai/chapter_linear-networks/softmax-regression-concise.html#softmax-implementation-revisited) 



### Motivation of Cross-Entropy Loss

The *likelihood* of the entire dataset is
$$
P (\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P (\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
$$
According to *maximum likelihood estimation*, which is equivalent to minimize the negative log-likelihood
$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = 
\sum_{i=1}^n -\log P (\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}) = 
\sum_{i=1}^n l (\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)})
$$

> Note : Maximizing the likelihood of the observed data here also means minimizing our surprisal required to communicate the labels.



## Optimization

$$
\mathbf{W}^*, \mathbf{b}^* = \operatorname*{argmin}_{\mathbf{W}, \mathbf{b}}\ L(\mathbf{W}, \mathbf{b})
$$



### Minibatch SGD

First derivative
$$
\begin{split}
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) &= \operatorname*{softmax}(\mathbf{o})_j - y_j 
\\&= \hat{y}_j - y_j
\end{split}
$$
Second derivative
$$
\partial^2_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \operatorname*{softmax} (\mathbf{o})_j (1 - \operatorname*{softmax} (\mathbf{o})_j)
$$
, which is equal to the variance of the distribution given by $\operatorname*{softmax} (\mathbf{o})_j$.





# Usages

1. Provide a sanity check:
   - See whether there is meaningful information in the data. 
   - If we cannot do better than random guessing here, then there might be a good chance that we have a data processing bug. 
   - If things work, the linear model will serve as a baseline giving us some intuition about how close the simple model gets to the best reported models, giving us a sense of how much gain we should expect from fancier models.



