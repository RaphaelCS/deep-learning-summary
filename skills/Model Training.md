# Model Training



## Bias-variance tradeoff

https://liam.page/2017/03/25/bias-variance-tradeoff/

|                      | Bias | Variance |
| -------------------- | ---- | -------- |
| Linear Models        | High | Low      |
| Deep Neural Networks | Low  | High     |





## Model Selection

1. Validation dataset
   - Large datasets
2. K-fold cross-validation
   - Small datasets





## Overfitting



### Solutions

1. More training data 
2. Limiting the number of features 
3. Regularization
4. Add noise to inputs
   - https://d2l.ai/chapter_multilayer-perceptrons/dropout.html#robustness-through-perturbations
5. Dropout



## Numerical Stability and Initialization



### Vanishing Gradients

$0 \leq \displaystyle \frac{\mathrm{d} \operatorname{sigmoid}(x)}{\mathrm{d} x} \leq \frac{1}{4}$ , thus the gradients will vanish if we multiply together too many derivatives smaller than 1.

#### Solutions

1. **ReLU** activation functions mitigate the vanishing gradient problem. This can accelerate convergence.



### Exploding Gradients

If we draw 100 Gaussian random matrices and multiply them with some initial matrix. For the scale that we picked (the choice of the variance $σ^2=1$), the matrix product explodes. When this happens due to the initialization of a deep network, we have no chance of getting a gradient descent optimizer to converge.



### Symmetry Inherent in the Parametrization

Imagine what would happen if we initialized all of the parameters of the hidden layer as $W^{(1)}=c$ for some constant $c$. In this case, during forward propagation either hidden unit takes the same inputs and parameters, producing the same activation, which is fed to the output unit. During backpropagation, differentiating the output unit with respect to parameters $W^{(1)}$ gives a gradient whose elements all take the same value. Thus, after gradient-based iteration (e.g., minibatch stochastic gradient descent), all the elements of $W^{(1)}$ still take the same value. Such iterations would never *break the symmetry* on its own and we might never be able to realize the network’s expressive power. The hidden layer would behave as if it had only a single unit.

[More information](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#breaking-the-symmetry) 

#### Solutions

- **Random initialization** is key to ensure that symmetry is broken before optimization.



### Parameter Initialization



#### Xavier Initialization

Xavier initialization suggests that, for each layer, variance of any output is not affected by the number of inputs, and variance of any gradient is not affected by the number of outputs.
$$
w \sim U \left( 
- \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}} , \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}
\right)
\quad\text{or}\quad 
N\left( 0,\
\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}} 
\right)
$$
<u>Proof:</u>

Suppose non-linearity, so 
$$
o_i = \sum_{j=1}^{n_\text{in}}w_{ij}x_j
$$
, where $E[w_{ij}] = 0, \mathrm{Var}[w_{ij}] = \sigma^2$, $E[x_j] = 0, \mathrm{Var}[x_j] = \sigma^2$. Thus, 
$$
\begin{split}\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}\end{split}
$$
One way to keep the variance fixed is to set $n_\text{in}\sigma^2=1$. Similarly, in backpropagation, the gradients' variance should be set to $n_\text{out}\sigma^2=1$. So we have 
$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$
If we use uniform distribution $U(-a, a)$ whose variance is $\displaystyle \frac{a^2}{3}$, then the result.



## Environment and Distribution Shift

- Our ill-considered leap from pattern recognition to decision-making and our failure to critically consider the environment might have disastrous consequences.
- By introducing our model-based decisions to the environment, we might break the model.



### Distribution Shift



#### Covariate Shift

<u>Assumptions:</u>

- While the distribution of inputs may change over time, the labeling function, i.e., the conditional distribution $P(y \mid \mathbf{x})$ does not change.

  > Note: Statisticians call this *covariate shift* because the problem arises due to a shift in the distribution of the covariates (features).

<u>Usage:</u>

1. Covariate shift is the natural assumption to invoke in settings where we believe that $\mathbf{x}$ causes $y$.

<u>Examples:</u>

1. Training on photos, while testing on cartoons 



#### Label Shift

<u>Assumptions:</u>

- The label marginal $P(y)$ can change but the class-conditional distribution $P(\mathbf{x} \mid y)$ remains fixed across domains. 

<u>Usage:</u>

1. When we believe that $y$ causes $\mathbf{x}$.

<u>Examples:</u>

1. Predict diagnoses given their symptoms (or other manifestations), even as the relative prevalence of diagnoses are changing over time.



#### Concept Shift

<u>Assumptions:</u>

- The very definitions of labels can change.

<u>Usage</u>:

1. Categories are subject to changes in usage over time.

<u>Examples:</u>

1. Diagnostic criteria for mental illness, what passes for fashionable, and job titles, are all subject to considerable amounts of concept shift.
2. If we navigate around the United States, shifting the source of our data by geography, we will find considerable concept shift regarding the distribution of names for *soft drinks*.
3. If we were to build a machine translation system, the distribution $P(y \mid \mathbf{x})$ might be different depending on our location. 



# Regularization



## $L_2$-regularization (weight decay)



### Motivation

- We have assumed a prior belief that weights take values from a Gaussian distribution with mean zero.
- More intuitively, we might argue that we encouraged the model to spread out its weights among many features rather than depending too much on a small number of potentially spurious associations.



### Ridge Regression

$$
\operatorname*{argmin}_{\mathbf{w}, b}\ \frac{1}{n} \sum_{i=1}^n\frac{1}{2}\left( \mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)} \right)^2 + \frac{\lambda}{2} \Vert \mathbf{w} \Vert^2
$$

Stochastic gradient descent
$$
\mathbf{w} \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),
$$

> Attention: Often, we do not regularize the bias term of a network’s output layer. Otherwise, a substantial underfitting will happen.



### Advantages

1. Outsize penalty on large components of the weight vector, thus distribute weight **evenly** across a larger number of features 
   - More **robust** to measurement error in a single variable 





## $L_1$-regulariaztion



### Advantages

1. Concentrate weight on a small set of features 





## Dropout

<img src="Model Training.assets/dropout2.svg" alt="../_images/dropout2.svg" style="zoom:120%;" />



- Training: Dropout replaces an activation $h$ with a random variable with expected value $h$:
  In standard dropout regularization, one debiases each layer by normalizing by the fraction of nodes that were retained (not dropped out). In other words, with *dropout probability* $p$, each intermediate activation $h$ is replaced by a random variable $h'$ as follows:
  $$
  h' = 
  \begin{cases}
  0 & \text{with probability } p
  \\
  \frac{h}{1-p} & \text{otherwise}
  \end{cases}
  $$
  By design, the expectation remains unchanged, i.e., $E[h′]=h$.

  > Note1 : When a hidden unit is set to 0, the weight corresponding to it will also have 0 gradient while optimizing, i.e. no update in this turn.
  >
  > Note2 : There are multiple ways to obtain same scale of the outputs, here for instance, we multiply $\displaystyle \frac{1}{1-p}$. See more about this in Dropout's original paper.



- Validation & Testing: Typically, we disable dropout at validation/test time. 

  > Note: But sometimes for estimating the *uncertainty* of neural network predictions: if the predictions agree across many different dropout masks, then we might say that the network is **more confident**.

<u>Reasons</u>:

1. It is kind of like *bagging*.



### Advantages

- The calculation of the output layer cannot be overly dependent on any one element (inputs or hidden units)



### Skills

- A common trend is to set a lower dropout probability closer to the input layer.
- In practice, we usually use a *dropouts* of 0.2 - 0.5 (0.2 recommended for input layer). 
  - Too low dropout cannot have regularization effect
  - Too high dropout may lead to underfitting 
- More parameters, higher probability 



