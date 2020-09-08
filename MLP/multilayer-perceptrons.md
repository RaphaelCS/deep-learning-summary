# Multilayer Perceptrons

<img src=".\img\mlp.svg" alt="Multilayer Perceptrons" style="zoom:150%;" />



## Motivation

- Linearity is a strong assumption
  - Outputs are always monotonic with inputs
- We want to learn a representation of our data (the first $L-1$ layers), take into account the relevant interactions among our features, on top of which a linear model (the final layer) would be suitable



## Objective

- With deep neural networks, we used observational data to jointly learn both a representation via hidden layers and a linear predictor that acts upon that representation
  - Capture complex interactions among our inputs via their hidden neurons
- **Universal Approximators** 
  - Even with a single-hidden-layer network, given enough nodes (possibly absurdly many), and the right set of weights, we can model any function
  - Approximate many functions much more compactly by using **deeper** (vs. **wider**) networks
  - Learning that function is actually the hard part



## Model

$$
\begin{split}
\mathbf{H} &= \sigma ( \mathbf{XW}_1 + \mathbf{b}_1 )
\\
\mathbf{O} &= \mathbf{H} \mathbf{W}_2 + \mathbf{b}_2
\end{split}
$$

- We can add more hidden-layers recursively 

### Notations

- $\mathbf{X} \in \R^{n \times d}$  :  A minibatch of $n$ examples where each example has $d$ inputs (features) 
- $\mathbf{W}_1 \in \R^{d \times h}$  :  Hidden-layer weights 
- $\mathbf{b}_1 \in \R^{1 \times h}$  :  Hidden-layer biases 
- $\mathbf{H} \in \R^{n \times h}$  :  A minibatch of $n$ examples where each example has $d$ inputs (features) 
- $\mathbf{W}_2 \in \R^{h \times q}$  :  Output-layer weights 
- $\mathbf{b}_2 \in \R^{1 \times q}$  :  Output-layer biases 
- $\mathbf{O} \in \R^{n \times q}$  :  A minibatch of $n$ examples where each example has $q$ outputs 



### Incorporating Hidden Layers

- Stack many fully-connected layers on top of each other 

> Note: We choose layer widths in powers of 2, which tend to be computationally efficient because of how memory is allocated and addressed in hardware.





# Activation Functions



## Motivation

- Avoiding the collapse of MLP into linear model by its non-linearity
  - An affine function of an affine function is itself an affine function
- Limiting the domain of outputs for
  - Numerical stability (avoiding too large numbers)
  - Outputs' requirements (for example, probability)



## ReLU Function

Rectified Linear Unit (ReLU)
$$
\mathrm{ReLU}(x) = \max (x, 0)
$$

![ReLU](multilayer-perceptrons.assets\output_mlp_76f463_18_0.svg)

Derivatives

![ReLU's derivatives](multilayer-perceptrons.assets/output_mlp_76f463_30_0.svg)

### Advantages

- Its derivatives are particularly well behaved: either they vanish or they just let the argument through 
- Mitigate the well-documented problem of vanishing gradients 
- Simpler and more easily trainable 



### Variants

1. Parameterized ReLU (pReLU)
   $$
   \mathrm{pReLU} (x) = \max (0, x) + \alpha \min (0, x)
   $$
   [He et al., 2015](https://d2l.ai/chapter_references/zreferences.html#he-zhang-ren-ea-2015) 



## Sigmoid Function

Sigmoid (often called *squashing function*)
$$
\operatorname{sigmoid} (x) = \frac{1}{1 + \exp(-x)}
$$
![../_images/output_mlp_76f463_42_0.svg](multilayer-perceptrons.assets/output_mlp_76f463_42_0.svg)

- When the input is close to 0, the sigmoid function approaches a linear transformation 
- Point symmetry about $(0, \frac{1}{2})$ 

Derivatives 
$$
\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).
$$
![../_images/output_mlp_76f463_54_0.svg](multilayer-perceptrons.assets/output_mlp_76f463_54_0.svg)



### Usages

- A special case of the softmax: binary probability
- Control the flow of information across time in RNN 
- Squash any input in the range $(-\infty, \infty)$ to some value in the range $(0, 1)$ 



## Tanh Function

Tanh
$$
\operatorname{tanh} (x) = \frac{1 - \exp (-2x)}{1 + \exp (-2x)}
$$
![../_images/output_mlp_76f463_66_0.svg](multilayer-perceptrons.assets/output_mlp_76f463_66_0.svg)

- As the input nears 0, the tanh function approaches a linear transformation
- Point symmetry about the origin of the coordinate system

Derivatives
$$
\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).
$$
![../_images/output_mlp_76f463_78_0.svg](multilayer-perceptrons.assets/output_mlp_76f463_78_0.svg)

Relationship with sigmoid
$$
\tanh (x) + 1 = 2 \operatorname{sigmoid} (2x)
$$


### Usages

- Squash any input in the range $(-\infty, \infty)$ to some value in the range $(-1, 1)$ 

