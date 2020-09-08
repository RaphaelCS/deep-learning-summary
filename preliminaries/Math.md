# Calculus



## Gradient

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top
$$

- Vector:
  - For all $\mathbf{A} \in \R^{m \times n}$, $\nabla_\mathbf{x} \mathbf{Ax} = \mathbf{A}^\top$ 
  - For all $\mathbf{A} \in \R^{n \times m}$, $\nabla_\mathbf{x} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$ 
  - For all $\mathbf{A} \in \R^{n \times n}$, $\nabla_\mathbf{x} \mathbf{x}^\top \mathbf{Ax} = (\mathbf{A} + \mathbf{A}^\top) \mathbf{x}$ 
  - $\nabla_\mathbf{x} \Vert \mathbf{x} \Vert^2 = \nabla_\mathbf{x} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$ 

- Matrix:
  - $\nabla_\mathbf{X} \Vert \mathbf{X} \Vert^2_F = 2\mathbf{X}$ 



## Chain Rule

Suppose that functions 
$$
\begin{split}
y &= f(u)
\\&= f(u_1, u_2, \ldots, u_m)
\end{split}
$$

$$
\begin{split}
\forall i,\ u_i &= g(x)
\\&= g(x_1, x_2, \ldots, x_n)
\end{split}
$$

Then the chain rule gives 
$$
\frac{\partial y}{\partial x_i} = 
\frac{\partial y}{\partial u_1}\frac{\partial u_1}{\partial x_i} + 
\frac{\partial y}{\partial u_2}\frac{\partial u_2}{\partial x_i} + \cdots +
\frac{\partial y}{\partial u_m}\frac{\partial u_m}{\partial x_i}
$$
for any $i=1, 2, \ldots, n$.





# Probability

In any [exponential family](https://en.wikipedia.org/wiki/Exponential_family) model:

- the gradients of the log-likelihood are given by precisely $\hat{\mathbf{y}} - \mathbf{y}$. ([Reference](https://d2l.ai/chapter_linear-networks/softmax-regression.html#softmax-and-derivatives)) 





# Information Theory



## Self-information

Measures the information of a single discrete event
$$
I(X) = - \log_2(p)
$$

> Example: $\displaystyle I("0010") = - \log_2 (p("0010")) = -\log \left( \frac{1}{2^4} \right) = 4\ \mathrm{bits}$.





## *Cross-Entropy

Cross-entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.