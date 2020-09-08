# Loss Functions



## Mean Squared Error (MSE)

$$
\begin{split}
l(\hat{y}^{(i)}, y^{(i)}) &= \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
\\
L(\hat{\mathbf{y}}, \mathbf{y}) &= \frac{1}{n} \sum_{i=1}^n l(\hat{y}^{(i)}, y^{(i)})
\\&= \frac{1}{2n} \sum_{i=1}^n \left( \hat{y}^{(i)} - y^{(i)} \right)^2
\end{split}
$$

<u>Reasons</u>:

1. If the noise of labels follows normal distribution, this loss function is obtained by MLE.

   > Example: Squared error in Linear Regression.



## Cross-Entropy Loss

$$
\begin{split}
l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}) 
&= -\sum_{j=1}^q y_j \log \hat{y_j}
\\
L(\mathbf{Y}, \hat{\mathbf{Y}}) &= \frac{1}{n} \sum_{i=1}^n 
l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}) 
\\&= \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^q - y_j \log \hat{y_j}
\end{split}
$$

If insert a Softmax operation before loss function, then
$$
\begin{split}
l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}) 
&= -\sum_{j=1}^q y_j \log \hat{y_j} 
\\&= - \sum_{j=1}^q y_j \log \frac{\exp (o_j)}{\sum_{k=1}^q \exp (o_k)} 
\\&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp (o_k) - \sum_{j=1}^q y_j o_j
\\&= \log \sum_{k=1}^q \exp (o_k) - \sum_{j=1}^q y_j o_j
\end{split}
$$
<u>Reasons</u>:

1. According to information theory and MLE.

   > Example: Cross-entropy loss in Softmax Regression.



## Root-Mean-Squared-Error between the Logarithm

$$
\begin{split}
l(y_i, \hat{y}_i) &= \left\vert \log \frac{y_i}{\hat{y}_i} \right\vert ^2
\\&= \left(\log y_i -\log \hat{y}_i\right)^2
\\
L(\mathbf{y}, \hat{\mathbf{y}}) &= \sqrt{ \frac{1}{n} \sum_{i=1}^n
l(y_i, \hat{y}_i) }
\\&= \sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}
\end{split}
$$

<u>Usages</u>:

1. We care about **relative** quantities $\displaystyle \frac{y - \hat{y}}{y}$ more than absolute quantities $y - \hat{y}$.

   > Explanation: A small value $\delta$ for $|\log y - \log \hat{y}| \leq \delta$ translates into $\displaystyle e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.