---
layout: post
title: "Jacobian, Chain rule and backpropagation"
date: 2018-04-04
mathjax: true
status: [Instructional]
categories: [Calculus, Machine Learning]
---

### Model Preliminaries
**The Jacobian matrix and backpropogation**
<br>

* The Jacobian of a $f:\mathbb{R}^n \rightarrow \mathbb{R}^m$ is a matrix of first-order partial derivatives of a vector-valued function. This is a generalization of the *gradients* (derivative of multivariate input functions with a *scalar* output) to the case where there is a *vector* output, i.e. $f(x) \in \mathbb{R}^m$. 

* The Jacobian matrix captures the rate of change of each jth component of $f$ with respect to each ith component of the input vector $x$. where $f_j$ for $j \in \\{1, 2, ..., m \\}$ and $x_i$ for $i \in \\{1, 2, ..., n\\}$. Then the element $J_{i,j}$ represents the rate of change of $f_j$ with respect to a change in $x_i$.

\begin{equation}
J = [\frac{\delta(f)}{\delta_(x_1)} ... \frac{\delta(f)}{\delta_(x_n)}] = 
\begin{bmatrix}
\frac{\delta f_1}{\delta x_1} & \cdots & \frac{\delta f_1}{\delta x_n} \\\
\vdots & \ddots & \vdots \\\
\frac{\delta f_m}{\delta x_1} & \cdots & \frac{\delta f_m}{\delta x_n}
\end{bmatrix}
\end{equation}
* It can be used when we need to compute the partial derivatives of multiple functions. E.g if $f(x, y) = 3x^2y$ and $g(x y) = 2x + y^8$, we can stack the gradients of the two functions into a Jacobian matrix, so that each row is a function and each column is the varible which we are differentiating with respect to:

\begin{equation}
J = \begin{bmatrix}
\nabla f(x, y) \\\
\nabla g(x, y) 
\end{bmatrix}  
=
\begin{bmatrix}
\frac{\delta f(x, y)}{\delta x} & \frac{\delta f(x, y)}{\delta y} \\\
\frac{\delta g(x, y)}{\delta x} & \frac{\delta g(x, y)}{\delta y} 
\end{bmatrix}
\end{equation}


**Chain rule and Calculating Derivatives with Computation Graphs (through backpropagation)**


In supervised learning, we wish to find the parameters which minimise some loss function: $\hat{\theta} = \mathrm{argmin}_{\theta} \mathcal{L}(y, f(x; \theta))$. If $f$ is a deep neural network, the layers define a series of composite mathematical operations. The weights are trained by **backpropagation**, which is a propagation of the error contributions (via the gradients) through the weights of the network. 

We wish to optimize  weights which transform the input from one layer to the next, and thus are interested in the gradients. These gradients can be several network layers away from the loss. The chain rule of calculus allows us to compute the derivatives of this composite function to compute the influence of early layer parameters on the final loss. Formally, if $f(x) = g(h(x))$, then by the chain rule: $\frac{\delta f}{\delta x} = \frac{\delta g}{\delta h} \times \frac{\delta h}{\delta x}$. Thus the Jacobian arises when we apply the chain rule to compute gradients for training neural networks. 


Given a fully connected neural net with 3 inputs, $x_1, x_2, x_3$, and hidden layer with
  two neurons $z_1, z_2$, and a single scalar output $y$. Then, the derivative of the lower
layers can be decomposed and nicely computed with matrix multiplication: 

$$
\begin{align}
\frac{\delta y}{\delta x} &= \frac{\delta y}{\delta z} \times \frac{\delta z}{\delta x} \\\
&= 
\begin{bmatrix}
\frac{\delta y}{\delta z_1} & \frac{\delta y}{\delta z_2}
\end{bmatrix} \times 
\begin{bmatrix}
\frac{\delta z_1}{\delta x_1} & \frac{\delta z_1}{\delta x_2} & \frac{\delta z_1}{x_3} \\\
\frac{\delta z_2}{\delta x_1} & \frac{\delta z_2}{\delta x_2} & \frac{\delta z_2}{x_3} 
\end{bmatrix}
\end{align}
$$
