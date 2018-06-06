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

* The Jacobian of a $f:\mathbb{R}^n \rightarrow \mathbb{R}^m$ is a matrix of first-order partial derivatives of a vector-valued function. This is simply a generalization of gradients(derivatives of multivariate input functions with scalar output) to the case where there is a vector output, i.e. $f(x) \in \mathbb{R}^m$. 

* Intuitively, it captures the rate of change of each jth component of $f$ with respect to each ith component of the input vector $x$. where $f_j$ for $j \in \\{1, 2, ..., m \\}$ and $x_i$ for $i \in \\{1, 2, ..., n\\}$. Then the element $J_{i,j}$ represents the rate of change of $f_j$ with respect to a change in $x_i$.

\begin{equation}
J = [\frac{\delta(f)}{\delta_(x_1)} ... \frac{\delta(f)}{\delta_(x_n)}] = 
\begin{bmatrix}
\frac{\delta f_1}{\delta x_1} & \cdots & \frac{\delta f_1}{\delta x_n}
\\\
\vdots & \ddots & \vdots
\\\
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

* In the context of Machine Learning, it can be used whenever there is a multivariate input to resulting in a vector (non-scalar) output.  

* A special case is when there is an element wise operation $f:\mathbb{R}^n \rightarrow \mathbb{R}^m$. Then the Jacobian of $f$ becomes a diagonal matrix whose $(i, i)$ entry is the value of the element wise operation.

**In the context of neural networks**

* Let $h = f(g(x))$, $h: \mathbb{R}^n \rightarrow \mathbb{R}^m$ describes a transformation from a layer of $n$ input nodes to the next layer of $m$ nodes. $g(x)$ describes the standard linear transformation $g(x) = Wx+b$, and $f$ is a non-linear differentiable function such as tanh. $W$ is a $n\times m$ weight matrix which corresponds to the edges of the $n$ input nodes to $m$ output nodes. 

* The Jacobian of $g$ with respect to $x$, is denoted by $J_x(g)$ and is equivalent to $W$ because of the linear relationship of $W$ and $x$. Note that each element $i, j$ in $J_x(g)$ represents how the output node $j$ changes with respect to changes in $i$.

* The Jacobian may also arise in the differentiation of the non-linear function $f: \mathbb{R}^m \rightarrow \mathbb{R}^m$. To differentiate from the above notation, let $\theta = g(x)$. Let $f$ be the sigmoid function applied element wise to $\theta$. 

\begin{equation}
f(\theta_i) = \frac{1}{1+exp(-\theta_i)}
\end{equation}

* As the sigmoid is an element wise operation, the Jacobian is a diagonal matrix $J_\theta(f)$ whose $(i, i)$ entry is $f'(\theta_i)$, that is:

\begin{equation}
J_{\theta_i}(f) = \frac{1}{(1+exp(\theta_i)).(1+exp(-\theta_i))}
\end{equation}

* The Jacobian also functions like a stacked gradient vector for $n$ input instances. 
**Chain rule and Calculating Derivatives with Computation Graphs (through backpropagation)**
<br>
* The chain rule of calculus is a way to calculate the derivatives of composite functions. Formally, if $f(x) = f(g(x))$, then by the chain rule: $\frac{\delta f}{\delta x} = \frac{\delta f}{\delta g} \times \frac{\delta g}{\delta x}$. This can be generalised to multivariate vector-valued functions. The Jacobian J of $f(g(x))$ is given by $J_{f(g)}(x) = J_fg(x).J_g(x)$

* It can be used whenever we want to optimise parameters of composite functions.

* In the context of Neural networks, the layers define a series of composite mathematical operations. We wish to optimize  weights which transform the input from one layer to the next, and thus are interested in the gradients. These gradients can be several network layers away from the loss. The chain rule thus provides a convenient tool which allows us to correctly find the influence of early layer parameters on the final loss.

* Local gradients .. 

Suppose we define a toy computational graph with the following

$Y = f^o(U\times f^h(W\times x+b) + b_u)$

\begin{equation}
q = x^2+2x
y = q+c
\end{equation}


