---
layout: post
title: "Lagrange Multipliers and Constrained Optimization"
date: 2018-04-7
mathjax: true
categories: [Calculus, Optimization]
---
**Lagrange Multipliers and constrained optimization**
* The method of lagrange multipliers is a strategy for finding the local minima and maxima of a differentiable function, $f(x_1, ... , x_n):\mathbb{R}^n \rightarrow \mathbb{R}$ subject to equality constraints on its independent variables.

* In constrained optimization, we have additional restrictions on the values which the independent variables can take on. The constraint can be expressed by the function $g(x_1, ..., x_n)$, and points which satisfy our constraint belong to the feasible region. Equality constraints restrict the feasible region to points lying on some surface inside $\mathbb{R}^n$. ![Fig1](/assets/Calculus-contour-constrain.png)

* The point at which the function and constraint are tangent to each other, i.e. touch but do not cross is the solution to the maximum or minimum of the optimization problem. The intuition is that $f$ cannot be decreasing in the direction of any neighboring point in the feasible region of $g$.

* The two curves being tangent at a point is equivalent to the normal vectors being parallel at that point $p$, i.e. we write this as related by a "lagrangian multiplier", $\lambda$. The normal vectors of the contour of $f$ is the gradient, $\nabla f$, and the normal vector of the contour of $g$ is also the gradient, $\nabla g$. Thus the gradients are parallel to each other. The Lagrangian (also known as Lagrangian function or expression) can be written as:

\begin{equation}
L(p, \lambda) = f(p) - \lambda g(p)
\end{equation}

* The solution for the constrained problem is obtained when solving for the points where the partial derivatives of $L$ are zero. Solving for the stationary point of the Lagrangian indicates that the gradient $\nabla f(p)$ does not have to be 0 at the solution, but it should be contained in the subspace spanned by $\nabla g(p)$. This amounts to solving $n+1$ equations with $n+1$ unknowns ($n$ independent variables and $\lambda$). 

\begin{equation}
\nabla L(p, \lambda) = \nabla f(p) - \lambda \nabla g(p)= 0. 
\end{equation}

* Generalizing this to multiple constraints, $g_i(x)=0$, the solutions $p$ must satisfy $g_i(p)=0$. $\nabla f(p)$ while not necessarily zero, is entirely contained in the subspace spanned by the $\nabla g_i(p)$ normals. This translates to $\nabla f(p)$ being a linear combination of the gradients of each constraint $g_i(x)$, with weights $\lambda_i$.

\begin{equation}
\nabla f(x) = \sum_i\lambda_i\nabla g_i(x)
\end{equation}

* The Lagrangian for the multiple constraint case then becomes a function of $n+m$ variables ($x\in \mathbb{R}^n$ and $m$ of $\lambda_i \in \lambda$). Differentiating $L$ gives $n+m$ equations, which are solved by setting the partial derivative to zero. 

\begin{equation}
L(x, \lambda) = f(x) + \sum_i\lambda_ig_i(x)
\end{equation}

