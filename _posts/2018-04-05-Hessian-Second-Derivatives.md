---
layout: post
title: "Hessian, second order derivatives, convexity, and saddle points"
date: 2018-04-05
mathjax: true
status: [Instructional]
categories: [Calculus]
---
**Hessian matrix: Second derivatives and Curvature of function**
* The Hessian is a square matrix of second-order partial derivatives of a scalar-valued function, $f:\mathbb{R}^n \rightarrow \mathbb{R}$. Let the second-order partial derivative $f\'\'(x)$, be the partial derivative of the gradient $f'(x)$. Then the Hessian, $H = f\'\'(x) \in \mathbb{R}^{n\times n}$. 
* Recall that the gradient of $f:\mathbb{R}^n \rightarrow \mathbb{R}$ can be written as 
\begin{equation}
f'(x) = \[\frac{\delta f}{\delta x_1}, \frac{\delta f}{\delta x_2}, \cdots, \frac{\delta f}{\delta x_n} \]\end{equation}
* Taking the second-order  derivative of $f(x)$ means we look at how each ith partial derivative, $\frac{\delta f}{\delta x_i}$ affects all other partial derivatives. It is equivalent to taking the Jacobian of $f'(x)$. Let $\delta^2 f$ be $(\delta f) (\delta f)$ in the following:
\begin{equation}
H = 
\begin{bmatrix}
\frac{\delta^2 f}{(\delta x_1)(\delta x_1)} & \cdots & \frac{\delta^2 f}{(\delta x_1)(\delta x_n)}
\\\
\vdots & \ddots & \vdots
\\\
\frac{\delta^2 f}{(\delta x_n)(\delta x_1)} & \cdots & \frac{\delta^2 f}{(\delta x_n)(\delta x_n)}
\end{bmatrix}
\end{equation}

* Intuitively, the second derivative of a function, is the rate of change of the slope of the function. This is analagous to it's *acceleration*. 

* The Hessian matrix of a function is the rate at which different input dimensions accelerate with respect to each other. If $H_{1,1}$ is high, it means there is a high acceleration in dimension 1. If $H_{1,2}$ is high, then we accelerate simultaneously in both dimensions. If $H_{1,2}$ is negative, then as we accelerate in the first dimension, we decelerate in the second dimension. 

* In the context of gradient descent in Machine Learning, the second derivative measures curvature of the loss function, as opposed to the slope(gradient) at a single coordinate. Information about the Hessian can therefore help us take appropriate gradient steps towards the minima.

  * Suppose we have a function $f$, where the gradient $f'(x)$ is negative. We know that the function is sloping downwards, however we dont know whether it is (1)sloping downwards more steeply, (2)sloping downwards at the same rate, or (3) sloping downwards less and potentially going upwards. These can be informed by the second derivative, which is the gradient of the gradient. 

  * In scenario (1), if the second derivative is negative, then the function is accelerating downwards, and the cost function will end up decreasing more than the gradient multiplied by step-size.

  * In scenario (2), if the second derivative $f''(x)$ is 0. The function is a straight line with no curvature because acceleration/decceleration is 0. Then a step-size of $\alpha$ along the negative gradient will decrease the $f(x)$ by $c.\alpha$. 

  * In scenario (3), if the second derivative is positive, then the function is deccelerating and eventually accelerates upward. Thus if the $\alpha$ too large, gradient descent might end up with coordinates that result in greater cost. If we are able to calculate the second derivative, then we can control the $\alpha$ to reduce oscillation around the local minima.

![Fig1](/assets/Calculus-curvature.png)
*Image taken from [Deep Learning Book Chapt 4](https://w.deeplearningbook.org/contents/numerical.html) on Numerical Computation*


<br>
**Hessian Matrix: Eigenvalues, Convexity and Saddle Points**

* Eigenvectors/eigenvalues of the Hessian describe the directions of principal curvature and the amount of curvature in each direction. Intuitively, the local geometry of curvature is measured by the Hessian.
 
* If the partial derivatives are continuous, the order of differentiation can be interchanged (Clairaut's theorem) so the Hessian matrix will be symmetric. In the context of deep learning, this is often the case because we force our functions to be continuous and differentiable. The Hessian can then be decomposed into a set of real eigenvalues and an orthogonal basis of eigenvectors.

* In the context of Machine Learning optimization, after we have converged to a critical point using gradient descent, we need to examine the eigenvalues of the Hessian to determine whether it is a min, max, or saddle point. Examining the properties of the eigenvalues tell us something about the convexity of the function. For all $x\in \mathbb{R}^n$,
    
  * If H is positive definite $H \succ 0$ (all eigenvalues are $>0$), the quadratic problem takes the shape of a "bowl" in higher dimensions and is strictly convex (has only one global minimum). If the gradient at coordinates $x$ is 0, $x$ is at the global minimum.

  * If H is positive semi-definite $H\succeq 0$ (all eignenvalues are $>=0$) then the function is convex. If the gradient at coordinate $x$ is 0, $x$ is at a local minimum.

  * If H is negative definite $H\prec 0$ (all all eigenvalues are $<0$), the quadratic problem takes the shape of an inverted bowl in higher dimensions, and is strictly concave (has only one global maximum). If our gradient at coordinates $x$ is 0, $x$ is at the global maximum.

  * If H is negative semi-definite $H\preceq 0$(all eigenvalues are $<=0$) then the function is concave. If the gradient at coordinate $x$ is 0, $x$ is at a local maximum.
 
  *Note that the opposite doesn't hold in the above conditions. E.g, strict convexity does not imply that the Hessian everywhere is positive definite.*

  * If H is indefinite, (has both positive and negative eigenvalues at $x$), this implies that $x$ is both a local minimum and a local maximum. Thus $x$ is a saddle point for $f$.

  * The style of proofs is given by Taylor series expansion and the fact that if all function values lie above the supporting hyperplane, the function is convex. Approximating the function with a second order polynomial:

\begin{equation}
f(x) = f(x_c)+ \nabla f(x_c).(x-x_c)+ \frac{1}{2}(x-x_c)^TH(x-x_c) 
\end{equation}

To show convexity, $H$ is PSD means that $d^THd \geq 0, \forall d \in mathbb{R}^n$. Hence we get $f(x) \geq f(x_c)
+ \nabla f(x_c)(x-x_c)$. Since $f(x_c) + \nabla f(x_c)^T(x-x_c)$ is the tangent hyperplane, if
  all $f(x)$ is above this hyperplane it indicates that the function is convex.

If $x_c$ is a stationary point, then $\nabla f(x_c)=0$, and the equation becomes $f(x_c) + \frac{1}{2}(x-x_c)^TH(x-x_c)$
  
  * If $H$ is positive definite, then this expression evaluates to $f(x)>f(x_c)$ near $x_c$, thus $x_c$ is a local minimum. 


* The Hessian of an error function is often used in second-order optimization such as Newton's methods, as an alternative to vanilla gradient descent. 

