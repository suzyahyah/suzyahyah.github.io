---
layout: post
title: "Calculus for Machine Learning"
date: 2014-04-02
mathjax: true
categories: [Math, Machine Learning]
---

### Model Preliminaries
Machine Learning often involves minimising a cost/objective function, which is a function that measures the error of our model, consisting of several parameters(variables). We use methods from *differential calculus* for finding the minimum of cost functions. (Or maximum of reward functions).

There are several ways to think about calculus
* the study of the relationship between variables and their rates of change.
* a set of tools for analysing the relationship between function and their inputs. Typically we want to find the parameter values which enable a function to best match the data.
* a set of tools for helping us navigate in high-dimensional spaces.

**Derivatives and functions** 
* The derivative a function is a measure of rate of change; it measures how much the value of function $f(x)$ changes when we change parameter $x$. Typically, we want to differentiate the dependent variables $f(x)$ or $y$, with respect to the independent variables.

* For single-variable functions, the derivative of $f$ with respect to $x$ is denoted by $f'(x)=\frac{df}{dx}(x)$, where the slope of the tangent line to the function $f$ at point $P(x, f(x))$ is given by:

\begin{equation}
\frac{df}{dx}(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}
\end{equation}

![Fig1](/assets/Calculus-slope.png)
*Image taken from Applied Calculus for the Managerial, Life and Social Sciences 8th ed*

* Without applying limits, the difference quotient $\frac{f(x+h)-f(x)}{h}$ measures the average (not point rate of change of $f(x)$ with respect to $x$ over the interval $\[x, x+h\]$.

* $\lim_{h\to0}$ is used to express $h$ which gets infinitely small but non-zero. We cannot have $h$ become 0 because division by 0 is undefined.
* The derivative of a function $f$ is itself a function $f'$ that gives the slope of the tangent line(slope) to the graph of $f$ at any point $(x, f(x))$, which we can use to calculate rate of change.

* As derivatives measure rate of change and represent the slope of the function, they can be used to determine intervals or points where the function is increasing or decreasing. If the derivative $f'(x)$ is positive at $x$, it means the function is increasing at $x$. If the derivative is negative at $x$, it means the function is decreasing at $x$.

* In the context of machine learning, we search for parameter values, $x$ along a family of functions $f$. i.e. when fitting a function, we want to find the parameter values by changing $x$ until we get a derivative that is close to or equal to 0. 


*High-school refresher - In order to maximise or minimise a function, we can set its derivative equal to zero, and solve for the parameters. This was easy because our functions were linear in terms of the input variable. However in ML it is common to have non-linear relationships, and functions that have hundreds of dimensions and it is not analytically tractable to obtain the closed-form solution. Thus, we need an iterative algorithm to solve.*

<br>
:
**Differentiability and Loss functions**
* In the real-world, we may encounter non-differentiable functions that do not have a derivative at certain values. These can occur for multiple reasons, primarily because there is a jump in $f(x)$, there is a sharp change in $f(x)$ as illustrated in the following:

![Fig2](/assets/Calculus-discontinuity.png)
*Image taken from Applied Calculus for the Managerial, Life and Social Sciences 8th ed*

* In a binary classification setting, the empirical loss $(0-1)$ is given by $h(x)\neq f(x)$ thatwhich is non-differentiable(and non-continuous). Thus we often use surrogate loss functions to make the errir differentiable with respect to the input parameters. Cross-entropy, mean-squared-error, logistic etc are functions that wrap around the true loss value to give a surrogate or approximate loss which is differentiable.

* This principle is also used when considering 'smooth' activation functions for neural networks and allows us to apply gradient descent.

* The significance of smoothness is that we can approximate it to be linear at different coordinate points.

*Note that although the loss function might be convex, the problem as a whole can still be non-convex, and there is no guarantee that the parameter values will converge to a global optimum.*


**Gradients, partial derivatives and gradient descent**
* Gradients are what we care about in the context of ML. Gradients generalises derivatives to multivariate functions. 

* The gradient of a multivariate input function is a vector with partial derivatives. Partial derivates is the derivative $\frac{\delta(f(x))}{\delta_{x_i}}$of one variable $x_i$ with respect to the others. This reflects the change in the function output when changing one variable and holding the rest constant. 

* Element $i$ of $\nabla_xf(x)$, is the partial derivative of $f$ with respect to $x_i$.
* For example the gradient, $\nabla f$ of function $f: \mathbb{R}^n \rightarrow  \mathbb{R}^1$ is:

\begin{equation}
 \nabla f = \[ \frac{\delta f(x_1, .. x_n)}{\delta(x_1)}, \frac{\delta f(x_1, .. x_n)}{\delta (x_2)}, ..,  \frac{\delta f(x_1, .. , x_n)}{\delta (x_n)} \]
\end{equation}

* Positive gradients $\nabla f(x)>0$ point in the direction of steepest ascent, while negative gradients, $\nabla f(x)<0$ point in the direction of steepest decent. Thus, if we wish to minimise a function such as a loss function, we can move in the direction of the negative gradient by calculating a 'change in $x$'(gradient descent step) as the gradient multiplied by a step-size $\alpha$.
\begin{equation}
x_{new} = x_{old} - \alpha \nabla f(x)
\end{equation}

* Todo: Derivation of the Gradient update step.

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

*  In  the context of neural networks, 

    - Let $h = f(g(x))$, $h: \mathbb{R}^n \rightarrow \mathbb{R}^m$ describes a transformation from a layer of $n$ input nodes to the next layer of $m$ nodes. $g(x)$ describes the standard linear transformation $g(x) = Wx+b$, and $f$ is a non-linear differentiable function such as tanh. $W$ is a $n\times m$ weight matrix which corresponds to the edges of the $n$ input nodes to $m$ output nodes. The Jacobian of $g$ with respect to $x$, is denoted by $J_x(g)$ and is equivalent to $W$ because of the linear relationship of $W$ and $x$. Note that each element $i, j$ in $J_x(g)$ represents how the output node $j$ changes with respect to changes in $i$.

    - The Jacobian may also arise in the differentiation of the non-linear function $f: \mathbb{R}^m \rightarrow \mathbb{R}^m$. To differentiate from the above notation, let $\theta = g(x)$. Let $f$ be the sigmoid function $f(\theta_i) = \frac{1}{1+exp(-\theta_i)}$ applied element wise to $\theta$. As the sigmoid is an element wise operation, the Jacobian is a diagonal matrix $J_\theta(f)$ whose $(i, i)$ entry is $f'(\theta_i)$, namely $\frac{1}{(1+exp(\theta_i)).(1+exp(-\theta_i))}$

    - It functions like a stacked gradient vector for $n$ input instances. 


**How to deal with discontinuity?**

**Natural base e and Softmax Function**
Why base e? "We prefer natural logs (logarithms with base e) because coefficients on the natural-log scale are directly interpretable as approximately proportional differences. With a coefficient of 0.06, a difference of 1 in x corresponds to an approximate 6% difference in y and so forth"

*[1] Andrew Gelman and Jennifer Hill (2007). Data Analysis using Regression and Multilevel/Hierarchical Models. Cambridge University Press: Cambridge; New York, pp. 60-61.*

*Exponentials are also used in Probabilistic ML for the exponential family but we will not go into that here*.

**Limits, delta-epsilon and theoretical guarantees**

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

* Eigenvectors/eigenvalues of the Hessian describe the directions of principal curvature and the amount of curvature in each direction.

* If the partial derivatives are continuous, the order of differentiation can be interchanged (Clairaut's theorem) so the Hessian matrix will be symmetric. In the context of deep learning, this is often the case because we force our functions to be continuous and differentiable. The Hessian can then be decomposed into a set of real eigenvalues and an orthogonal basis of eigenvectors.

* Intuitively, the local geometry of curvature is measured by the Hessian. Taking it further, the properties of theaeigenvalue, tell us something about the convexity of the function. For all $x\in \mathbb{R}^n$,
    
  * If H is positive definite $H \succ 0$ (all eigenvalues are $>0$), the quadratic problem takes the shape of a "bowl" in higher dimensions and is strictly convex (has only one global minimum). If the gradient at coordinates $x$ is 0, $x$ is at the global minimum.

  * If H is positive semi-definite $H\succeq 0$ (all eignenvalues are $>=0$) then the function is convex. If the gradient at coordinate $x$ is 0, $x$ is at a local minimum.

  * If H is negative definite $H\prec 0$ (all all eigenvalues are $<0$), the quadratic problem takes the shape of an inverted bowl in higher dimensions, and is strictly concave (has only one global maximum). If our gradient at coordinates $x$ is 0, $x$ is at the global maximum.

  * If H is negative semi-definite $H\preceq 0$(all eigenvalues are $<=0$) then the function is concave. If the gradient at coordinate $x$ is 0, $x$ is at a local maximum.
 
  *Note that the opposite doesn't hold in the above conditions. E.g, strict convexity does not imply that the Hessian everywhere is positive definite.*

  * If H is indefinite, (has both positive and negative eigenvalues at $x$), this implies that $x$ is both a local minimum and a local maximum. Thus $x$ is a saddle point for $f$.

  * The proof is given by Taylor series expansion. If $x_c$ is a stationary point, then $\nabla f(x_c)=0$. 

\begin{equation}
f(x) = f(x_c)+ \nabla f(x_c).(x-x_c)+ \frac{1}{2}(x-x_c)^TH(x-x_c) = f(x_c) + \frac{1}{2}(x-x_c)^TH(x-x_c)
\end{equation}
  
  * If $H$ is positive definite, then this expression evaluates to $f(x)>f(x_c)$ near $x_c$, thus $x_c$ is a local minimum. 


* The Hessian of an error function is often used in second-order optimization such as Newton's methods, as an alternative to vanilla gradient descent. 

<br>

**Taylor Series approximation and non-differentiability**
* Taylor series approximates a complicated function using a series of simpler polynomial functions that are often easier to evaluate. The key idea is to use a series of increasing powers to express complicated yet well-behaved (infinitely differentiable and continuous) functions.

* For univariate functions, the first-order polynomial approximates $f$ at point $P$ as a straight line tangent to $f$ at point $P$. The second-order polynomial approximates $f$ as a quadratic equation whose line passes through point $P$. Increasing powers of polynomial result in better approximations to complicated functions. 

![Fig1](/assets/Calculus-taylor.png)
*Image taken from Applied Calculus for the Managerial, Life and Social Sciences 8th ed*

* At any arbitary point $p$, we use our knowledge of what $f(p), f'(p), f'\'(p)$ etc looks like in order to approximate $f(x)$. The Taylor series approximation can be written as:

\begin{align}
f(x) = f(p) + f'(p)(x-p) + \frac{1}{2}f''(p)(x-p)^2 + ... + \frac{1}{n!}f^n(p)(x-p)^n + ...
\\\
f(x) = \sum_{n=0}^{\infty} \frac{1}{n!}f^n(p)(x-p)^n
\end{align}

* This can be equivalently expressed as $f(x_p+\Delta x)$, where $x_p$ is the point where the derivatives are evaluated, and $(x_p+\Delta x)$ is the new point which we wish to approximate. Then, 

\begin{equation}
f(x_p+\Delta x) = \sum_{n=0}^{\infty} \frac{1}{n!}f^n(x_p)(\Delta x)^n
\end{equation}

* Expected error of Taylor Series

* For multivariate functions, the Taylor series can be expressed in terms of the Jacobian and Hessian, which reflect the interaction of the first-order derivatives of $J$ and second-order derivatives of $H$ with the $\Delta X = \[\Delta x_1, \Delta x_2, ... \Delta x_n\]$

\begin{equation}
f(X_p+\Delta X) = f(X_p) + J_f\Delta X + \frac{1}{2}\Delta X.H_f\Delta X + ...
\end{equation}
* For multivariate functions, the truncated first-order taylor series approximation of a multivariate function $f$ at point $P$ is a *hyperplane* tangent to $f$ at point P. 

* Taylor series are used to minimise non-differentiable functions. We find a posiiton where we can differentiate the function, and use it to find an approximation of where the function will be at the minimum.  This often involves truncating Taylor series polynomials and can be thought of as a 'linearisation' (first-order) or quadratic approximation (second-order) of a function. 

<br>

**Newton's method, root finding, and optimization**

* Newton's method is an iterative method for approximating solutions (finding roots) to equations. Given a function $f$, and an initial guess $x_0$, Newton's method makes use of the derivative at $x_0$, $f'(x_0)$ to approximate a better guess $x_1$. 

![Fig1](/assets/Calculus-newton.png)
*Image taken from Applied Calculus for the Managerial, Life and Social Sciences 8th ed*


* Recall that the Taylor Series approximates $f(x)$ by polynomials of increasing powers, $f(x_p+\Delta x) = f(x_p)+f'(x_p)\Delta x + \frac{1}{2}f'\'(x_p)\Delta x^2 + ... $ To get an estimate of $x$ when $f(x)=0$, we can truncate and solve for the first-order Taylor polynomial, $f(x_p) + f'(x_p)\Delta x=0$.


* Let our initial estimate and the point where we evaluate the derivatives of $f$ be $x_0$, and $\Delta x = x_1-x_0$. Then, 

\begin{align}
f(x_0) + f'(x_0)\Delta x = 0
\\\
\Delta x = -\frac{f(x_0)}{f'(x_0)}
\\\
x_1 = x_0 - \frac{f(x_0)}{f'(x_0)}
\end{align}

* $x_1$ becomes our new estimate, and we can find the next update by $x_2 = x_1 - \frac{f(x_1)}{f'(x_1)}$. 


<br>
**Optimization: Newton's method, Taylor series, and Hessian Matrix**

* In optimization problems, we wish to solve for derivative $f'(x)=0$ to find stationary/critical points. Newton's method is applied to the derivative of a twice-differentiable function. The new estimate $x_1$ is now based on minimising a quadratic function approximated around $x_0$, instead of a linear function.

* For the single variable case, 

\begin{equation}
x_1 = x_0 - \frac{f'(x_0)}{f'\'(x_0)} 
\end{equation}

* For the multivariate case,

\begin{equation}
x_1 = x_0 - \[Hf(x_0)\]^{-1} \nabla f(x_0)
\end{equation}

![Fig1](/assets/Calculus-newton-optimization.png)
*Image taken from: http://netlab.unist.ac.kr/wp-content/uploads/sites/183/2014/03/newton_method.png*


* Second-order methods often converge much more quickly but it can be very expensive to calculate and store the Hessian matrix. In general, most people prefer quasi-newton methods to approximate the Hessian. These are first order methods which need only the value of the error function and its gradient with respect to the parameters. This can even be better than the true Hessian, because we can constrain the approximation to always be positive definite.


<br>

**Conjugate Gradients**

**Lagrange Multipliers**

<br><br>

{% highlight python %}
{% endhighlight %}

#### References ####
[WikipediaReference](https://wikipedia.org)

