---
layout: post
title: "Gradients, partial derivatives, directional derivatives, and gradient descent"
date:  2018-04-03
mathjax: true
status: Ongoing, Instructional
categories: [Calculus, Machine Learning, Optimization]
---

### Model Preliminaries
**Gradients and partial derivatives**
* Gradients are what we care about in the context of ML. Gradients generalises derivatives to multivariate functions. 

* The gradient of a multivariate input function is a vector with partial derivatives. Partial derivates is the derivative $\frac{\delta(f(x))}{\delta_{x_i}}$of one variable $x_i$ with respect to the others. This reflects the change in the function output when changing one variable and holding the rest constant. 

* For example the gradient, $\nabla f(x)$ of function $f(x): \mathbb{R}^n \rightarrow  \mathbb{R}^1$, where the $i$-th element of $\nabla f(x)$ is the partial derivative of $f$ with respect to $x_i$ is:

\begin{equation}
 \nabla f(x) = \[ \frac{\delta f(x_1, .. x_n)}{\delta(x_1)}, \frac{\delta f(x_1, .. x_n)}{\delta (x_2)}, ..,  \frac{\delta f(x_1, .. , x_n)}{\delta (x_n)} \]
\end{equation}

* To minimise the function, we would like to move $x$ in the direction of steepest descent. In the univariate case this is straightforward as we either increase or decrease values of $x_1$ for that single variable. 

<br>

**Directional Derivatives and Gradient Descent**

* We can generalise the gradient at a point as being the direction of steepest descent to the multivariate case - where we move in the opposite direction of the positive gradient. The reason is because the directional derivative which maximises descent of the function, is in the same direction as the gradient. 

* Let the unit vector $\vec{v}$ represent the direction of which we would like to move in. $\vec{v}$ is the direction of steepest descent, when the directional derivative $D_{\vec{v}}f(x)$ is maximised. This means finding $\vec{v}$, where the rate of change is maximised for $f(x)$ in the direction of $\vec{v}$ at that point $x$.

* It can be shown that the directional derivative, i.e. how fast $f$ is changing along a particular vector $\vec{v}$, at the point $(x, f(x))$, can be expressed as the following dot product:

\begin{equation}
D_{\vec{v}}f(x) = (\nabla f(x)).\vec{v}
\end{equation}

* This can be rewritten as $D_{\vec{v}}f(x) = \|\|\nabla f(x)\|\| \|\|\vec{v}\|\| cos(\theta)$, where $\theta$ is the angle between the two vectors. The directional derivative is maximised when $cos(\theta)=1$ and $\theta=0$, and $\vec{v}$ points in the same direction of the gradient. It is minimised when $cos(\theta)=-1$ and $\theta=180^{\circ}$, and $\vec{v}$ points in the opposite direction of the gradient.  

* To summarise, positive gradients $\nabla f(x)>0$ point in the direction of steepest ascent, while negative gradients, $\nabla f(x)<0$ point in the direction of steepest decent. Thus, if we wish to minimise a function such as a loss function, we can move in the direction of the negative gradient by calculating a 'change in $x$'(gradient descent step) as the gradient multiplied by a step-size $\alpha$.
\begin{equation}
x_{new} = x_{old} - \alpha \nabla f(x)
\end{equation}

#### References ####

