---
layout: post
title: "Equivalence of constrained and unconstrained form for Ridge Regression"
date: 2018-07-20
mathjax: true
status: [Instructional]
categories: [Optimization]
---

### Introduction

Linear regression least squares estimate has low bias but high variance (overfitting) when there are many correlated variables in a linear regression model. Model coefficients can get very large as a large positive coefficient on one variable can be cancelled by a large negative coefficient on another variable. A natural regularization for the model is to try and shrink the coefficients of the model by penalising large values.

#### Unconstrained Form of Ridge Regression
The ridge regression shrinks the coefficients by imposing a penalty in the objective function:

\begin{equation}
argmin_{\beta} (\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^{p}x_{ij}\beta_j)^2 + \lambda \sum_{j=1}^{p}\beta_j^2)
\end{equation}

Here $\lambda$ is a hyperparameter on how much to penalize the size of the $\beta$ coefficients. Applying the ridge regression penalty has the effect of shrinking the estimates towards zero - introducing bias but reducing the variance of the estimate.

The solution to the ridge objective is not equivariant under scaling of the inputs, and hence we standardize the inputs. Thus we do not include the intercept $\beta_0$ in the penalty term after centering inputs around 0. 

\begin{equation}
argmin_{\beta} (\sum_{i=1}^N(y_i - x_i \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2
\end{equation}

In matrix form, this can be written as:

\begin{equation}
argmin_{\beta} \|\|y-X\beta\|\|_2^2 + \lambda \|\|\beta\|\|_2^2
\end{equation}

<br><br>
#### Constrained Form of Ridge Regression

The familiar ridge regression objective function can be formulated as an equivalent constrained optimization problem. This makes the size constraint on the coefficients explicit, and prevents the coefficients from exploding in scenarios of high collinearity.

\begin{equation}
argmin_{\|\|\beta\|\|_2^2 \leq c} \|\|Y - X\beta \|\|_2^2
\end{equation}

The above can be solved by the KKT multiplier method, which minimizes a function subject to inequality constraints. First we rewrite it to the unconstrained form. The unconstrained form for the above problem is given by:

\begin{equation}
\|\|Y-X\beta\|\|_2^2 + v(\|\|\beta\|\|_2^2 - c)
\end{equation}

<br><br>
#### The equivalence of the constrained and unconstrained form

We can show the equivalence of the constrained and unconstrained form if we can show that $\beta$ values obtained are the same in both forms. This can be shown when there is a one-to-one correspondence between the parameters $\lambda$ in eq (3) and $v$ in eq (5). 

The first KKT condition (stationarity) says that the gradient with respect to $\beta$ of the lagrangian equals to 0.

By expanding eq(5), we can see that eq (3) is equivalent to eq (5), plus a constant that does not depend on $\beta$. 

\begin{equation}
\|\|Y-X\beta\|\|_2^2 + v(\|\|\beta\|\|_2^2) - v(c)
\end{equation}

Thus solving for the derivative of eq (5) is thus equivalent to solving for the derivate of eq (3) when $\lambda = v$ because of the term that does not depend on $\beta$.

The second KKT condition (complementarity) says that 

\begin{equation}
v(\|\|\beta\|\|_2^2 -c) = 0
\end{equation}

Thus the $\beta$ values that minimise eq (5) must be $c=\|\|\beta\|\|_2^2$. Thus the values of $\beta$ in eq (2) are constrained by $\|\|\beta\|\|^2 \leq c$ in eq (4). 

### References
[Stack Overflow - Why are additional constraint and penalty terms equivalent](https://math.stackexchange.com/questions/335306/why-are-additional-constraint-and-penalty-term-equivalent-in-ridge-regression)

[Elements of Statistical Learning, Section 3.4.1 Ridge Regression](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)

