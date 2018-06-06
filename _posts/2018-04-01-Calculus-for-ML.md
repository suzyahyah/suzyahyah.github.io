---
layout: post
title: "Calculus for Machine Learning"
date: 2018-04-01
mathjax: true
status: [Instructional]
categories: [Math, Machine Learning, Calculus]
---

### Model Preliminaries
Machine Learning often involves minimising a cost/objective function, which is a function that measures the error of our model, consisting of several parameters(variables). We use methods from *differential calculus* for finding the minimum of cost functions. (Or maximum of reward functions).

There are several ways to think about calculus
* the study of the relationship between variables and their rates of change.
* a set of tools for analysing the relationship between function and their inputs. Typically we want to find the parameter values which enable a function to best match the data.
* a set of tools for helping us navigate in high-dimensional spaces.

The following posts link mathematical concepts in calculus with Optimization and Machine Learning.

1. [Derivatives and functions]({{ site.baseurl }}{% link _posts/2018-04-02-Derivatives-and-Functions.md %})
2. [Gradients, partial derivatives, directional derivatives and gradient descent]({{ site.baseurl }}{% link _posts/2018-04-03-Gradient-and-Gradient-Descent.md %})
3. [Jacobian, Chain rule and backpropagation]({{ site.baseurl }}{% link _posts/2018-04-04-Jacobian-and-Backpropagation.md %})
4. [Hessian, second derivatives, function convexity, saddle points]({{ site.baseurl }}{% link _posts/2018-04-05-Hessian-Second-Derivatives.md %})
5. [Taylor Series, Newton's method]({{ site.baseurl }}{% link _posts/2018-04-06-Taylor-Series-Newtons-Method.md %})
6. [Lagrange Multipliers and Constrained Optimization]({{ site.baseurl }}{% link _posts/2018-04-07-Lagrange-Multiplier.md %})
7. Limits, delta-epsilon and theoretical guarantees 
8. Conjugate Gradients
9. Discontinuity



