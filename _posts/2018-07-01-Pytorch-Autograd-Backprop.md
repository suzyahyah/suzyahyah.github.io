---
layout: post
title: "PyTorch Automatic differentiation for non-scalar variables; Reconstructing the Jacobian"
date: 2018-07-01
mathjax: true
status: [Code samples, Instructional]
shape: dot
categories: [Calculus, PyTorch]
---

#### Introduction

PyTorch is a popular Deep Learning library which provides automatic differentiation for all operations on Tensors. It's in-built `output.backward()` function computes the gradients for all composite variables that contribute to the `output` variable. Mysteriously, calling `.backward()` only works on scalar variables. When called on vector variables, an additional 'gradient' argument is required. 

In the official PyTorch 0.4.0 tutorials, 

> If you want to compute the derivatives, you can call .backward() on a Tensor. If Tensor is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to backward(), however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.

<u>Scalar variables: specifying a gradient argument</u>

In most cases when training the neural network, `.backward()` is called by some *loss* or *cost* variable from which we wish to perform backpropagation of gradients down to the weight parameters in the neural network. 

In a basic single layer sigmoid network, $\hat{y}$ is the predicted output, $\sigma$ is the sigmoid function, $W$ and $b$ are weight parameters in the neural network to be learned via updating weights through something like gradient descent.

$$
\begin{align}
\hat{y} &= \sigma(W.x + b) \\\
losses &= (y - \hat{y})^2 \\\
avgLoss &= \frac{1}{n}\sum_{i}^{n}losses
\end{align}
$$

In the following example, a single training batch that we are feeding into the network is given by $X \in \mathbb{R}^{n\times m}$, where $n$ is the number of instances, and $m$ is the number of dimensions of one training instance. The target variable can take on either $0$ or $1$, $y \in \\{0,1\\}$, and the output of the neural network is given by $Y \in \mathbb{R}^{n}$

{% highlight python %}
torch.manual_seed(1)
x = torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.6, 0.8]])
y_true = torch.Tensor([[0, 1]])

W = Variable(torch.randn(1, 3), requires_grad=True)
b = Variable(torch.randn(1, 2), requires_grad=True)

y_pred = torch.nn.Sigmoid()(torch.mm(W, torch.transpose(x, 0, 1))+b)
losses = (y_true - y_pred)**2
avg_loss = losses.sum()/2
avg_loss.backward()
{% endhighlight %}

After backpropagation from `avg_loss`, the values of `W.grad` are $(1.00000e-02 * [[-2.6718, -3.2673, -3.8628]])$ and the values of `b.grad` are $ [[0.1481, -0.1038]]$.

We can then use these gradients to update the new value of $W$ and $b$ manually, or with in-built PyTorch modules like `torch.nn.optimize`.

Note that the gradient arguments here are implicit, it is simply torch.Tensor([1]). `avg_loss.backward()` is equivalent to `avg_loss.backward(torch.Tensor([1])`.

<u>Vector variables: specifying a gradient argument</u>

In the above examples, `avg_loss` is a scalar variable. If we attempted to backpropogate from `losses.backward()` this would throw the following: *<span style="color:red">RuntimeError:</span> grad can be implicitly created only for scalar outputs.*

**losses** is a vector which contains the squared-error loss for each of the $\hat{y}$'s. That is, $losses = [loss^1, loss^2]$. If we wanted to call `losses.backward()` to the same effect as `avg_loss.backward()`, we would need to provide the gradient of `losses` with respect to `avg_loss`, $\frac{\delta(avgLoss)}{\delta(losses)}$ as an argument in backward. By chain rule,

\begin{equation}
\frac{\delta(avgLoss)}{\delta(W)} = \frac{\delta(avgLoss)}{\delta(losses)} \times \frac{\delta(losses)}{\delta(W)} 
\end{equation}

Note that `W.grad` is equivalent to $\frac{\delta(avgLoss)}{\delta(W)}$.

This translates to `losses.backward(torch.Tensor([0.5, 0.5]))`, where the input argument, $\frac{\delta(avgLoss)}{\delta(losses)}$ can be obtained by:

$$
\begin{align}
avgLoss &= \frac{1}{2} [loss^1, loss^2] \\\
\frac{\delta(avgLoss)}{\delta(losses)} &= [\frac{1}{2}, \frac{1}{2}] 
\end{align}
$$

Running the code block with the gradient input gives the same `W.grad` and `b.grad` as the commented out code. The values of `W.grad` are $(1.00000e-02 * [[-2.6718, -3.2673, -3.8628]])$ and the values of `b.grad` are $ [[0.1481, -0.1038]]$.

{% highlight python %}
...
losses = (y_true-y_pred)**2
# avg_loss = losses.sum()/2
# avg_loss.backward()
losses.backward(torch.FloatTensor([0.5, 0.5])
{% endhighlight %}

<u>Reconstructing the Jacobian</u>

Sometimes we may wish to obtain the Jacobian matrix of partial derivatives to capture the rate of change of each component of the output with respect to the input vector. The following example shows the Jacobian matrix of weights derivatives with respect to losses.

\begin{equation}
\frac{\delta(losses)}{\delta(W)} = 
\begin{bmatrix}
\frac{\delta loss^1}{\delta w_1} & \cdots & \frac{\delta loss^n}{\delta w_1}
\\\
\vdots & \ddots & \vdots
\\\
\frac{\delta loss^1}{\delta w_m} & \cdots & \frac{\delta loss^n}{\delta w_m}
\end{bmatrix}
\end{equation}

Because `.backward()` requires gradient arguments as inputs and performs a matrix multiplication internally to give the output (see eq 4), the recommmended way to obtain the Jacobian is by feeding in a gradient input which accounts for that specific row of the Jacobian. This is done by providing a mask for the specific dimension in the gradient vector, i.e. the gradient vector has a non-zero value on that dimension, and 0 everywhere else. 

{% highlight python %}
...
losses = (y_true - y_pred)**2
#loss_sum = losses.sum()/2
#loss_sum.backward()

jacobian = torch.zeros((x.shape[0], W.shape[1]))

losses.backward(torch.FloatTensor([0.5, 0], retain_variables=True)
jacobian[0,:] = W.grad.data
W.grad.data.zero_()

losses.backward(torch.FloatTensor([0, 0.5], retain_variables=True)
jacobian[1,:] = W.grad.data
{% endhighlight %}

The `jacobian` $\frac{\delta(losses)}{\delta W}$ is now $(1.00000e-02 * [[-4.1526, -6.2289, -8.3053], [1.4808, 2.9616, 4.4424]])$. 

Summing the two rows element-wise of the Jacobian would simply result in `W.grad`. The masked gradient that we input to `losses.backward(..)` allows us to generate individual rows of the Jacobian matrix instead of summing all the rows in the matrix multiplication operation of eq 4.

#### References ####
[PyTorch Forums](https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059/5)

