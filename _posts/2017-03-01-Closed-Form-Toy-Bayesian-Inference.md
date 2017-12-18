---
layout: post
title:  "Closed form Bayesian Inference for Bernouli distributions"
date:   2017-03-01 21:09:09 +0800
mathjax: true
categories: jekyll update
---
### Key Concepts
* Bayesian Inference is the use of Bayes theorem to estimate parameters of an unknown probability distribution.

* A Bernoulli distribution is the discrete probability distribution of a random variable \$ X \in \\{0, 1\\}\$ for a single trial. Because there are only two possible outcomes, $P(X=1) = 1-P(X=0)$. 

* A Binomial distribution is the probability of observing $k$ "successes", i.e. no. of times $k=1$ given a sequence of $n$ Bernoulli trials. 

* A Beta distribution is the continuous probability distribution of a random variable $\theta \in [0, 1]$. It is the conjugate prior probability distribution for the Bernoulli and Binomial distributions.

### Model Preliminaries

* In Bayes Theorem, we have some phenomenon that we are trying to represent with a statistical model. 
The model represents the data-generating process with a probability distribution $p(x\|\theta)$, 
where $\theta$ are the parameters of this model. We start with some prior belief over the values of the parameters, $\pi(\theta)$, and wish to update out beliefs after observing data $D_n=\\{x_1, x_2, .. x_n \\}$. By Bayes rule, we can perform the update by:

\begin{equation}
p(\theta\|D_n) = \frac{p(D_n\|\theta)\pi(\theta)}{p(D_n)}
\end{equation}

* $X$ follows a Bernouli distribution with parameter $p$ can be written as $X\sim Ber(p)$. For each value that $X$ can take on, $f(x\|p) = p^x(1-p)^x$, where $x$ is either 1 or 0.

* $K$ follows a Binomial distribution with parameters $n$ and $p\in[0,1]$ can be written as $K \sim Bin(n, p)$. Here, $K$ is a random variable reflecting the number of "successes(1s)". The probability of getting exactly $k$ successes in $n$ trials is given by the probability mass function $P(K=k)={n \choose k} p^k (1-p)^{n-k}$

{%highlight python%}
K = scipy.stats.binom(n, p)
K.pmf(k)

{%endhighlight%}

* Beta distribution has the form ...

* if we assume that $\theta$ comes from a Beta distribution, then 

\begin{equation}
p(\theta\|\alpha, \beta, D_n) \propto p(D_n\|\theta)p(\theta\|\alpha, \beta)
\end{equation}

\begin{equation}
\propto \theta^k(1-\theta)^{n-k} \theta^{\alpha-1}(1-\theta)^{\beta-1}
\end{equation}

\begin{equation}
= \theta^{k+\alpha-1}(1-\theta)^{n-k+\beta+1}
\end{equation}

* with normalization, ...

### Updating the posterior given observations

* $\theta$ follows a Beta distribution with parameters $\alpha$ and $\beta$ can be written as $\theta \sim Beta(\alpha, \beta)$. We can perform parameter updates on the posterior by the following $\theta\|D_n \sim Beta(\alpha_{new}, \alpha_{old})$, where $\alpha_{new} = (k+\alpha_{old}), \beta_{new} = (n-k+\beta_{old})$
* Assume a uniform distribution (we have not observed any outcomes), then $\alpha=1, \beta=1$.
* Next, if we observe 10 trials with 50% likelihood of success, then

{%highlight python%}
a, b = 1, 1
n = 10
k = 0.5*n

thetas = np.linspace(0, 1, 500) #generate 500 theta values from 0 to 1 for plotting
prior = scipy.stats.beta(a, b).pdf(thetas)

k=k+a
b = n-k+b

posterior10 = scipy.stats.beta(a, b).pdf(thetas)
plt.plot(thetas, prior, label="Prior")
plt.plot(thetas, posterior10, label="Posterior_10n")
display_plot(plt)
{%endhighlight%}

* if we observe another 100 trials with random likelihood of success, then

{%highlight python%}
n=100
k = np.floor(random.random()*n)

a = k+a
b = n-k+b
posterior100 = scipy.stats.beta(a, b).pdf(thetas)
plt.plot(thetas, posterior100, label="Posterior_100n")
display_plot(plt)
{%endhighlight%}

{%highlight python%}
def display_plot(plt):
  plt.xlabel(r'$\theta$', fontsize=14)
  plt.ylabel('Density', fontsize=14)
  plt.legend()
  plt.show()
{%endhighlight%}

