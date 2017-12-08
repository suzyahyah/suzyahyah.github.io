---
layout: post
title:  "Noise Contrastive Estimation"
date:   2017-07-01 21:09:09 +0800
mathjax: true
categories: jekyll update
---
### Key Concepts
* A language model is a probability distribution over sequence of words. It is a generative model that can be used to generate words based on its surrounding context (e.g previous words, or window of words)
<br><br>
* Noise Contrastive Estimation is a general parameter estimation technique for locally normalized language models.
<br><br>
* The "Noise" is a distribution that generates samples, for which a probabilistic binary classifier learns to distinguish from the real distribution. 

### Model Preliminaries

In statistical learning, we are often uncertain about the parameters $ \theta $ of our model. Under a Bayesian framework, we can learn the parameters $\theta$ of our model using probabilistic inference from data/observations.  Ultimately, we want \$ p(y\|\theta, x) \$, but instead of getting a single hard estimate on $ \theta $, we want to take the distribution \$ p(\theta \| x)\$ into account. We can do that by taking the expected value \$ \mathbb{E}[p(y\|\theta)]\$ which depends on \$ p(\theta \|x)\$. Since \$ \theta \$ is a continuous variable, the expected value is given by:

\begin{equation}
p(y|X, \theta) = \mathbb{E}[p(y\|X, \theta)] = \int p(y\|\theta)p(\theta\|X) d\theta
\end{equation}

Inference of \$ p(\theta\|X) \$ is generally untractable in real world problems. By Bayes Rule,

\begin{equation}
p(\theta\|X) = \frac{p(X\|\theta)p(\theta)}{\int p(X\|\theta)p(\theta)d(\theta)}
\end{equation}

Where the integral in the denominator is intractable. Hence we rely on sampling from a distribution that asymptotically follows \$ p(\theta\|X, y)\$ without having to explicitly calculate the integrals. Why does sampling work? 

If we sample N points of \$\theta\$ at random from the probability density \$ p(\theta\|X) \$, then 

\begin{equation}
\mathbb{E}[p(y\|\theta)]=\mathbb{E}[f(\theta)] = \lim_{N\rightarrow\infty}\frac{1}{N}\sum_{t=1}^{N}f(\theta^t)
\end{equation}

Different types of Markov Chain Monte Carlo algorithms are different ways to sample, such that the likelihood is proportional to the true distribution. 

The key idea is to sample \$\theta\$ proportional to \$p(\theta)\$, by transitioning between states (the "chain"). We sample by transitioning between parameter values with a transition probability. That is, \$\theta^{(t+1)}:=g(\theta^{(t)})\$, where \$g\$ is a transition function. "Markov" means that the next transition only depends on the previous state, i.e.

\begin{equation}
g(\theta^{(t+1)}\|\theta^{(t)})
\end{equation}
 
#### References
* [Resnik, P., & Hardisty, E. (2010). Gibbs sampling for the uninitiated](https://www.cs.umd.edu/~hardisty/papers/gsfu.pdf)
<br>
* [MCMC-Wikipedia](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
<br>
* [Emily Fox-UW/Coursera](://www.coursera.org/learn/ml-clustering-and-retrieval/lecture/T36G9/a-standard-gibbs-sampler-for-lda)
<br>
* [Nando de Freitas - UBC CPSC 540](https://www.youtube.com/watch?v=TNZk8lo4e-Q)
