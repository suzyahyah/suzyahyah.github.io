---
layout: post
title: "Modes of Convergence"
date: 2019-05-20
mathjax: true
status: [Instructional]
categories: [Statistics, Information Theory]
---

**What do we mean by convergence?**
Convergence refers to what happens to a sequence of random variables $X_1, X_2, \cdot X_n$.
Different types of convergence refer to different ways of defining what happens to this
sequence. Some types of convergences are stronger than others, i.e, convergence in one implies
the other but not vice versa.


**Why we care?**
In statistics we often want to estimate some population parameter with estimators. Because our
random variables are functions of sample data, we never actually get
to observe the population parameter, we only have an estimate. To get our statistical
guarantees, we need to know the convergence properties of our estimator for large sample statistics.

So our random variable $X$ is really a mapping from the sample space $S=\{s_1, s_2, \cdots, s_k\}$ to a real number. In particular, each $X_n$ is a function of $S$ to real numbers, and a sequence of random variables is a sequence of functions $X_n:S\rightarrow \mathbb{R}$. And our sequence is really $X_1(s_i), X_2(s_i), \cdots$

There are 4 modes of convergence we care about, and these are related to various limit
theorems.

* Convergence with probability 1
* Convergence in probability
* Convergence in Distribution

Finally, Slutsky's theorem enables us to combine various modes of convergence to say something
about the overall convergence.


#### **Convergence with Probability 1**

A sequence $X_1, X_2, \cdots $ converges to a random variable $X$ with probability 1 if for any
(fixed) point/sample $s \in S$, as $n \rightarrow \infty$, the sequence converges to the limit
value $X(s)$.

\begin{equation}
P( \\{s \in S: lim_{n\rightarrow \infty} X_n(s) = X(s) \\}) = 1
\end{equation}

**Why we care?** Convergence with Probability 1 reflects "strong consistency" and is seen in the strong law of large numbers.
This states that if random variables $X_1, X_2,...$ are iid with $\mathbb{E}\[X_1\]$, then the sample mean, $\frac{1}{n}\sum^n_{i=1} X_i$ will converge to a finite expected value $\mathbb{E}\[X\]=\mu$. This result is used in 

<u>Asymptotic equipartition property in Information Theory</u>

Directly analogous to the Law of Large Numbers, the AEP states that
$\frac{1}{n}log\frac{1}{p(X_1, X_2, \cdots, X_n)} \rightarrow H(X)$ as $n\rightarrow \infty$,
where $P(X_1, X_2, \cdots, X_n)$ is the probability of observing the sequence  $X_1, X_2, \cdots X_n$. The proof follows directly from the above expectation:

\begin{align}
-\frac{1}{n}logp(X_1, X_2, \cdots, X_n) &= \frac{1}{n}log\prod_{i}p(X_i) \\\
&= -\frac{1}{n}\sum_i logp(X_i) \\\
&= -\mathbb{E}(logp(X_i)), n \rightarrow \infty \\\
&= H(X)
\end{align}

This gives us a way to bound the probability of a "typical set" of sequences, because then
$p(X_1, X_2, \cdots, X_n)$ will be close to $2^{-nH(X)}$.




* Convergence in discrete Markov Chains
* Convergence with stochastic gradient descent

#### **Convergence in Probability**
A sequence of random variables convergences in probability to $X$,  $X_n \rightarrow X$ if $\forall \delta \geq 0$, as $n \rightarrow \infty$

\begin{equation}
lim_{n\rightarrow \infty} P(|X_n - X|)\geq \delta) = 0, \forall \delta \geq 0
\end{equation}

Note that convergence in probability is weaker than convergence with probability 1. 

**Why we care?** Convergence in probability is related to the consistency of estimators, by
weak law of large numbers. This states that for $n\rightarrow \infty$, the sequence of
$\bar{X}_1, \bar{X}_2, \cdots$ converges in probability to $\mu$. The average of a large number of iid random variables converges in probability to its expected value.

The difference between the strong and weak law, is that since the weak law is based on convergence *in* probability, there
is still a possibility that $|X_n-X|\geq \delta$ can happen, just that it becomes vanishingly
small as $n \rightarrow \infty$.

<u>A note about the Asymptotic Equipartition Property</u>

Sometimes we see the AEP appearing as a Convergence *in* probability. It all depends on where
the $lim_{n\rightarrow \infty}$ is, "inside" the Probability i.e., $P(lim_{n\rightarrow \infty} \cdots)$ or "outside" i.e, $\lim_{n\rightarrow \infty}P(\cdots)$. 


#### **Convergence in Distribution**

A sequence of random variables $X_1$, $X_2$, ... converges in distribution to a random variable
$X$, if for every $x \in \mathbb{R}$, where $F_n$ and $F$ are CDF of random variables $X_n$ and
$X$ respectively.

\begin{equation}
lim_{n \rightarrow \infty} F_n(x) \rightarrow F(x)
\end{equation}

**Why we care** Convergence in distribution is related to the Central Limit Theorem (CLT). The CLT states that the normalized average of a sequence of iid random variables converges in distribution to a standard normal distribution. 

\begin{equation}
\frac{\sum_{i=1}^n X_i - \mathbb{E}\[X_i\]}{\sigma \sqrt{n}} \rightarrow \mathcal{N}(0, 1),
n \rightarrow \infty
\end{equation}

If we can prove convergence in distribution to other common distributions (often based on the
Gaussian), this allows us to provide asymptotic confidence intervals of the sample statistic.
