---
layout: post
title: 'Conjugate Priors'
date: 2017-05-06
mathjax: true
status: Instructional
categories: [Bayesian Inference]
---

### Key Concepts
A prior is conjugate to the posterior $P(\pi\|D)$, if the likelihood $P(D\|\pi)$ is of the same family as the prior $p(\pi)$. In Bayesian Inference, this allows us to update the posterior easily with new observations or 'pseudocounts', and draw new model parameters from the updated posterior distribution.

### Model Preliminaries
* The posterior $P(\pi\|L)$ is the probability of the new model parameter given the data observed. By Bayes rule, the posterior is proportional to the likelihood * prior

\begin{equation}
P(\pi\|L) \propto P(L\|\pi)P(\pi)
\end{equation}

* The likelihood, $P(L\pi)$ models the probability of seeing Label $L$ given the parameters $\pi$ of the model. It is typically fixed under the generative assumptions of the model. E.g, for binary labels, we may assume a binomial distribution. For more than 2 classes, we may assume a multinomial distribution.

* For binomial distribution,
\begin{equation}
P(L\|pi)=\pi^{C_1}(1-\pi)^{C_0}
\end{equation}

* The prior, $P(\pi)$, represents our beliefs over the value of the model parameter. From Eq(1), the choice of $P(\pi)$ becomes important, as it changes the form of $P(L\|\pi)P(\pi)$. When the model is a binomial distribution, the beta distribution is a conjugate prior because the resulting posterior has the same family as the likelihood.

\begin{eqnarray}
P(\pi\|L; \gamma_{\pi_0}, \gamma_{\pi_1}) = P(L\|\pi).P(\pi; \gamma_{\pi_0}, \gamma_{\pi_1}) \nonumber
\\\
= \pi^{C_1}(1-\pi)^{C_0}.c\pi^{\gamma_{\pi_1}-1}(1-\pi)^{\gamma_{\pi_0}-1} \nonumber
\\\
\propto \pi^{C_1+\gamma_{\pi_1}-1}(1-\pi)^{C_0+\gamma_{\pi_0}-1}
\end{eqnarray}

where $P(\pi; \gamma_{\pi_0}, \gamma_{\pi_1}) = c.\pi^{\gamma_{\pi-1}}(1-\pi)^{\gamma_{\pi_0}-1}$ is the Beta distribution over the prior, with parameters $\gamma_{\pi_0}, \gamma_{\pi_1}$. Observe that the posterior is of the same form as the prior, but with additional counts $C_0$ and $C_1$ that come from the likelihood. This allows the observed evidence, to be added directly into the hyperparameters (sometimes referred to as pseudocounts).
